# backend/main.py
import io
import json
import base64
import logging
import asyncio
import queue
import threading
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from config import (
    IMAGES_DIR,
    MODELS_DIR,
    COMPACTED_MODELS_DIR,
    MODEL_HF_NAME,
    NUM_LABELS,
    CLASS_NAMES,
    CLASS_FOLDER_NAMES,
    MODEL_DISPLAY_NAMES,
    DEFAULT_DISCARD_RATIO,
    DEFAULT_HEAD_FUSION,
    DATASET_FOLDER,
    DATASET_SPLITS,
    DEVICE,
    BASELINE_MODEL_ID,
    BASELINE_PARAM_COUNT,
)
from rollout import generate_patch_map, build_overlay, build_raw_heatmap
from compacted_model import load_compacted_model
from benchmark import benchmark_model

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="ViT Attention Rollout Explorer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the entire assets/images tree as static files
app.mount("/static/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jpeg"}

def dataset_root() -> Path:
    if DATASET_FOLDER:
        return IMAGES_DIR / DATASET_FOLDER
    return IMAGES_DIR


def is_nested() -> bool:
    return DATASET_FOLDER is not None


def get_class_dirs() -> list:
    root = dataset_root()
    seen = {}  # folder_name -> first path found

    if is_nested():
        for split in DATASET_SPLITS:
            split_dir = root / split
            if not split_dir.exists():
                continue
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir() and cls_dir.name not in seen:
                    seen[cls_dir.name] = cls_dir
    else:
        for cls_dir in sorted(root.iterdir()):
            if cls_dir.is_dir():
                seen[cls_dir.name] = cls_dir

    return list(seen.values())


def get_images_for_class(class_id: str) -> list:
    root = dataset_root()
    results = []
    seen_ids = set()

    if is_nested():
        for split in DATASET_SPLITS:
            cls_dir = root / split / class_id
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in EXTS and f.is_file():
                    unique_id = f"{split}__{f.stem}"
                    if unique_id not in seen_ids:
                        seen_ids.add(unique_id)
                        rel = f.relative_to(IMAGES_DIR).as_posix()
                        results.append({
                            "id": unique_id,
                            "filename": f.name,
                            "url": f"/static/images/{rel}",
                            "split": split,
                            "class_id": class_id,
                        })
    else:
        cls_dir = root / class_id
        if cls_dir.exists():
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in EXTS and f.is_file():
                    rel = f.relative_to(IMAGES_DIR).as_posix()
                    results.append({
                        "id": f"flat__{f.stem}",
                        "filename": f.name,
                        "url": f"/static/images/{rel}",
                        "split": "all",
                        "class_id": class_id,
                    })

    return results


_image_index = None

def build_image_index() -> dict:
    global _image_index
    if _image_index is not None:
        return _image_index

    log.info("Building image index...")
    index = {}
    root = dataset_root()

    if is_nested():
        for split in DATASET_SPLITS:
            split_dir = root / split
            if not split_dir.exists():
                continue
            for cls_dir in split_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                for f in cls_dir.iterdir():
                    if f.suffix.lower() in EXTS and f.is_file():
                        img_id = f"{split}__{f.stem}"
                        index[img_id] = f
    else:
        for item in root.iterdir():
            if item.is_dir():
                for f in item.iterdir():
                    if f.suffix.lower() in EXTS and f.is_file():
                        index[f"flat__{f.stem}"] = f
            elif item.suffix.lower() in EXTS:
                index[f"flat__{item.stem}"] = item

    _image_index = index
    log.info(f"Image index built: {len(index)} images.")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# In-memory caches
# ─────────────────────────────────────────────────────────────────────────────

_processor = None
_model_cache = {}
_pil_cache = {}
_baseline_bench_model = None   # cached baseline for benchmarking (loaded once)

log.info(f"Using device: {DEVICE}")


def get_processor():
    global _processor
    if _processor is None:
        log.info(f"Loading processor from {MODEL_HF_NAME}")
        _processor = ViTImageProcessor.from_pretrained(MODEL_HF_NAME)
    return _processor


def get_model(stem: str):
    if stem in _model_cache:
        return _model_cache[stem]

    path = MODELS_DIR / f"{stem}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    log.info(f"Loading model: {stem}")
    model = ViTForImageClassification.from_pretrained(
        MODEL_HF_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        output_attentions=True,
        attn_implementation="eager",
    )
    state_dict = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    _model_cache[stem] = model
    log.info(f"Model '{stem}' loaded and cached.")
    return model


def get_image_pil(image_id: str):
    if image_id in _pil_cache:
        return _pil_cache[image_id]
    index = build_image_index()
    if image_id not in index:
        raise FileNotFoundError(f"Image not found: {image_id}")
    img = Image.open(index[image_id]).convert("RGB")
    _pil_cache[image_id] = img
    return img


def stem_to_display(stem: str) -> str:
    if stem in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[stem]
    return stem.replace("_", " ").replace("-", " ").title()


def image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/models")
def list_models():
    files = sorted(MODELS_DIR.glob("*.pth"))
    result = [{"id": f.stem, "name": stem_to_display(f.stem)} for f in files]
    return {"models": result}


@app.get("/api/classes")
def list_classes():
    class_dirs = get_class_dirs()
    result = []
    for cls_dir in class_dirs:
        cid = cls_dir.name
        result.append({
            "id": cid,
            "name": CLASS_FOLDER_NAMES.get(cid, cid),
        })
    return {"classes": result}


@app.get("/api/classes/{class_id}/images")
def list_class_images(
    class_id: str,
    limit: int = Query(500, ge=1, le=5000),
):
    images = get_images_for_class(class_id)
    if not images:
        raise HTTPException(404, f"No images found for class: {class_id}")
    return {"class_id": class_id, "images": images[:limit]}


@app.get("/api/rollout")
def compute_rollout(
    model_id: str = Query(...),
    image_id: str = Query(...),
    discard_ratio: float = Query(DEFAULT_DISCARD_RATIO, ge=0.0, le=0.5),
    head_fusion: str = Query(DEFAULT_HEAD_FUSION, regex="^(mean|max)$"),
    alpha: float = Query(0.55, ge=0.0, le=1.0),
    view: str = Query("overlay", regex="^(overlay|heatmap|both)$"),
):
    try:
        model = get_model(model_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    try:
        img_pil = get_image_pil(image_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    processor = get_processor()
    inputs = processor(images=img_pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        patch_map, logits = generate_patch_map(
            model, pixel_values,
            discard_ratio=discard_ratio,
            head_fusion=head_fusion,
        )

    probs = torch.softmax(logits[0], dim=-1).cpu().tolist()
    pred_idx = int(torch.argmax(logits[0]).item())
    pred_label = CLASS_NAMES[pred_idx] if CLASS_NAMES else f"Class {pred_idx}"
    img_display = img_pil.resize((224, 224), Image.LANCZOS)

    response = {
        "model_id": model_id,
        "model_name": stem_to_display(model_id),
        "image_id": image_id,
        "prediction": {
            "class_index": pred_idx,
            "class_name": pred_label,
            "confidence": round(probs[pred_idx] * 100, 1),
            "top5": sorted(
                [
                    {
                        "class_index": i,
                        "class_name": CLASS_NAMES[i] if CLASS_NAMES else f"Class {i}",
                        "confidence": round(p * 100, 1),
                    }
                    for i, p in enumerate(probs)
                ],
                key=lambda x: -x["confidence"],
            )[:5],
        },
    }

    if view in ("overlay", "both"):
        overlay = build_overlay(img_display, patch_map, alpha=alpha)
        response["overlay_b64"] = image_to_b64(overlay)

    if view in ("heatmap", "both"):
        heatmap = build_raw_heatmap(img_display, patch_map)
        response["heatmap_b64"] = image_to_b64(heatmap)

    return JSONResponse(response)


@app.get("/api/rollout/batch")
def compute_rollout_batch(
    model_ids: str = Query(...),
    image_id: str = Query(...),
    discard_ratio: float = Query(DEFAULT_DISCARD_RATIO),
    head_fusion: str = Query(DEFAULT_HEAD_FUSION),
    alpha: float = Query(0.55),
    view: str = Query("overlay"),
):
    stems = [s.strip() for s in model_ids.split(",") if s.strip()]
    results = []
    for stem in stems:
        try:
            result = compute_rollout(
                model_id=stem,
                image_id=image_id,
                discard_ratio=discard_ratio,
                head_fusion=head_fusion,
                alpha=alpha,
                view=view,
            )
            results.append(json.loads(result.body))
        except HTTPException as e:
            results.append({"model_id": stem, "error": e.detail})

    return JSONResponse({"results": results})


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "models_loaded": list(_model_cache.keys()),
        "images_indexed": len(_image_index) if _image_index else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes — compacted models list + benchmark SSE (new)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/compacted_models")
def list_compacted_models():
    """Return all .pth files available in assets/compacted_models/."""
    if not COMPACTED_MODELS_DIR.exists():
        return {"models": []}
    files = sorted(COMPACTED_MODELS_DIR.glob("*.pth"))
    result = [
        {"id": f.name, "name": MODEL_DISPLAY_NAMES.get(f.stem, stem_to_display(f.stem))}
        for f in files
    ]
    return {"models": result}


def _load_baseline_for_benchmark() -> ViTForImageClassification:
    global _baseline_bench_model
    if _baseline_bench_model is not None:
        log.info("Baseline benchmark model served from cache")
        return _baseline_bench_model

    path = MODELS_DIR / BASELINE_MODEL_ID
    if not path.exists():
        raise FileNotFoundError(f"Baseline model not found: {path}")
    log.info("Loading baseline model for benchmark (first run — will be cached)...")
    model = ViTForImageClassification.from_pretrained(
        MODEL_HF_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    state_dict = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    _baseline_bench_model = model
    return _baseline_bench_model


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


_BENCH_DONE = object()  # sentinel pushed to queue when worker finishes


def _run_benchmark_thread(ids: list, q: queue.Queue):
    """
    Runs in a background thread. Pushes SSE-ready dicts onto q.
    Pushes _BENCH_DONE sentinel when finished.
    """
    def cb(phase: str, current: int, total: int, msg: str):
        q.put({"type": "tick", "phase": phase,
               "current": current, "total": total, "msg": msg})

    try:
        # ── Baseline ──────────────────────────────────────────────────────────
        q.put({"type": "progress", "step": "baseline",
               "model_id": BASELINE_MODEL_ID, "index": 0})
        try:
            q.put({"type": "tick", "phase": "loading", "current": 0, "total": 1,
                   "msg": "Loading baseline model (first run takes 1–2 min, cached after)…"})
            baseline_model = _load_baseline_for_benchmark()
            q.put({"type": "tick", "phase": "loading", "current": 1, "total": 1,
                   "msg": "Baseline model loaded ✓"})
            baseline_result = benchmark_model(
                model=baseline_model,
                model_name="Baseline",
                device=DEVICE,
                baseline_latency_ms=None,
                baseline_energy_mj=None,
                baseline_params=BASELINE_PARAM_COUNT,
                cb=cb,
            )
            # Don't del baseline_model — it's cached in _baseline_bench_model for reuse
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            q.put({"type": "result", "model_id": "baseline", "data": baseline_result})
            bl_latency = baseline_result["mean_batch_latency_ms"]
            bl_energy  = baseline_result["avg_energy_per_sample_mj"]

        except Exception as e:
            log.exception("Baseline benchmark failed")
            q.put({"type": "error", "model_id": "baseline", "message": str(e)})
            q.put(_BENCH_DONE)
            return

        # ── Compacted models ──────────────────────────────────────────────────
        for idx, model_id in enumerate(ids, start=1):
            q.put({"type": "progress", "step": "model",
                   "model_id": model_id, "index": idx})
            try:
                model_path = str(COMPACTED_MODELS_DIR / model_id)
                cm, _ = load_compacted_model(model_path, device=DEVICE)
                q.put({"type": "tick", "phase": "loading", "current": 1, "total": 1,
                       "msg": f"{model_id} loaded, starting benchmark…"})
                result = benchmark_model(
                    model=cm,
                    model_name=model_id,
                    device=DEVICE,
                    baseline_latency_ms=bl_latency,
                    baseline_energy_mj=bl_energy,
                    baseline_params=BASELINE_PARAM_COUNT,
                    cb=cb,
                )
                del cm
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

                q.put({"type": "result", "model_id": model_id, "data": result})

            except Exception as e:
                log.exception(f"Benchmark failed for {model_id}")
                q.put({"type": "error", "model_id": model_id, "message": str(e)})

    finally:
        q.put(_BENCH_DONE)


@app.get("/api/benchmark")
async def run_benchmark(
    model_ids: str = Query(..., description="Comma-separated compacted model filenames"),
):
    """
    Server-Sent Events stream.

    Event shapes:
      { type: 'start',    total: N }
      { type: 'progress', step: 'baseline'|'model', model_id, index }
      { type: 'tick',     phase, current, total, msg }
      { type: 'result',   model_id, data: {...} }
      { type: 'error',    model_id, message }
      { type: 'done' }
    """
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]

    async def generate():
        yield _sse({"type": "start", "total": len(ids) + 1})

        q: queue.Queue = queue.Queue()
        t = threading.Thread(target=_run_benchmark_thread, args=(ids, q), daemon=True)
        t.start()

        loop = asyncio.get_event_loop()

        while True:
            try:
                item = await loop.run_in_executor(None, q.get, True, 0.05)
            except queue.Empty:
                await asyncio.sleep(0)
                continue

            if item is _BENCH_DONE:
                break

            yield _sse(item)

        yield _sse({"type": "done"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )