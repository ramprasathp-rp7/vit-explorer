# ViT Attention Rollout Explorer

A web app to compare Attention Rollout visualizations across different ViT pruning methods.

---

## Project Structure

```
vit-explorer/
├── backend/          # Python FastAPI server
│   ├── main.py
│   ├── rollout.py    # Your attention rollout logic
│   └── requirements.txt
├── frontend/         # React (Vite) app
│   ├── src/
│   ├── package.json
│   └── ...
├── assets/
│   ├── images/       # ← PUT YOUR TEST IMAGES HERE (any jpg/png)
│   └── models/       # ← PUT YOUR MODEL .pth FILES HERE
└── setup.sh          # One-command setup script
```

## Quick Setup

```bash
# 1. Run the setup script (creates conda env + installs all deps)
chmod +x setup.sh
./setup.sh

# 2. Put your data in place:
#    - Test images → assets/images/  (jpg, png, etc.)
#    - Model weights → assets/models/ (*.pth files)
#    - Edit backend/config.py to set MODEL_NAME and NUM_LABELS if needed

# 3. Start both servers (two terminals):

# Terminal 1 - Backend
conda activate vit-explorer
cd backend && uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend && npm run dev
```

Then open http://localhost:5173

## Model File Naming

The model name shown in the UI is derived from the filename.
- `baseline_vit_best.pth` → "baseline vit best"
- `hybrid_magnitude_weighted.pth` → "hybrid magnitude weighted"

You can override display names in `backend/config.py`.

## Configuration (`backend/config.py`)

```python
MODEL_HF_NAME = "google/vit-base-patch16-224-in21k"  # HuggingFace model identifier
NUM_LABELS = 10                                         # Number of classes
CLASS_NAMES = ["tench", "English springer", ...]       # Optional: class names
```
