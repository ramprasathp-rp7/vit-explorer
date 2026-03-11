// src/api.js

const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000/api'

// When running through ngrok, all fetch calls need this header to bypass
// the browser warning interstitial page that ngrok shows on first access.
// Has no effect when talking directly to localhost.
const HEADERS = { 'ngrok-skip-browser-warning': '1' }

// Static images are served from the same host as the API but at /static/images/...
// Derive the root (strip /api suffix) so image URLs work through ngrok too.
export const STATIC_ROOT = BASE.replace(/\/api$/, '')

export async function fetchModels() {
  const r = await fetch(`${BASE}/models`, { headers: HEADERS })
  if (!r.ok) throw new Error('Failed to load models')
  return (await r.json()).models
}

export async function fetchClasses() {
  const r = await fetch(`${BASE}/classes`, { headers: HEADERS })
  if (!r.ok) throw new Error('Failed to load classes')
  return (await r.json()).classes   // [{id, name}]
}

export async function fetchClassImages(classId) {
  const r = await fetch(`${BASE}/classes/${classId}/images`, { headers: HEADERS })
  if (!r.ok) throw new Error(`Failed to load images for class: ${classId}`)
  return (await r.json()).images    // [{id, filename, url, split}]
}

export async function fetchRolloutBatch({ modelIds, imageId, discardRatio, headFusion, alpha }) {
  const params = new URLSearchParams({
    model_ids: modelIds.join(','),
    image_id: imageId,
    discard_ratio: discardRatio ?? 0.05,
    head_fusion: headFusion ?? 'mean',
    alpha: alpha ?? 0.55,
    view: 'overlay',
  })
  const r = await fetch(`${BASE}/rollout/batch?${params}`, { headers: HEADERS })
  if (!r.ok) throw new Error('Rollout request failed')
  return (await r.json()).results
}

export async function fetchCompactedModels() {
  const r = await fetch(`${BASE}/compacted_models`, { headers: HEADERS })
  if (!r.ok) throw new Error('Failed to load compacted models')
  const data = await r.json()
  return data.models // [{id, name}]
}

export function streamBenchmark(modelIds, onEvent) {
  const controller = new AbortController()
  const params = new URLSearchParams({ model_ids: modelIds.join(',') })

  fetch(`${BASE}/benchmark?${params}`, {
    signal: controller.signal,
    headers: HEADERS,
  })
    .then(async (res) => {
      if (!res.ok) {
        onEvent({ type: 'error', model_id: 'stream', message: `HTTP ${res.status}` })
        return
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()
        for (const part of parts) {
          const line = part.trim()
          if (line.startsWith('data: ')) {
            try {
              onEvent(JSON.parse(line.slice(6)))
            } catch (_) {}
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onEvent({ type: 'error', model_id: 'stream', message: err.message })
      }
    })

  return controller
}