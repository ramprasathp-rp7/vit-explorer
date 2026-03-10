// src/useRollout.js
import { useState, useEffect, useRef } from 'react'
import { fetchRolloutBatch } from './api.js'

/**
 * Orchestrates fetching rollout results whenever models/image/settings change.
 * Debounces settings changes (discard ratio slider) to avoid hammering the API.
 */
export function useRollout({ modelIds, imageId, discardRatio, headFusion, alpha }) {
  const [results, setResults] = useState({})   // keyed by modelId
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const abortRef = useRef(null)

  useEffect(() => {
    if (!imageId || modelIds.length === 0) {
      setResults({})
      return
    }

    // Cancel any in-flight request
    if (abortRef.current) abortRef.current.abort()
    const controller = new AbortController()
    abortRef.current = controller

    const run = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await fetchRolloutBatch({ modelIds, imageId, discardRatio, headFusion, alpha })
        if (controller.signal.aborted) return
        const map = {}
        for (const item of data) {
          map[item.model_id ?? item.error] = item
        }
        setResults(map)
      } catch (e) {
        if (!controller.signal.aborted) setError(e.message)
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }

    run()
    return () => controller.abort()
  }, [modelIds.join(','), imageId, discardRatio, headFusion, alpha])

  return { results, loading, error }
}
