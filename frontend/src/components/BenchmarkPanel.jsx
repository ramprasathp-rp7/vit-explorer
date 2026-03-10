// components/BenchmarkPanel.jsx
import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchCompactedModels, streamBenchmark } from '../api.js'
import styles from './BenchmarkPanel.module.css'

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function fmt(val, decimals = 2) {
  if (val === null || val === undefined) return '—'
  return Number(val).toFixed(decimals)
}

function fmtParams(n) {
  if (n === null || n === undefined) return '—'
  return `${(n / 1e6).toFixed(2)} M`
}

function fmtReduction(pct) {
  if (pct === null || pct === undefined) return <span className={styles.neutral}>—</span>
  const v = Number(pct)
  return <span className={v > 0 ? styles.good : styles.neutral}>{v > 0 ? `↓ ${v.toFixed(1)}%` : `${v.toFixed(1)}%`}</span>
}

function fmtSpeedup(v) {
  if (v === null || v === undefined) return <span className={styles.neutral}>—</span>
  return <span className={v > 1 ? styles.good : styles.neutral}>{Number(v).toFixed(2)}×</span>
}

const PHASE_LABELS = {
  init:        'Initialising CUDA',
  warmup:      'Warming up',
  latency:     'Measuring latency',
  energy_init: 'Preparing energy measurement',
  energy:      'Measuring energy',
  loading:     'Loading model',
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-progress bar shown during active benchmarking
// ─────────────────────────────────────────────────────────────────────────────

function SubProgress({ tick }) {
  if (!tick) return null
  const { phase, current, total, msg } = tick
  const label = PHASE_LABELS[phase] ?? phase
  const pct = total > 1 ? Math.round((current / total) * 100) : 0

  return (
    <div className={styles.subProgress}>
      <div className={styles.subProgressHeader}>
        <span className={styles.subPhase}>{label}</span>
        <span className={styles.subMsg}>{msg}</span>
        {total > 1 && <span className={styles.subCount}>{current}/{total}</span>}
      </div>
      {total > 1 && (
        <div className={styles.subBar}>
          <div className={styles.subFill} style={{ width: `${pct}%` }} />
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export default function BenchmarkPanel() {
  const [compactedModels, setCompactedModels] = useState([])
  const [selectedIds,     setSelectedIds]     = useState([])
  const [loadError,       setLoadError]       = useState(null)

  const [running,      setRunning]      = useState(false)
  const [currentModel, setCurrentModel] = useState(null)
  const [overallIndex, setOverallIndex] = useState(0)
  const [overallTotal, setOverallTotal] = useState(0)
  const [latestTick,   setLatestTick]   = useState(null)
  const [results,      setResults]      = useState({})
  const [errors,       setErrors]       = useState({})
  const [done,         setDone]         = useState(false)
  const abortRef = useRef(null)

  useEffect(() => {
    fetchCompactedModels()
      .then(setCompactedModels)
      .catch(e => setLoadError(e.message))
  }, [])

  const toggleModel = useCallback((id) => {
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )
  }, [])

  const selectAll = () => setSelectedIds(compactedModels.map(m => m.id))
  const clearAll  = () => setSelectedIds([])

  const handleRun = () => {
    if (running || selectedIds.length === 0) return
    setRunning(true)
    setDone(false)
    setResults({})
    setErrors({})
    setCurrentModel(null)
    setLatestTick(null)
    setOverallIndex(0)

    const ctrl = streamBenchmark(selectedIds, (event) => {
      switch (event.type) {
        case 'start':
          setOverallTotal(event.total)
          break
        case 'progress':
          setCurrentModel(event.model_id)
          setOverallIndex(event.index)
          setLatestTick(null)
          break
        case 'tick':
          setLatestTick(event)
          break
        case 'result':
          setResults(prev => ({ ...prev, [event.model_id]: event.data }))
          setLatestTick(null)
          break
        case 'error':
          setErrors(prev => ({ ...prev, [event.model_id]: event.message }))
          setLatestTick(null)
          break
        case 'done':
          setRunning(false)
          setDone(true)
          setCurrentModel(null)
          setLatestTick(null)
          break
        default:
          break
      }
    })
    abortRef.current = ctrl
  }

  const handleStop = () => {
    abortRef.current?.abort()
    setRunning(false)
    setCurrentModel(null)
    setLatestTick(null)
  }

  // Result rows: baseline first, then compacted in selection order
  const baseline   = results['baseline']
  const resultRows = [
    ...(baseline ? [{ id: 'baseline', name: 'Baseline', data: baseline }] : []),
    ...selectedIds
      .filter(id => results[id])
      .map(id => ({
        id,
        name: compactedModels.find(m => m.id === id)?.name ?? id,
        data: results[id],
      })),
  ]

  const hasResults  = resultRows.length > 0
  const overallPct  = overallTotal > 0 ? Math.round((overallIndex / overallTotal) * 100) : 0
  const currentName = currentModel
    ? (compactedModels.find(m => m.id === currentModel)?.name ?? currentModel)
    : null

  const energyMethod = resultRows[0]?.data?.energy_method
  const nvmlReason   = resultRows[0]?.data?.nvml_reason

  // ── Render ─────────────────────────────────────────────────────────────────

  if (loadError) {
    return <div className={styles.errorBanner}>⚠ Could not load compacted models: {loadError}</div>
  }

  if (compactedModels.length === 0) {
    return (
      <div className={styles.empty}>
        <p className={styles.emptyTitle}>No compacted models found</p>
        <p className={styles.emptyHint}>
          Place compacted <code>.pth</code> files in{' '}
          <code>assets/compacted_models/</code> to enable benchmarking.
        </p>
      </div>
    )
  }

  return (
    <div className={styles.panel}>

      {/* ── Disclaimer ────────────────────────────────────────────────────── */}
      <div className={styles.disclaimer}>
        <span className={styles.disclaimerIcon}>ℹ</span>
        Demo mode — indicative measurements only (10 warmup / 50 iterations / batch 32). Not publication-grade.
      </div>

      {/* ── Model selector ────────────────────────────────────────────────── */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>Compacted Models</span>
          <div className={styles.sectionActions}>
            <button className={styles.linkBtn} onClick={selectAll}>All</button>
            <span className={styles.sep}>·</span>
            <button className={styles.linkBtn} onClick={clearAll}>None</button>
          </div>
        </div>
        <div className={styles.modelList}>
          {compactedModels.map(m => {
            const checked   = selectedIds.includes(m.id)
            const hasResult = !!results[m.id]
            const hasError  = !!errors[m.id]
            const isActive  = running && currentModel === m.id
            return (
              <label key={m.id} className={[
                styles.modelItem,
                checked  ? styles.modelItemChecked : '',
                isActive ? styles.modelItemActive  : '',
              ].join(' ')}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleModel(m.id)}
                  disabled={running}
                  className={styles.checkbox}
                />
                <span className={styles.modelName}>{m.name}</span>
                {isActive  && <span className={styles.spinner} />}
                {hasResult && !isActive && <span className={`${styles.badge} ${styles.badgeOk}`}>✓</span>}
                {hasError  && !isActive && <span className={`${styles.badge} ${styles.badgeErr}`}>!</span>}
              </label>
            )
          })}
        </div>
      </section>

      {/* ── Run / Stop ────────────────────────────────────────────────────── */}
      <div className={styles.runRow}>
        {!running ? (
          <button className={styles.runBtn} onClick={handleRun} disabled={selectedIds.length === 0}>
            ▶ Run Benchmark
          </button>
        ) : (
          <button className={styles.stopBtn} onClick={handleStop}>■ Stop</button>
        )}
        {done && !running && <span className={styles.doneLabel}>✓ Complete</span>}
      </div>

      {/* ── Overall + sub progress ────────────────────────────────────────── */}
      {running && (
        <div className={styles.progressSection}>
          <div className={styles.progressHeader}>
            <span className={styles.progressModel}>
              {currentName ? `Benchmarking ${currentName}…` : 'Starting…'}
            </span>
            <span className={styles.progressFraction}>{overallIndex}/{overallTotal} models</span>
          </div>
          <div className={styles.progressBar}>
            <div className={styles.progressFill} style={{ width: `${overallPct}%` }} />
          </div>
          <SubProgress tick={latestTick} />
        </div>
      )}

      {/* ── Per-model errors ──────────────────────────────────────────────── */}
      {Object.entries(errors).map(([id, msg]) => (
        <div key={id} className={styles.errorRow}>
          ⚠ <strong>{id}</strong>: {msg}
        </div>
      ))}

      {/* ── Results table ─────────────────────────────────────────────────── */}
      {hasResults && (
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}>Results</span>
          </div>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.thName}>Model</th>
                  <th>Device</th>
                  <th>Params</th>
                  <th>Param ↓</th>
                  <th>Batch latency (ms)</th>
                  <th>Per-sample (ms)</th>
                  <th>Speedup</th>
                  <th>Energy/sample (mJ)</th>
                  <th>Energy ↓</th>
                </tr>
              </thead>
              <tbody>
                {resultRows.map(({ id, name, data }) => {
                  const isBaseline = id === 'baseline'
                  return (
                    <tr key={id} className={isBaseline ? styles.baselineRow : ''}>
                      <td className={styles.tdName}>
                        {name}
                        {isBaseline && <span className={styles.baselinePill}>baseline</span>}
                      </td>
                      <td>
                        <span className={data.device === 'cuda' ? styles.good : styles.warn}>
                          {data.device ?? '—'}
                        </span>
                      </td>
                      <td>{fmtParams(data.total_params)}</td>
                      <td>{isBaseline ? <span className={styles.neutral}>—</span> : fmtReduction(data.param_reduction_pct)}</td>
                      <td>{fmt(data.mean_batch_latency_ms)} ± {fmt(data.std_batch_latency_ms)}</td>
                      <td>{fmt(data.mean_sample_latency_ms, 3)}</td>
                      <td>{isBaseline ? <span className={styles.neutral}>1.00×</span> : fmtSpeedup(data.speedup)}</td>
                      <td>
                        {data.energy_supported
                          ? `${fmt(data.avg_energy_per_sample_mj, 4)} ± ${fmt(data.std_energy_per_sample_mj, 4)}`
                          : <span className={styles.noData} title={data.nvml_reason ?? ''}>N/A</span>}
                      </td>
                      <td>
                        {isBaseline || !data.energy_supported
                          ? <span className={styles.neutral}>—</span>
                          : fmtReduction(data.energy_reduction_pct)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className={styles.tableNote}>
            Latency: batch={resultRows[0]?.data?.batch_size ?? 32}, {resultRows[0]?.data?.measure_iters ?? 50} iters.
            {energyMethod === 'hw_counter' && ' Energy: hardware counter (nvml).'}
            {energyMethod === 'power_sampling' && ' Energy: power sampling (hw counter unavailable).'}
            {(!energyMethod || energyMethod === 'unavailable') && nvmlReason && ` Energy N/A — ${nvmlReason}.`}
            {' '}Device column confirms cuda was used.
          </p>
        </section>
      )}
    </div>
  )
}
