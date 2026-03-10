// src/components/ComparisonGrid.jsx
import { useState } from 'react'
import { Download, AlertTriangle } from 'lucide-react'
import { getModelColor } from './ModelSelector.jsx'
import styles from './ComparisonGrid.module.css'

export default function ComparisonGrid({ models, results, loading, error, imageId }) {
  const [zoom, setZoom] = useState(null) // model id being zoomed

  if (error) {
    return (
      <div className={styles.errorBox}>
        <AlertTriangle size={18} style={{ color: 'var(--warn)' }} />
        <span>{error}</span>
      </div>
    )
  }

  return (
    <div>
      {/* Section header */}
      <div className={styles.gridHeader}>
        <h2 className={styles.gridTitle}>
          Attention Rollout — <span className={styles.mono}>{imageId}</span>
        </h2>
        <p className={styles.gridSub}>
          Comparing {models.length} model{models.length > 1 ? 's' : ''}
        </p>
      </div>

      {/* Card grid */}
      <div
        className={styles.grid}
        style={{ '--cols': Math.min(models.length, 3) }}
      >
        {models.map((m, i) => {
          const result = results[m.id]
          const color = getModelColor(i)
          const isLoading = loading || !result
          const hasError = result?.error

          return (
            <ModelCard
              key={m.id}
              model={m}
              result={result}
              color={color}
              loading={isLoading && !hasError}
              onZoom={() => setZoom(zoom === m.id ? null : m.id)}
              zoomed={zoom === m.id}
            />
          )
        })}
      </div>

      {/* Zoom modal */}
      {zoom && results[zoom]?.overlay_b64 && (
        <ZoomModal
          model={models.find(m => m.id === zoom)}
          result={results[zoom]}
          onClose={() => setZoom(null)}
          color={getModelColor(models.findIndex(m => m.id === zoom))}
        />
      )}
    </div>
  )
}

// ─── Model Card ──────────────────────────────────────────────────────────────

function ModelCard({ model, result, color, loading, onZoom, zoomed }) {
  const pred = result?.prediction

  const handleDownload = (e) => {
    e.stopPropagation()
    if (!result?.overlay_b64) return
    const a = document.createElement('a')
    a.href = `data:image/png;base64,${result.overlay_b64}`
    a.download = `rollout_${model.id}.png`
    a.click()
  }

  return (
    <div
      className={`${styles.card} ${zoomed ? styles.cardZoomed : ''}`}
      style={{ '--color': color }}
      onClick={onZoom}
    >
      {/* Card header */}
      <div className={styles.cardHead}>
        <span className={styles.colorDot} style={{ background: color }} />
        <span className={styles.cardTitle}>{model.name}</span>
        {result && !result.error && (
          <button className={styles.dlBtn} onClick={handleDownload} title="Download">
            <Download size={12} />
          </button>
        )}
      </div>

      {/* Image area */}
      <div className={styles.imgWrap}>
        {loading ? (
          <div className={`${styles.skeleton} skeleton`} />
        ) : result?.error ? (
          <div className={styles.errorInCard}>
            <AlertTriangle size={16} />
            <span>{result.error}</span>
          </div>
        ) : result?.overlay_b64 ? (
          <img
            src={`data:image/png;base64,${result.overlay_b64}`}
            alt={`Rollout for ${model.name}`}
            className={styles.rolloutImg}
          />
        ) : null}
      </div>

      {/* Prediction badge */}
      {pred && (
        <div className={styles.predRow}>
          <span className={styles.predLabel}>{pred.class_name}</span>
          <span className={styles.predConf} style={{ color }}>
            {pred.confidence}%
          </span>
        </div>
      )}

      {/* Top-5 mini bars */}
      {pred?.top5 && (
        <div className={styles.bars}>
          {pred.top5.map((t) => (
            <div key={t.class_index} className={styles.bar}>
              <span className={styles.barLabel}>{t.class_name}</span>
              <div className={styles.barTrack}>
                <div
                  className={styles.barFill}
                  style={{
                    width: `${t.confidence}%`,
                    background: t.class_index === pred.class_index ? color : 'var(--border-hi)',
                  }}
                />
              </div>
              <span className={styles.barVal}>{t.confidence}%</span>
            </div>
          ))}
        </div>
      )}

      <div className={styles.zoomHint}>Click to {zoomed ? 'shrink' : 'expand'}</div>
    </div>
  )
}

// ─── Zoom Modal ───────────────────────────────────────────────────────────────

function ZoomModal({ model, result, onClose, color }) {
  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.modalHead}>
          <span className={styles.colorDot} style={{ background: color }} />
          <span className={styles.cardTitle}>{model.name}</span>
          <button className={styles.closeBtn} onClick={onClose}>✕</button>
        </div>
        <img
          src={`data:image/png;base64,${result.overlay_b64}`}
          alt="Zoomed rollout"
          className={styles.modalImg}
        />
        {result.prediction && (
          <div className={styles.modalPred}>
            <span className={styles.predLabel}>{result.prediction.class_name}</span>
            <span className={styles.predConf} style={{ color }}>
              {result.prediction.confidence}% confidence
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
