// src/components/SettingsPanel.jsx
import styles from './SettingsPanel.module.css'

export default function SettingsPanel({
  discardRatio, setDiscardRatio,
  headFusion, setHeadFusion,
  alpha, setAlpha,
}) {
  return (
    <section>
      <div className={styles.sectionHead}>
        <span className={styles.label}>Rollout Settings</span>
      </div>

      <div className={styles.fields}>
        {/* Head fusion */}
        <Field label="Head Fusion" hint="How to aggregate attention heads">
          <div className={styles.segmented}>
            {['mean', 'max'].map(v => (
              <button
                key={v}
                className={`${styles.seg} ${headFusion === v ? styles.segActive : ''}`}
                onClick={() => setHeadFusion(v)}
              >
                {v}
              </button>
            ))}
          </div>
        </Field>

        {/* Discard ratio */}
        <Field
          label="Discard Ratio"
          hint="Fraction of low-attention tokens to zero out"
          value={discardRatio.toFixed(2)}
        >
          <input
            type="range" min="0" max="0.5" step="0.01"
            value={discardRatio}
            onChange={e => setDiscardRatio(parseFloat(e.target.value))}
            className={styles.slider}
          />
        </Field>

        {/* Alpha blend */}
        <Field
          label="Overlay Alpha"
          hint="Heatmap opacity on the original image"
          value={alpha.toFixed(2)}
        >
          <input
            type="range" min="0.1" max="0.9" step="0.05"
            value={alpha}
            onChange={e => setAlpha(parseFloat(e.target.value))}
            className={styles.slider}
          />
        </Field>
      </div>
    </section>
  )
}

function Field({ label, hint, value, children }) {
  return (
    <div className={styles.field}>
      <div className={styles.fieldHead}>
        <span className={styles.fieldLabel}>{label}</span>
        {value !== undefined && <span className={styles.fieldValue}>{value}</span>}
      </div>
      {hint && <p className={styles.fieldHint}>{hint}</p>}
      {children}
    </div>
  )
}
