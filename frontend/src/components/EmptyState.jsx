// src/components/EmptyState.jsx
import styles from './EmptyState.module.css'

export default function EmptyState({ hasModels, hasImage }) {
  const steps = [
    { done: hasModels, text: 'Select at least one model from the sidebar' },
    { done: hasImage,  text: 'Click an image in the strip above' },
  ]

  return (
    <div className={styles.wrap}>
      <div className={styles.graphic}>
        <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
          <rect x="10" y="10" width="25" height="25" rx="4" fill="none" stroke="var(--border-hi)" strokeWidth="1.5"/>
          <rect x="45" y="10" width="25" height="25" rx="4" fill="none" stroke="var(--border-hi)" strokeWidth="1.5"/>
          <rect x="10" y="45" width="25" height="25" rx="4" fill="none" stroke="var(--border-hi)" strokeWidth="1.5"/>
          <rect x="45" y="45" width="25" height="25" rx="4" fill="none" stroke="var(--border-hi)" strokeWidth="1.5"/>
          {/* heatmap dots pattern */}
          {[
            [16, 16], [22, 16], [28, 16],
            [16, 22], [22, 22], [28, 22],
            [16, 28], [22, 28], [28, 28],
          ].map(([cx, cy], i) => (
            <circle key={i} cx={cx} cy={cy} r="2"
              fill={i === 4 ? 'var(--accent)' : 'var(--border-hi)'}
              opacity={i === 4 ? 1 : 0.4}
            />
          ))}
        </svg>
      </div>

      <h3 className={styles.title}>No comparison to show yet</h3>
      <p className={styles.sub}>Follow the steps below to visualize attention rollout:</p>

      <div className={styles.steps}>
        {steps.map((s, i) => (
          <div key={i} className={`${styles.step} ${s.done ? styles.stepDone : ''}`}>
            <span className={styles.stepNum}>{s.done ? '✓' : i + 1}</span>
            <span className={styles.stepText}>{s.text}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
