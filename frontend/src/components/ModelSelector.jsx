// src/components/ModelSelector.jsx
import { Check } from 'lucide-react'
import styles from './ModelSelector.module.css'

// Assign a consistent color per model index
const COLORS = ['#5b8cff', '#3ecf8e', '#f59e0b', '#ef4444', '#b08cf5', '#38bdf8', '#fb923c']

export function getModelColor(index) {
  return COLORS[index % COLORS.length]
}

export default function ModelSelector({ models, selected, onToggle }) {
  return (
    <section>
      <div className={styles.sectionHead}>
        <span className={styles.label}>Models</span>
        <span className={styles.count}>{selected.length} / {models.length}</span>
      </div>

      <p className={styles.hint}>Select models to compare. Baseline is always shown.</p>

      <div className={styles.list}>
        {models.map((m, i) => {
          const active = selected.includes(m.id)
          const isBaseline = m.id.toLowerCase().includes('baseline')
          const color = getModelColor(i)
          return (
            <button
              key={m.id}
              className={`${styles.item} ${active ? styles.itemActive : ''}`}
              onClick={() => onToggle(m.id)}
              style={active ? { '--model-color': color } : {}}
            >
              <span
                className={styles.dot}
                style={{ background: active ? color : 'var(--border-hi)' }}
              />
              <span className={styles.name}>{m.name}</span>
              {isBaseline && <span className="tag tag-dim" style={{ fontSize:'0.6rem', padding:'1px 5px' }}>baseline</span>}
              {active && <Check size={12} style={{ marginLeft:'auto', color }} />}
            </button>
          )
        })}
      </div>

      {selected.length > 1 && (
        <p className={styles.hint} style={{ marginTop: 10 }}>
          {selected.length} models will be compared side by side.
        </p>
      )}
    </section>
  )
}
