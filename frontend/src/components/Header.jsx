// src/components/Header.jsx
import { PanelLeft, Cpu, Image as ImageIcon, CheckCircle } from 'lucide-react'
import styles from './Header.module.css'

export default function Header({ sidebarOpen, onToggleSidebar, modelsCount, imageSelected }) {
  return (
    <header className={styles.header}>
      <div className={styles.left}>
        <button
          className={styles.toggleBtn}
          onClick={onToggleSidebar}
          title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
        >
          <PanelLeft size={18} />
        </button>

        <div className={styles.brand}>
          <span className={styles.brandMono}>ViT</span>
          <span className={styles.brandText}>Attention Explorer</span>
        </div>
      </div>

      <div className={styles.status}>
        <StatusPill
          icon={<Cpu size={12} />}
          label={modelsCount === 0 ? 'No models' : `${modelsCount} model${modelsCount > 1 ? 's' : ''}`}
          active={modelsCount > 0}
        />
        <StatusPill
          icon={<ImageIcon size={12} />}
          label={imageSelected ? 'Image selected' : 'No image'}
          active={imageSelected}
        />
      </div>
    </header>
  )
}

function StatusPill({ icon, label, active }) {
  return (
    <div className={`${styles.pill} ${active ? styles.pillActive : ''}`}>
      {icon}
      <span>{label}</span>
    </div>
  )
}
