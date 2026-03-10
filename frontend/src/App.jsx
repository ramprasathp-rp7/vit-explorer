// src/App.jsx
import { useState, useEffect, useCallback } from 'react'
import { fetchModels, fetchClasses } from './api.js'
import { useRollout } from './useRollout.js'
import Header from './components/Header.jsx'
import ModelSelector from './components/ModelSelector.jsx'
import ClassBrowser from './components/ClassBrowser.jsx'
import SettingsPanel from './components/SettingsPanel.jsx'
import ComparisonGrid from './components/ComparisonGrid.jsx'
import EmptyState from './components/EmptyState.jsx'
import styles from './App.module.css'
import BenchmarkPanel from './components/BenchmarkPanel.jsx'

export default function App() {
  // ── Remote data ────────────────────────────────────────────────────────────
  const [allModels,  setAllModels]  = useState([])
  const [allClasses, setAllClasses] = useState([])
  const [dataLoading, setDataLoading] = useState(true)
  const [dataError,   setDataError]   = useState(null)

  useEffect(() => {
    Promise.all([fetchModels(), fetchClasses()])
      .then(([models, classes]) => {
        setAllModels(models)
        setAllClasses(classes)
      })
      .catch(e => setDataError(e.message))
      .finally(() => setDataLoading(false))
  }, [])

  // ── Tab ────────────────────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState('rollout') // 'rollout' | 'benchmark'

  // ── Selection state ────────────────────────────────────────────────────────
  const [selectedModels, setSelectedModels] = useState([])
  const [selectedImage,  setSelectedImage]  = useState(null)

  // Auto-select baseline on load
  useEffect(() => {
    if (allModels.length === 0) return
    const baseline = allModels.find(m => m.id.toLowerCase().includes('baseline'))
    if (baseline && !selectedModels.includes(baseline.id)) {
      setSelectedModels([baseline.id])
    }
  }, [allModels])

  // ── Settings ───────────────────────────────────────────────────────────────
  const [discardRatio, setDiscardRatio] = useState(0.05)
  const [headFusion,   setHeadFusion]   = useState('mean')
  const [alpha,        setAlpha]        = useState(0.55)

  // ── Rollout ────────────────────────────────────────────────────────────────
  const { results, loading: rolloutLoading, error: rolloutError } = useRollout({
    modelIds:    selectedModels,
    imageId:     selectedImage,
    discardRatio,
    headFusion,
    alpha,
  })

  // ── Sidebar ────────────────────────────────────────────────────────────────
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const toggleModel = useCallback((id) => {
    setSelectedModels(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )
  }, [])

  if (dataLoading) return <LoadingScreen />
  if (dataError)   return <ErrorScreen message={dataError} />

  const ready = selectedModels.length > 0 && selectedImage !== null

  return (
    <div className={styles.layout}>
      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      {activeTab === 'rollout' && (
        <aside className={`${styles.sidebar} ${sidebarOpen ? '' : styles.sidebarCollapsed}`}>
          <div className={styles.sidebarInner}>
            <ModelSelector
              models={allModels}
              selected={selectedModels}
              onToggle={toggleModel}
            />
            <div className={styles.divider} />
            <SettingsPanel
              discardRatio={discardRatio} setDiscardRatio={setDiscardRatio}
              headFusion={headFusion}     setHeadFusion={setHeadFusion}
              alpha={alpha}               setAlpha={setAlpha}
            />
          </div>
        </aside>
      )}

      {/* ── Main ─────────────────────────────────────────────────────────── */}
      <main className={styles.main}>
        <Header
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen(v => !v)}
          modelsCount={selectedModels.length}
          imageSelected={!!selectedImage}
        />

        {/* ── Tab bar ──────────────────────────────────────────────────── */}
        <div className={styles.tabBar}>
          <button
            className={`${styles.tab} ${activeTab === 'rollout' ? styles.tabActive : ''}`}
            onClick={() => setActiveTab('rollout')}
          >
            Attention Rollout
          </button>
          <button
            className={`${styles.tab} ${activeTab === 'benchmark' ? styles.tabActive : ''}`}
            onClick={() => setActiveTab('benchmark')}
          >
            Benchmark
          </button>
        </div>

        {activeTab === 'rollout' && (
          <>
            {/* Two-level class → image browser */}
            <ClassBrowser
              classes={allClasses}
              selectedImage={selectedImage}
              onSelect={setSelectedImage}
            />

            {/* Results */}
            <div className={styles.content}>
              {!ready ? (
                <EmptyState
                  hasModels={selectedModels.length > 0}
                  hasImage={!!selectedImage}
                />
              ) : (
                <ComparisonGrid
                  models={allModels.filter(m => selectedModels.includes(m.id))}
                  results={results}
                  loading={rolloutLoading}
                  error={rolloutError}
                  imageId={selectedImage}
                />
              )}
            </div>
          </>
        )}
        {activeTab === 'benchmark' && (
          <div className={styles.benchmarkWrap}>
            <BenchmarkPanel />
          </div>
        )}
      </main>
    </div>
  )
}

function LoadingScreen() {
  return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100vh', flexDirection:'column', gap:16 }}>
      <div style={{ width:36, height:36, border:'3px solid var(--border)', borderTopColor:'var(--accent)', borderRadius:'50%' }} className="animate-spin" />
      <p style={{ color:'var(--text-2)', fontFamily:'var(--font-mono)', fontSize:'0.8rem' }}>Connecting to backend…</p>
    </div>
  )
}

function ErrorScreen({ message }) {
  return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100vh', flexDirection:'column', gap:12, padding:32 }}>
      <p style={{ color:'var(--danger)', fontFamily:'var(--font-mono)', fontSize:'0.85rem' }}>
        ⚠ Backend connection failed
      </p>
      <p style={{ color:'var(--text-2)', fontSize:'0.8rem', maxWidth:400, textAlign:'center' }}>
        {message}<br /><br />
        Make sure the backend is running:<br />
        <code style={{ color:'var(--accent)' }}>cd backend &amp;&amp; uvicorn main:app --port 8000</code>
      </p>
    </div>
  )
}
