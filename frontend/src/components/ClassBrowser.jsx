// src/components/ClassBrowser.jsx
// Two-level image picker: class grid → image strip

import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchClassImages, STATIC_ROOT } from '../api.js'
import { ChevronLeft, ChevronRight, ArrowLeft, Layers } from 'lucide-react'
import styles from './ClassBrowser.module.css'

// Colour per class slot (cycles if > 10)
const CLASS_COLORS = [
  '#5b8cff','#3ecf8e','#f59e0b','#ef4444','#b08cf5',
  '#38bdf8','#fb923c','#a3e635','#f472b6','#34d399',
]

// ── Fetch-based image component ───────────────────────────────────────────────
// <img> tags can't send custom headers, so ngrok's interstitial blocks them
// (ERR_BLOCKED_BY_ORB). Instead we fetch the image via JS (which can set
// headers), create a blob URL, and hand that to the <img> tag.
// Falls back gracefully if fetch fails.
function AuthImg({ src, alt, className }) {
  const [blobUrl, setBlobUrl] = useState(null)
  const [failed,  setFailed]  = useState(false)
  const prevUrl = useRef(null)

  useEffect(() => {
    if (!src) return

    // Revoke previous blob to avoid memory leaks
    if (prevUrl.current) URL.revokeObjectURL(prevUrl.current)

    let cancelled = false
    fetch(src, { headers: { 'ngrok-skip-browser-warning': '1' } })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.blob()
      })
      .then(blob => {
        if (cancelled) return
        const url = URL.createObjectURL(blob)
        prevUrl.current = url
        setBlobUrl(url)
      })
      .catch(() => {
        if (!cancelled) setFailed(true)
      })

    return () => { cancelled = true }
  }, [src])

  // Revoke on unmount
  useEffect(() => {
    return () => { if (prevUrl.current) URL.revokeObjectURL(prevUrl.current) }
  }, [])

  if (failed) return <div className={className} style={{ background: 'var(--bg-3)' }} />
  if (!blobUrl) return <div className={`${className} skeleton`} />
  return <img src={blobUrl} alt={alt} className={className} />
}

// ─────────────────────────────────────────────────────────────────────────────

export default function ClassBrowser({ classes, selectedImage, onSelect }) {
  const [activeClass, setActiveClass] = useState(null)
  const [images, setImages]           = useState([])
  const [loading, setLoading]         = useState(false)
  const scrollRef = useRef(null)

  useEffect(() => {
    if (!activeClass) return
    setLoading(true)
    setImages([])
    fetchClassImages(activeClass.id)
      .then(setImages)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [activeClass?.id])

  const scroll = (dir) => {
    scrollRef.current?.scrollBy({ left: dir * 300, behavior: 'smooth' })
  }

  // ── Class grid view ────────────────────────────────────────────────────────
  if (!activeClass) {
    return (
      <div className={styles.wrapper}>
        <div className={styles.topBar}>
          <Layers size={13} style={{ color: 'var(--text-3)' }} />
          <span className={styles.label}>Select a Class</span>
          <span className={styles.count}>{classes.length} classes</span>
        </div>
        <div className={styles.classGrid}>
          {classes.map((cls, i) => (
            <button
              key={cls.id}
              className={styles.classCard}
              style={{ '--cls-color': CLASS_COLORS[i % CLASS_COLORS.length] }}
              onClick={() => setActiveClass(cls)}
            >
              <span className={styles.classColorBar} />
              <span className={styles.className}>{cls.name}</span>
              <span className={styles.classId}>{cls.id}</span>
            </button>
          ))}
        </div>
      </div>
    )
  }

  // ── Image strip view ───────────────────────────────────────────────────────
  const colorIdx = classes.findIndex(c => c.id === activeClass.id)
  const color = CLASS_COLORS[colorIdx % CLASS_COLORS.length]

  return (
    <div className={styles.wrapper}>
      <div className={styles.topBar}>
        <button className={styles.backBtn} onClick={() => setActiveClass(null)}>
          <ArrowLeft size={13} />
          <span>Classes</span>
        </button>
        <span className={styles.activeName} style={{ color }}>
          {activeClass.name}
        </span>
        <span className={styles.count}>{images.length} images</span>

        <div className={styles.scrollBtns}>
          <button className={styles.scrollBtn} onClick={() => scroll(-1)}>
            <ChevronLeft size={13} />
          </button>
          <button className={styles.scrollBtn} onClick={() => scroll(1)}>
            <ChevronRight size={13} />
          </button>
        </div>
      </div>

      <div className={styles.strip} ref={scrollRef}>
        {loading
          ? Array.from({ length: 12 }).map((_, i) => (
              <div key={i} className={`${styles.thumb} skeleton`} style={{ height: 80 }} />
            ))
          : images.map((img) => {
              const isSelected = selectedImage === img.id
              return (
                <button
                  key={img.id}
                  className={`${styles.thumb} ${isSelected ? styles.thumbSelected : ''}`}
                  style={isSelected ? { '--sel-color': color } : {}}
                  onClick={() => onSelect(img.id)}
                  title={img.filename}
                >
                  <AuthImg
                    src={`${STATIC_ROOT}${img.url}`}
                    alt={img.filename}
                    className={styles.thumbImg}
                  />
                  <span className={`${styles.splitBadge} ${img.split === 'val' ? styles.splitVal : styles.splitTrain}`}>
                    {img.split}
                  </span>
                  {isSelected && (
                    <div className={styles.selectedRing} style={{ borderColor: color }} />
                  )}
                </button>
              )
            })
        }
      </div>
    </div>
  )
}