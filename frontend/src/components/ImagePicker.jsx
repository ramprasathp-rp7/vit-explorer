// src/components/ImagePicker.jsx
import { useRef } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import styles from './ImagePicker.module.css'

export default function ImagePicker({ images, selected, onSelect }) {
  const scrollRef = useRef(null)

  const scroll = (dir) => {
    if (scrollRef.current) {
      scrollRef.current.scrollBy({ left: dir * 240, behavior: 'smooth' })
    }
  }

  if (images.length === 0) return null

  return (
    <div className={styles.wrapper}>
      <div className={styles.topBar}>
        <span className={styles.label}>Test Images</span>
        <span className={styles.count}>{images.length} available</span>
        <div className={styles.scrollBtns}>
          <button className={styles.scrollBtn} onClick={() => scroll(-1)}>
            <ChevronLeft size={14} />
          </button>
          <button className={styles.scrollBtn} onClick={() => scroll(1)}>
            <ChevronRight size={14} />
          </button>
        </div>
      </div>

      <div className={styles.strip} ref={scrollRef}>
        {images.map((img) => (
          <button
            key={img.id}
            className={`${styles.thumb} ${selected === img.id ? styles.thumbSelected : ''}`}
            onClick={() => onSelect(img.id)}
            title={img.filename}
          >
            <img
              src={img.url}
              alt={img.filename}
              loading="lazy"
              className={styles.thumbImg}
            />
            <span className={styles.thumbName}>{img.id}</span>
            {selected === img.id && <div className={styles.selectedRing} />}
          </button>
        ))}
      </div>
    </div>
  )
}
