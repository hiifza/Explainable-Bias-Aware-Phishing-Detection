import { useRef, useEffect } from 'react'
import * as THREE from 'three'
import { useAppStore } from '@/store'
import styles from './ThreatGlobe.module.css'

interface NodeData {
  x: number; y: number; z: number
  isPhishing: boolean
  speed: number
}

export default function ThreatGlobe() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const theme = useAppStore((s) => s.theme)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const frameRef    = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const W = canvas.clientWidth  || window.innerWidth
    const H = canvas.clientHeight || window.innerHeight

    // ── Renderer ──
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true })
    renderer.setSize(W, H)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    rendererRef.current = renderer

    const scene  = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 500)
    camera.position.set(0, 4, 36)

    // ── Colors from design system ──
    const isDark  = theme === 'dark'
    const cPhish  = new THREE.Color(isDark ? 0xF87171 : 0xDC2626)
    const cLegit  = new THREE.Color(isDark ? 0x4ADE80 : 0x16A34A)
    const cNeutral= new THREE.Color(isDark ? 0xD3B99F : 0xC97D60)

    // ── Generate URL nodes ──
    const COUNT = 180
    const nodes: NodeData[] = []
    const positions: number[] = []
    const colors: number[]    = []

    for (let i = 0; i < COUNT; i++) {
      const isPhishing = Math.random() < 0.43
      // Cluster phishing nodes to one side, legit to the other
      const clusterBias = isPhishing ? -4 : 4
      const r     = 6 + Math.random() * 11
      const theta = Math.random() * Math.PI * 2
      const phi   = (Math.random() - 0.5) * Math.PI * 0.85
      const x     = r * Math.cos(phi) * Math.cos(theta) + clusterBias * (0.5 + Math.random() * 0.5)
      const y     = r * Math.cos(phi) * Math.sin(theta)
      const z     = r * Math.sin(phi) - 2

      nodes.push({ x, y, z, isPhishing, speed: 0.3 + Math.random() * 0.7 })
      positions.push(x, y, z)

      const c = isPhishing
        ? cPhish
        : Math.random() < 0.75 ? cLegit : cNeutral
      colors.push(c.r, c.g, c.b)
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geo.setAttribute('color',    new THREE.Float32BufferAttribute(colors, 3))

    const mat = new THREE.PointsMaterial({
      size: 0.28,
      vertexColors: true,
      transparent: true,
      opacity: 0.82,
      sizeAttenuation: true,
    })

    const points = new THREE.Points(geo, mat)
    scene.add(points)

    // ── Connection edges (sparse, only between nearby nodes) ──
    const edgePositions: number[] = []
    for (let i = 0; i < COUNT; i++) {
      for (let j = i + 1; j < COUNT; j++) {
        const a = nodes[i], b = nodes[j]
        const dist = Math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        if (dist < 6 && Math.random() < 0.25) {
          edgePositions.push(a.x, a.y, a.z, b.x, b.y, b.z)
        }
      }
    }
    const edgeGeo = new THREE.BufferGeometry()
    edgeGeo.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3))
    const edgeMat = new THREE.LineBasicMaterial({
      color: isDark ? 0x3a1a1a : 0xC9A090,
      transparent: true,
      opacity: isDark ? 0.14 : 0.18,
    })
    scene.add(new THREE.LineSegments(edgeGeo, edgeMat))

    // ── Ambient ambient particles (fine dust) ──
    const dustGeo = new THREE.BufferGeometry()
    const dustPos: number[] = []
    for (let i = 0; i < 600; i++) {
      dustPos.push(
        (Math.random() - 0.5) * 60,
        (Math.random() - 0.5) * 40,
        (Math.random() - 0.5) * 40
      )
    }
    dustGeo.setAttribute('position', new THREE.Float32BufferAttribute(dustPos, 3))
    const dustMat = new THREE.PointsMaterial({
      size: 0.06,
      color: isDark ? 0x660707 : 0xC97D60,
      transparent: true,
      opacity: 0.35,
    })
    scene.add(new THREE.Points(dustGeo, dustMat))

    // ── Mouse parallax ──
    let mouseX = 0, mouseY = 0
    const onMouse = (e: MouseEvent) => {
      mouseX = (e.clientX / window.innerWidth  - 0.5) * 2
      mouseY = (e.clientY / window.innerHeight - 0.5) * 2
    }
    window.addEventListener('mousemove', onMouse)

    // ── Animate ──
    let t = 0
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate)
      t += 0.004

      points.rotation.y  += 0.0006
      points.rotation.x  += 0.0002
      dustGeo.attributes.position.needsUpdate = true

      camera.position.x += (mouseX * 3 - camera.position.x) * 0.025
      camera.position.y += (-mouseY * 2 - camera.position.y) * 0.025

      renderer.render(scene, camera)
    }
    animate()

    // ── Resize ──
    const onResize = () => {
      const w = canvas.clientWidth, h = canvas.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(frameRef.current)
      window.removeEventListener('mousemove', onMouse)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      geo.dispose()
      mat.dispose()
      edgeGeo.dispose()
      edgeMat.dispose()
      dustGeo.dispose()
      dustMat.dispose()
    }
  }, [theme])

  return (
    <canvas
      ref={canvasRef}
      className={styles.canvas}
      aria-hidden="true"
    />
  )
}
