import { useState, useEffect } from 'react'
import type { Theme } from '../themes/research'

interface HealthResponse {
  ollama: { status: string; model: string | null; ready: boolean }
  huggingface: { status: string; model: string | null; ready: boolean }
  ready: boolean
}

interface SetupStatus {
  stage: string
  status: string
  progress?: number
  detail?: string
  model?: string
}

interface Props {
  theme: Theme
}

export default function StatusBanner({ theme }: Props) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [setup, setSetup] = useState<SetupStatus | null>(null)

  // Poll health every 5s
  useEffect(() => {
    let active = true

    const check = async () => {
      try {
        const res = await fetch('/api/health')
        if (res.ok && active) {
          const data = await res.json()
          setHealth(data)

          // If not ready and no setup stream active, start one
          if (!data.ready && !setup) {
            startSetupStream()
          }
        }
      } catch {
        // Backend not reachable
      }
    }

    check()
    const interval = setInterval(check, 5000)
    return () => { active = false; clearInterval(interval) }
  }, [])

  const startSetupStream = () => {
    const eventSource = new EventSource('/api/setup/stream')

    eventSource.addEventListener('status', (event) => {
      const data: SetupStatus = JSON.parse(event.data)
      setSetup(data)
    })

    eventSource.addEventListener('done', () => {
      setSetup(null)
      eventSource.close()
      // Re-check health
      fetch('/api/health').then(r => r.json()).then(setHealth).catch(() => {})
    })

    eventSource.addEventListener('error', () => {
      eventSource.close()
    })
  }

  // Backend not reachable
  if (!health) {
    return (
      <div className="px-4 py-2.5 bg-yellow-900/30 border-b border-yellow-700/30 text-center text-xs text-yellow-300">
        <span className="inline-block animate-pulse mr-2">◉</span>
        Connecting to backend...
      </div>
    )
  }

  // Everything ready
  if (health.ready) return null

  // Show download progress
  if (setup && setup.status !== 'ready') {
    const progress = setup.progress ?? 0
    return (
      <div className="px-4 py-2.5 bg-indigo-900/30 border-b border-indigo-700/30 text-xs text-indigo-300">
        <div className="flex items-center gap-3">
          <span className="inline-block animate-pulse">◉</span>
          <span className="flex-shrink-0">
            Setting up models — first time only
          </span>
          <div className="flex-1 bg-indigo-950/50 rounded h-2 overflow-hidden">
            <div
              className="bg-indigo-400 h-full rounded transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <span className="flex-shrink-0 tabular-nums">
            {setup.detail || `${progress}%`}
          </span>
        </div>
      </div>
    )
  }

  // Generic "not ready" fallback
  return (
    <div className="px-4 py-2.5 bg-indigo-900/30 border-b border-indigo-700/30 text-center text-xs text-indigo-300">
      <span className="inline-block animate-pulse mr-2">◉</span>
      Models are loading — first time only, please wait...
    </div>
  )
}
