import { useState, useCallback } from 'react'
import type { SignalResult, SummaryResult } from '../types/signals'

interface SignalStreamState {
  signals: Record<string, SignalResult>
  summary: SummaryResult | null
  isStreaming: boolean
  error: string | null
}

export function useSignalStream() {
  const [state, setState] = useState<SignalStreamState>({
    signals: {},
    summary: null,
    isStreaming: false,
    error: null,
  })

  const startStream = useCallback((requestId: string) => {
    setState({ signals: {}, summary: null, isStreaming: true, error: null })

    const eventSource = new EventSource(`/api/signals/stream?request_id=${requestId}`)

    eventSource.addEventListener('signal', (event) => {
      const signal: SignalResult = JSON.parse(event.data)
      setState(prev => ({
        ...prev,
        signals: { ...prev.signals, [signal.signal_id]: signal },
      }))
    })

    eventSource.addEventListener('summary', (event) => {
      const summary: SummaryResult = JSON.parse(event.data)
      setState(prev => ({ ...prev, summary }))
    })

    eventSource.addEventListener('done', () => {
      setState(prev => ({ ...prev, isStreaming: false }))
      eventSource.close()
    })

    eventSource.addEventListener('error', () => {
      if (eventSource.readyState === EventSource.CLOSED) {
        setState(prev => ({ ...prev, isStreaming: false }))
      } else {
        setState(prev => ({ ...prev, error: 'Connection lost', isStreaming: false }))
        eventSource.close()
      }
    })
  }, [])

  const reset = useCallback(() => {
    setState({ signals: {}, summary: null, isStreaming: false, error: null })
  }, [])

  return { ...state, startStream, reset }
}
