import { useState, useCallback } from 'react'
import { useTheme } from '../hooks/useTheme'
import { useSignalStream } from '../hooks/useSignalStream'
import ChatPanel from '../components/ChatPanel'
import SignalDashboard from '../components/SignalDashboard'
import SignalChart from '../components/SignalChart'
import SettingsPanel from '../components/SettingsPanel'
import StatusBanner from '../components/StatusBanner'
import ThemeToggle from '../components/ThemeToggle'
import type { ChatResponse, LayoutMode, GroupingMode } from '../types/signals'

export default function HomePage() {
  const { mode, theme, toggle } = useTheme()
  const { signals, summary, isStreaming, error, startStream, reset } = useSignalStream()

  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null)
  const [lastQuestion, setLastQuestion] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('split')
  const [groupingMode, setGroupingMode] = useState<GroupingMode>('school')

  const handleSubmit = useCallback(async (question: string) => {
    setIsLoading(true)
    setLastQuestion(question)
    setChatResponse(null)
    reset()

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      if (!res.ok) throw new Error(`Chat failed: ${res.status}`)

      const data: ChatResponse = await res.json()
      setChatResponse(data)
      startStream(data.request_id)
    } catch (err) {
      console.error('Chat error:', err)
    } finally {
      setIsLoading(false)
    }
  }, [reset, startStream])

  if (layoutMode === 'single') {
    return (
      <div className={`min-h-screen ${theme.colors.bg}`}>
        <StatusBanner theme={theme} />
        <SettingsPanel layoutMode={layoutMode} onLayoutChange={setLayoutMode} theme={theme} />
        <div className="max-w-4xl mx-auto">
          <ChatPanel
            theme={theme}
            onSubmit={handleSubmit}
            chatResponse={chatResponse}
            summary={summary}
            isLoading={isLoading}
            lastQuestion={lastQuestion}
          />
          {Object.keys(signals).length > 0 && (
            <div className="p-4">
              <SignalChart signals={signals} theme={theme} />
              <SignalDashboard
                signals={signals}
                isStreaming={isStreaming}
                theme={theme}
                groupingMode={groupingMode}
                onGroupingChange={setGroupingMode}
              />
            </div>
          )}
        </div>
        <ThemeToggle mode={mode} onToggle={toggle} />
      </div>
    )
  }

  return (
    <div className={`h-screen flex flex-col ${theme.colors.bg}`}>
      <SettingsPanel layoutMode={layoutMode} onLayoutChange={setLayoutMode} theme={theme} />
      <div className="flex-1 flex overflow-hidden">
        <div className="w-[40%] border-r border-slate-700/30">
          <ChatPanel
            theme={theme}
            onSubmit={handleSubmit}
            chatResponse={chatResponse}
            summary={summary}
            isLoading={isLoading}
            lastQuestion={lastQuestion}
          />
        </div>
        <div className="w-[60%] flex flex-col overflow-hidden">
          {Object.keys(signals).length > 0 && (
            <div className="flex-shrink-0">
              <SignalChart signals={signals} theme={theme} />
            </div>
          )}
          <div className="flex-1 min-h-0">
            <SignalDashboard
              signals={signals}
              isStreaming={isStreaming}
              theme={theme}
              groupingMode={groupingMode}
              onGroupingChange={setGroupingMode}
            />
          </div>
        </div>
      </div>
      <ThemeToggle mode={mode} onToggle={toggle} />
    </div>
  )
}
