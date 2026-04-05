import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import type { ChatResponse } from '../types/signals'
import type { Theme } from '../themes/research'
import ExecutiveSummary from './ExecutiveSummary'
import type { SummaryResult } from '../types/signals'

interface Props {
  theme: Theme
  onSubmit: (question: string) => Promise<void>
  chatResponse: ChatResponse | null
  summary: SummaryResult | null
  isLoading: boolean
  lastQuestion: string
}

export default function ChatPanel({ theme, onSubmit, chatResponse, summary, isLoading, lastQuestion }: Props) {
  const [input, setInput] = useState('')
  const t = theme.copy

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return
    const question = input.trim()
    setInput('')
    await onSubmit(question)
  }

  return (
    <div className={`flex flex-col h-full ${theme.colors.panel}`}>
      <div className="px-4 py-3 border-b border-slate-700/50 text-center">
        {theme.name === 'oracle' && (
          <div className={`text-[10px] tracking-[4px] ${theme.colors.textDim} uppercase`}>
            {t.appSubtitle}
          </div>
        )}
        <div className={`text-lg font-bold ${theme.colors.accent}`}>{t.appTitle}</div>
        {t.appTagline && (
          <div className={`text-[10px] ${theme.colors.textDim} italic`}>{t.appTagline}</div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {lastQuestion && (
          <div>
            <div className={`text-[9px] ${theme.colors.textDim} tracking-widest uppercase mb-1`}>
              {t.user}
            </div>
            <div className={`${theme.colors.card} p-3 rounded-lg ${theme.colors.text} text-sm border ${theme.colors.cardBorder}`}>
              {lastQuestion}
            </div>
          </div>
        )}

        {chatResponse && (
          <div>
            <div className={`text-[9px] ${theme.colors.textDim} tracking-widest uppercase mb-1`}>
              {t.model(chatResponse.model)}
            </div>
            <div className={`${theme.colors.panel} p-3 rounded-lg ${theme.colors.text} text-sm border ${theme.colors.cardBorder} prose prose-invert prose-sm max-w-none`}>
              <ReactMarkdown>{chatResponse.response}</ReactMarkdown>
            </div>
          </div>
        )}

        {summary && <ExecutiveSummary theme={theme} summary={summary} />}
      </div>

      <form onSubmit={handleSubmit} className="p-3 border-t border-slate-700/50">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            disabled={isLoading}
            className={`flex-1 ${theme.colors.card} border ${theme.colors.cardBorder} rounded-md px-3 py-2 text-sm ${theme.colors.text} focus:outline-none focus:ring-1 focus:ring-indigo-500`}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-md text-white text-sm font-medium"
          >
            {isLoading ? t.computing : t.send}
          </button>
        </div>
      </form>
    </div>
  )
}
