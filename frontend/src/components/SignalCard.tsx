import { useState } from 'react'
import type { SignalResult } from '../types/signals'
import type { Theme } from '../themes/research'
import { getSignalById } from '../data/signalDefinitions'

interface Props {
  signal: SignalResult
  theme: Theme
}

export default function SignalCard({ signal, theme }: Props) {
  const [expanded, setExpanded] = useState(false)
  const definition = getSignalById(signal.signal_id)
  const t = theme.copy

  const numValue = typeof signal.value === 'number' ? signal.value : 0
  const thresholds = definition?.thresholds ?? { low: 0.5, high: 2.0 }
  const inverted = thresholds.inverted ?? false

  const normalized = Math.max(0, Math.min(100, ((numValue) / (thresholds.high * 1.5)) * 100))

  const getColor = () => {
    if (inverted) {
      if (numValue > thresholds.high) return 'text-green-400'
      if (numValue > thresholds.low) return 'text-yellow-400'
      return 'text-red-400'
    }
    if (numValue < thresholds.low) return 'text-green-400'
    if (numValue < thresholds.high) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getBarColor = () => {
    if (inverted) {
      if (numValue > thresholds.high) return 'bg-green-400'
      if (numValue > thresholds.low) return 'bg-yellow-400'
      return 'bg-red-400'
    }
    if (numValue < thresholds.low) return 'bg-green-400'
    if (numValue < thresholds.high) return 'bg-yellow-400'
    return 'bg-red-400'
  }

  return (
    <div className={`${theme.colors.card} border ${theme.colors.cardBorder} rounded-md p-3 text-xs`}>
      <div className="flex justify-between items-start">
        <div className={`font-bold text-[11px] ${theme.colors.text}`}>
          {definition?.name ?? signal.signal_id}
        </div>
        <div className={`text-base font-bold ${getColor()}`}>
          {typeof signal.value === 'number' ? signal.value.toFixed(2) : '—'}
        </div>
      </div>

      <div className="mt-1.5">
        <div className={`${theme.colors.bg} rounded-sm h-1 overflow-hidden`}>
          <div className={`${getBarColor()} h-full rounded-sm`} style={{ width: `${normalized}%` }} />
        </div>
      </div>

      <div className={`${theme.colors.textMuted} text-[10px] mt-1.5 leading-relaxed`}>
        {signal.interpretation}
      </div>

      <button
        onClick={() => setExpanded(!expanded)}
        className={`${theme.colors.textDim} text-[10px] mt-1 hover:underline`}
      >
        {expanded ? '▾' : '▸'} {t.expand}
      </button>

      {expanded && definition && (
        <div className="mt-2 space-y-2">
          <div className={`text-[11px] ${theme.colors.textMuted} leading-relaxed`}>
            {definition.briefDescription}
          </div>

          <div className={`${theme.colors.bg} rounded p-2 text-[10px] ${theme.colors.textDim} leading-relaxed border-l-2 ${theme.colors.cardBorder}`}>
            <div className={`${theme.colors.textMuted} mb-1`}>{t.behindCurtain}</div>
            <span className={theme.colors.textMuted}>Computed as:</span> {definition.formula}
          </div>

          <a
            href={`/signals/${signal.signal_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className={`block ${theme.colors.accent} text-[10px] hover:underline`}
          >
            {t.learnMore}
          </a>
        </div>
      )}
    </div>
  )
}
