import type { Theme } from '../themes/research'

interface Props {
  score: number | null
  totalSignals: number
  computedSignals: number
  theme: Theme
}

export default function CompositeScore({ score, totalSignals, computedSignals, theme }: Props) {
  const t = theme.copy
  const displayScore = score ?? 0
  const color = displayScore >= 80 ? 'bg-green-400' : displayScore >= 50 ? 'bg-yellow-400' : 'bg-red-400'
  const textColor = displayScore >= 80 ? 'text-green-400' : displayScore >= 50 ? 'text-yellow-400' : 'text-red-400'

  return (
    <div className={`px-4 py-2.5 ${theme.colors.bg} border-b border-slate-700/30 flex items-center gap-3`}>
      <span className={`text-[10px] ${theme.colors.textDim} tracking-wide uppercase`}>{t.verdict}</span>
      <div className={`flex-1 ${theme.colors.card} rounded h-2 overflow-hidden`}>
        <div
          className={`${color} h-full rounded transition-all duration-500`}
          style={{ width: `${displayScore}%` }}
        />
      </div>
      <span className={`text-sm font-bold ${textColor}`}>
        {score !== null ? Math.round(displayScore) : '—'}
      </span>
      <span className={`text-[9px] ${theme.colors.textDim}`}>
        {computedSignals} of {totalSignals} signals
      </span>
    </div>
  )
}
