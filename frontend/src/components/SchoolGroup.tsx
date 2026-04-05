import type { SignalResult, SchoolId } from '../types/signals'
import type { Theme } from '../themes/research'
import SignalCard from './SignalCard'

interface Props {
  schoolId: SchoolId
  signals: SignalResult[]
  isComputing: boolean
  theme: Theme
}

const SCHOOL_SIGNAL_COUNTS: Record<SchoolId, number> = {
  information_theoretic: 6,
  layer_wise: 4,
  geometric: 2,
  behavioral: 4,
  calibration: 2,
}

export default function SchoolGroup({ schoolId, signals, isComputing, theme }: Props) {
  const schoolConfig = theme.schools[schoolId]
  const total = SCHOOL_SIGNAL_COUNTS[schoolId]
  const computed = signals.length
  const t = theme.copy

  return (
    <div className="mb-4">
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-0.5 h-4 ${schoolConfig.border} rounded`} />
        <span className={`${schoolConfig.color} font-bold text-xs tracking-wide`}>
          {schoolConfig.name}
        </span>
        <span className={`${theme.colors.textDim} text-[10px]`}>
          {computed} / {total} signals
        </span>
        <span className="text-[10px] ml-auto">
          {computed === total ? (
            <span className="text-green-400">✓ Complete</span>
          ) : isComputing ? (
            <span className="text-yellow-400">{t.computing}</span>
          ) : (
            <span className={theme.colors.textDim}>{t.queued}</span>
          )}
        </span>
      </div>

      {signals.length > 0 ? (
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
          {signals.map(signal => (
            <SignalCard key={signal.signal_id} signal={signal} theme={theme} />
          ))}
        </div>
      ) : (
        <div className={`${theme.colors.card} border border-dashed ${theme.colors.cardBorder} rounded-md p-3 text-center ${theme.colors.textDim} text-[11px] italic`}>
          {isComputing ? t.computing : t.queued}
        </div>
      )}
    </div>
  )
}
