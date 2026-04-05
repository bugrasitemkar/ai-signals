import type { SignalResult, SchoolId, GroupingMode } from '../types/signals'
import type { Theme } from '../themes/research'
import SchoolGroup from './SchoolGroup'
import SignalCard from './SignalCard'
import CompositeScore from './CompositeScore'
import { getSignalById } from '../data/signalDefinitions'

interface Props {
  signals: Record<string, SignalResult>
  isStreaming: boolean
  theme: Theme
  groupingMode: GroupingMode
  onGroupingChange: (mode: GroupingMode) => void
}

const SCHOOL_ORDER: SchoolId[] = [
  'information_theoretic',
  'layer_wise',
  'geometric',
  'behavioral',
  'calibration',
]

const SCHOOL_COUNTS: Record<SchoolId, number> = {
  information_theoretic: 6,
  layer_wise: 4,
  geometric: 2,
  behavioral: 4,
  calibration: 2,
}

interface BehaviorGroup {
  label: string
  color: string
  border: string
  description: string
  signals: SignalResult[]
}

function classifySignal(signal: SignalResult): 'confident' | 'moderate' | 'uncertain' {
  if (typeof signal.value !== 'number') return 'moderate'
  if (signal.signal_id === 'composite_score') return 'moderate'

  const def = getSignalById(signal.signal_id)
  if (!def) return 'moderate'

  const v = signal.value
  const { low, high, inverted } = def.thresholds

  if (inverted) {
    if (v > high) return 'confident'
    if (v > low) return 'moderate'
    return 'uncertain'
  } else {
    if (v < low) return 'confident'
    if (v < high) return 'moderate'
    return 'uncertain'
  }
}

function groupByBehavior(signals: SignalResult[]): BehaviorGroup[] {
  const confident: SignalResult[] = []
  const moderate: SignalResult[] = []
  const uncertain: SignalResult[] = []

  for (const s of signals) {
    if (s.signal_id === 'composite_score') continue
    const category = classifySignal(s)
    if (category === 'confident') confident.push(s)
    else if (category === 'uncertain') uncertain.push(s)
    else moderate.push(s)
  }

  const groups: BehaviorGroup[] = []

  if (confident.length > 0) {
    groups.push({
      label: 'High Confidence Signals',
      color: 'text-green-400',
      border: 'border-green-400',
      description: `${confident.length} signals indicate the model is confident`,
      signals: confident,
    })
  }

  if (moderate.length > 0) {
    groups.push({
      label: 'Mixed / Moderate Signals',
      color: 'text-yellow-400',
      border: 'border-yellow-400',
      description: `${moderate.length} signals show moderate certainty`,
      signals: moderate,
    })
  }

  if (uncertain.length > 0) {
    groups.push({
      label: 'Uncertainty Indicators',
      color: 'text-red-400',
      border: 'border-red-400',
      description: `${uncertain.length} signals suggest uncertainty — verify these areas`,
      signals: uncertain,
    })
  }

  return groups
}

export default function SignalDashboard({ signals, isStreaming, theme, groupingMode, onGroupingChange }: Props) {
  const t = theme.copy
  const signalList = Object.values(signals)
  const compositeSignal = signals['composite_score']
  const compositeScore = compositeSignal && typeof compositeSignal.value === 'number' ? compositeSignal.value : null

  const signalsBySchool = (school: SchoolId) =>
    signalList.filter(s => s.school === school)

  const isSchoolComputing = (school: SchoolId) => {
    if (!isStreaming) return false
    const schoolSignals = signalsBySchool(school)
    return schoolSignals.length > 0 && schoolSignals.length < SCHOOL_COUNTS[school]
  }

  const behaviorGroups = groupByBehavior(signalList)

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-slate-700/50 flex justify-between items-center">
        <div>
          <span className={`${theme.colors.text} font-bold text-sm`}>{t.dashboard}</span>
          <span className={`${theme.colors.textDim} text-[10px] ml-2`}>18 signals across 5 schools</span>
        </div>
        <div className="flex gap-1.5 text-[11px]">
          <button
            onClick={() => onGroupingChange('school')}
            className={`px-2.5 py-1 rounded ${groupingMode === 'school' ? `${theme.colors.card} ${theme.colors.accent}` : theme.colors.textDim}`}
          >
            {t.bySchool}
          </button>
          <button
            onClick={() => onGroupingChange('behavior')}
            className={`px-2.5 py-1 rounded ${groupingMode === 'behavior' ? `${theme.colors.card} ${theme.colors.accent}` : theme.colors.textDim}`}
          >
            {t.byBehavior}
          </button>
        </div>
      </div>

      <CompositeScore
        score={compositeScore}
        totalSignals={18}
        computedSignals={signalList.length}
        theme={theme}
      />

      <div className="flex-1 overflow-y-auto p-4">
        {signalList.length === 0 && !isStreaming ? (
          <div className={`text-center ${theme.colors.textDim} text-sm py-20`}>
            Ask a question to see signals
          </div>
        ) : groupingMode === 'school' ? (
          SCHOOL_ORDER.map(school => (
            <SchoolGroup
              key={school}
              schoolId={school}
              signals={signalsBySchool(school)}
              isComputing={isSchoolComputing(school) || (isStreaming && signalsBySchool(school).length === 0)}
              theme={theme}
            />
          ))
        ) : (
          <>
            {behaviorGroups.length === 0 && isStreaming && (
              <div className={`text-center ${theme.colors.textDim} text-sm py-10 italic`}>
                Waiting for signals to classify...
              </div>
            )}
            {behaviorGroups.map(group => (
              <div key={group.label} className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className={`w-0.5 h-4 ${group.border} rounded`} />
                  <span className={`${group.color} font-bold text-xs tracking-wide`}>
                    {group.label}
                  </span>
                  <span className={`${theme.colors.textDim} text-[10px]`}>
                    {group.description}
                  </span>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
                  {group.signals.map(signal => (
                    <SignalCard key={signal.signal_id} signal={signal} theme={theme} />
                  ))}
                </div>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  )
}
