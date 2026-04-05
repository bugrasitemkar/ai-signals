import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import type { SignalResult } from '../types/signals'
import type { Theme } from '../themes/research'

interface Props {
  signals: Record<string, SignalResult>
  theme: Theme
}

export default function SignalChart({ signals, theme }: Props) {
  const numericSignals = Object.entries(signals)
    .filter(([_, s]) => typeof s.value === 'number' && s.signal_id !== 'composite_score')
    .map(([id, s]) => ({
      name: id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()).substring(0, 15),
      value: s.value as number,
      school: s.school,
    }))

  if (numericSignals.length === 0) return null

  return (
    <div className={`${theme.colors.card} border ${theme.colors.cardBorder} rounded-md p-3 mb-4`}>
      <div className={`text-[11px] font-bold ${theme.colors.textMuted} mb-2`}>Signal Overview</div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={numericSignals} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <XAxis dataKey="name" tick={{ fontSize: 8, fill: '#64748b' }} angle={-45} textAnchor="end" height={60} />
          <YAxis tick={{ fontSize: 9, fill: '#64748b' }} />
          <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', fontSize: 11 }} />
          <Bar dataKey="value" fill="#818cf8" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
