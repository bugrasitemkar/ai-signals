import ReactMarkdown from 'react-markdown'
import type { SummaryResult } from '../types/signals'
import type { Theme } from '../themes/research'

interface Props {
  theme: Theme
  summary: SummaryResult
}

export default function ExecutiveSummary({ theme, summary }: Props) {
  const t = theme.copy

  return (
    <div className={`border ${theme.colors.cardBorder} rounded-lg overflow-hidden`}>
      <div className={`px-3 py-2 ${theme.colors.card} border-b ${theme.colors.cardBorder} flex justify-between items-center`}>
        <span className={`text-xs font-bold ${theme.colors.accent}`}>{t.summary}</span>
        <span className={`text-[9px] ${theme.colors.textDim} italic`}>{t.summaryDisclaimer}</span>
      </div>
      <div className={`p-3 text-xs ${theme.colors.textMuted} leading-relaxed prose prose-invert prose-xs max-w-none`}>
        <ReactMarkdown>{summary.executive_summary}</ReactMarkdown>
        <div className={`mt-2 text-right text-[10px] ${theme.colors.textDim}`}>
          Reliability: <span className="text-green-400 font-bold">{Math.round(summary.composite_score)} / 100</span>
        </div>
      </div>
    </div>
  )
}
