import { useParams, Link } from 'react-router-dom'
import { useTheme } from '../hooks/useTheme'
import { getSignalById } from '../data/signalDefinitions'
import ThemeToggle from '../components/ThemeToggle'

export default function SignalDetailPage() {
  const { signalId } = useParams<{ signalId: string }>()
  const { mode, theme, toggle } = useTheme()
  const definition = signalId ? getSignalById(signalId) : undefined

  if (!definition) {
    return (
      <div className={`min-h-screen ${theme.colors.bg} p-8`}>
        <Link to="/" className={`${theme.colors.accent} text-sm`}>← Back to dashboard</Link>
        <div className={`${theme.colors.text} mt-8`}>Signal not found</div>
      </div>
    )
  }

  const schoolConfig = theme.schools[definition.school]

  return (
    <div className={`min-h-screen ${theme.colors.bg} p-8`}>
      <div className="max-w-2xl mx-auto">
        <Link to="/" className={`${theme.colors.textDim} text-xs hover:${theme.colors.textMuted}`}>
          ← Back to dashboard
        </Link>

        <div className="border-b border-slate-700/50 pb-4 mt-4 mb-6">
          <div className={`text-[10px] ${schoolConfig.color} tracking-widest uppercase mb-1`}>
            {schoolConfig.name}
          </div>
          <h1 className={`text-2xl font-bold ${theme.colors.text}`}>{definition.name}</h1>
          <p className={`text-sm ${theme.colors.textMuted} mt-1`}>{definition.briefDescription}</p>
        </div>

        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>What It Is</h2>
          <p className={`text-sm ${theme.colors.text} leading-relaxed`}>{definition.detailPage.whatItIs}</p>
        </section>

        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>The Formula</h2>
          <div className={`${theme.colors.card} border ${theme.colors.cardBorder} p-3 rounded text-center font-mono text-sm ${theme.colors.text}`}>
            {definition.formula}
          </div>
        </section>

        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>How We Compute It</h2>
          <p className={`text-sm ${theme.colors.text} leading-relaxed mb-2`}>{definition.detailPage.howWeCompute}</p>
          <pre className={`${theme.colors.card} border ${theme.colors.cardBorder} p-3 rounded text-xs font-mono text-indigo-300 overflow-x-auto`}>
            {definition.detailPage.codeSnippet}
          </pre>
        </section>

        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>Interpretation</h2>
          <div className={`${theme.colors.card} border ${theme.colors.cardBorder} rounded p-3 text-sm space-y-1`}>
            <div className="flex gap-3">
              <span className="text-green-400 font-bold w-16">{'< '}{definition.thresholds.low}</span>
              <span className={theme.colors.text}>{definition.thresholds.inverted ? 'Uncertain' : 'Confident'}</span>
            </div>
            <div className="flex gap-3">
              <span className="text-yellow-400 font-bold w-16">{definition.thresholds.low}–{definition.thresholds.high}</span>
              <span className={theme.colors.text}>Moderate</span>
            </div>
            <div className="flex gap-3">
              <span className="text-red-400 font-bold w-16">{'> '}{definition.thresholds.high}</span>
              <span className={theme.colors.text}>{definition.thresholds.inverted ? 'Confident' : 'Uncertain'}</span>
            </div>
          </div>
        </section>

        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>Limitations</h2>
          <ul className={`text-sm ${theme.colors.textMuted} space-y-1.5 leading-relaxed`}>
            {definition.detailPage.limitations.map((lim, i) => (
              <li key={i}>• {lim}</li>
            ))}
          </ul>
        </section>

        <section className="mb-6">
          <h2 className={`text-sm font-bold ${theme.colors.accent} mb-2`}>When To Use</h2>
          <p className={`text-sm ${theme.colors.text} leading-relaxed`}>{definition.detailPage.whenToUse}</p>
        </section>

        <section className="border-t border-slate-700/50 pt-4 mt-8">
          <h2 className={`text-xs ${theme.colors.textDim} mb-2`}>References</h2>
          {definition.detailPage.references.map((ref, i) => (
            <div key={i} className={`text-xs ${theme.colors.textDim} mb-1`}>
              {ref.url ? (
                <a href={ref.url} target="_blank" rel="noopener noreferrer" className={`${theme.colors.accent} hover:underline`}>
                  {ref.title}
                </a>
              ) : (
                ref.title
              )}
            </div>
          ))}
        </section>
      </div>
      <ThemeToggle mode={mode} onToggle={toggle} />
    </div>
  )
}
