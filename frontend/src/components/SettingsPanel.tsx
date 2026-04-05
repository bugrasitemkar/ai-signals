import type { LayoutMode } from '../types/signals'
import type { Theme } from '../themes/research'

interface Props {
  layoutMode: LayoutMode
  onLayoutChange: (mode: LayoutMode) => void
  theme: Theme
}

export default function SettingsPanel({ layoutMode, onLayoutChange, theme }: Props) {
  return (
    <div className={`px-4 py-2 border-b border-slate-700/30 flex items-center gap-4 text-[11px] ${theme.colors.textDim}`}>
      <span>Layout:</span>
      <button
        onClick={() => onLayoutChange('split')}
        className={layoutMode === 'split' ? theme.colors.accent : ''}
      >
        Split Panel
      </button>
      <span>|</span>
      <button
        onClick={() => onLayoutChange('single')}
        className={layoutMode === 'single' ? theme.colors.accent : ''}
      >
        Single Page
      </button>
    </div>
  )
}
