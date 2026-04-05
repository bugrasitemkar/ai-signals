import type { ThemeMode } from '../types/signals'

interface Props {
  mode: ThemeMode
  onToggle: () => void
}

export default function ThemeToggle({ mode, onToggle }: Props) {
  return (
    <button
      onClick={onToggle}
      className="fixed bottom-4 right-4 text-xl opacity-40 hover:opacity-80 transition-opacity z-50"
    >
      {mode === 'research' ? '🔮' : '🔍'}
    </button>
  )
}
