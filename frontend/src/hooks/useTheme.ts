import { useState, useEffect } from 'react'
import { researchTheme, type Theme } from '../themes/research'
import { oracleTheme } from '../themes/oracle'
import type { ThemeMode } from '../types/signals'

const STORAGE_KEY = 'ai-signals-theme'

export function useTheme() {
  const [mode, setMode] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    return (saved === 'oracle' ? 'oracle' : 'research') as ThemeMode
  })

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, mode)
  }, [mode])

  const toggle = () => setMode(m => (m === 'research' ? 'oracle' : 'research'))

  const theme: Theme = mode === 'oracle' ? oracleTheme : researchTheme

  return { mode, theme, toggle }
}
