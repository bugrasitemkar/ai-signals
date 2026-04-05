export interface SignalResult {
  signal_id: string
  school: SchoolId
  value: number | number[] | Record<string, unknown>
  interpretation: string
  metadata?: Record<string, unknown>
}

export interface SummaryResult {
  type: 'summary'
  composite_score: number
  executive_summary: string
  behavioral_groups?: BehavioralGroup[]
}

export interface BehavioralGroup {
  theme: string
  signal_ids: string[]
}

export type SchoolId =
  | 'information_theoretic'
  | 'layer_wise'
  | 'geometric'
  | 'behavioral'
  | 'calibration'

export interface SignalDefinition {
  id: string
  name: string
  school: SchoolId
  formula: string
  briefDescription: string
  thresholds: { low: number; high: number; inverted?: boolean }
  detailPage: {
    whatItIs: string
    howWeCompute: string
    codeSnippet: string
    limitations: string[]
    whenToUse: string
    references: { title: string; url?: string }[]
  }
}

export interface ChatResponse {
  response: string
  request_id: string
  model: string
  generation_time_ms: number
}

export interface HealthStatus {
  ollama: { status: string; model: string | null }
  huggingface: { status: string; model: string | null }
}

export type LayoutMode = 'split' | 'single'
export type GroupingMode = 'school' | 'behavior'
export type ThemeMode = 'research' | 'oracle'
