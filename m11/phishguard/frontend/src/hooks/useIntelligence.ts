/**
 * useIntelligence.ts
 * Shared data-fetching hook for all M1-M10 intelligence API endpoints.
 * Returns live data when backend is available, falls back to known results.
 */

import { useState, useEffect } from 'react'
import {
  getShapData,
  getLimeData,
  getBiasData,
  getBlindspotData,
  getModelMetrics,
  getReliabilityData,
  getArchetypeData,
  getDatasetStats,
} from '@/lib/api'

export type FetchState<T> = {
  data: T | null
  loading: boolean
  error: string | null
  source: 'api' | 'fallback' | null
}

function useFetch<T>(
  fetcher: () => Promise<T>,
  fallback: T,
  deps: unknown[] = []
): FetchState<T> {
  const [state, setState] = useState<FetchState<T>>({
    data: fallback,
    loading: true,
    error: null,
    source: null,
  })

  useEffect(() => {
    let cancelled = false
    setState((s) => ({ ...s, loading: true, error: null }))

    fetcher()
      .then((data) => {
        if (!cancelled) {
          setState({ data, loading: false, error: null, source: 'api' })
        }
      })
      .catch(() => {
        if (!cancelled) {
          setState({ data: fallback, loading: false, error: 'Using cached results', source: 'fallback' })
        }
      })

    return () => { cancelled = true }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return state
}

// ── Per-endpoint hooks ────────────────────────────────────────────────────────

export function useShapData() {
  return useFetch(getShapData, {
    source: 'fallback',
    features: [
      { rank: 1,  feature: 'LetterRatioInURL',           importance_pct: 10.51 },
      { rank: 2,  feature: 'LineOfCode',                  importance_pct: 10.46 },
      { rank: 3,  feature: 'IsHTTPS',                     importance_pct:  9.26 },
      { rank: 4,  feature: 'NoOfDegitsInURL',             importance_pct:  8.65 },
      { rank: 5,  feature: 'DomainLength',                importance_pct:  7.09 },
      { rank: 6,  feature: 'NoOfSelfRef',                 importance_pct:  5.80 },
      { rank: 7,  feature: 'NoOfOtherSpecialCharsInURL',  importance_pct:  4.90 },
      { rank: 8,  feature: 'LargestLineLength',           importance_pct:  4.40 },
      { rank: 9,  feature: 'NoOfExternalRef',             importance_pct:  4.00 },
      { rank: 10, feature: 'SpacialCharRatioInURL',       importance_pct:  3.50 },
    ],
  })
}

export function useLimeData() {
  return useFetch(getLimeData, {
    source: 'fallback',
    summary: {
      mean_agreement: 0.52,
      feature_consistency: 0.60,
      shared_top20: 12,
      local_agreement_pct: 0,
    },
    agreement_data: [],
  })
}

export function useModelData() {
  return useFetch(getModelMetrics, {
    source: 'fallback',
    models: {
      track_A: [
        { name: 'Logistic Regression', accuracy: 99.9958, f1: 99.9958, roc_auc: 1.00 },
        { name: 'Random Forest',       accuracy: 100.00,  f1: 100.00,  roc_auc: 1.00 },
        { name: 'XGBoost',             accuracy: 99.9958, f1: 99.9958, roc_auc: 1.00 },
        { name: 'LightGBM',            accuracy: 100.00,  f1: 100.00,  roc_auc: 1.00 },
      ],
      track_B: [
        { name: 'LightGBM',            accuracy: 99.9936, f1: 99.9936, roc_auc: 1.00, deploy: true },
        { name: 'Logistic Regression', accuracy: 99.9936, f1: 99.9936, roc_auc: 1.00 },
        { name: 'XGBoost',             accuracy: 99.9894, f1: 99.9894, roc_auc: 1.00 },
        { name: 'Random Forest',       accuracy: 99.9851, f1: 99.9851, roc_auc: 1.00 },
      ],
      deployment: { model: 'LightGBM', track: 'B', accuracy: 99.9936, roc_auc: 1.00 },
    },
    raw_metrics: [],
  })
}

export function useBlindspotData() {
  return useFetch(getBlindspotData, {
    top3: [
      { rank: 1, sample_id: 17372, type: 'False Negative', confidence: 62.61, severity_score: 0.712, risk: 'CRITICAL' },
      { rank: 2, sample_id: 11301, type: 'False Negative', confidence: 87.46, severity_score: 0.638, risk: 'CRITICAL' },
      { rank: 3, sample_id: 30588, type: 'False Negative', confidence: 88.35, severity_score: 0.635, risk: 'CRITICAL' },
    ],
    top20: [],
    severity: [],
    archetypes: [
      { name: 'Archetype Alpha', signals: ['Non-HTTPS', 'Gov/Edu Domain', 'Password Form'] },
      { name: 'Archetype Beta',  signals: ['HTTPS',     'Gov/Edu Domain', 'Social Linked'] },
      { name: 'Archetype Gamma', signals: ['Non-HTTPS', 'Gov/Edu Domain', 'Social Linked'] },
    ],
  })
}

export function useBiasData() {
  return useFetch(getBiasData, {
    source: 'fallback',
    summary: {
      dimensions: [
        { name: 'URL Length Groups',        min_performance: 99.98, status: 'PASS', most_biased: false },
        { name: 'Domain Length Groups',     min_performance: 99.98, status: 'PASS', most_biased: false },
        { name: 'HTTPS Groups',             min_performance: 99.99, status: 'PASS', most_biased: false },
        { name: 'TLD Groups',               min_performance: 99.99, status: 'PASS', most_biased: true  },
        { name: 'External Resource Groups', min_performance: 99.98, status: 'PASS', most_biased: false },
      ],
      verdict: 'NO_SIGNIFICANT_DRIFT',
      violations: 0,
    },
    metrics: [],
    disparities: [],
    shap_bias: [],
  })
}

export function useReliabilityData() {
  return useFetch(getReliabilityData, {
    source: 'fallback',
    summary: {
      zones: [
        { zone: 'GREEN',  agreement_range: '0.8-1.0', samples: 'Majority', error_rate: 0.0,   mean_confidence: null },
        { zone: 'YELLOW', agreement_range: '0.2-0.8', samples: 'Moderate', error_rate: 5.0,   mean_confidence: null },
        { zone: 'RED',    agreement_range: '0.0-0.2', samples: 23,         error_rate: 13.04, mean_confidence: 97.31 },
      ],
      key_finding: 'Lower explanation agreement correlates with increased prediction risk despite high model confidence.',
    },
    bin_stats: [],
  })
}

export function useArchetypeData() {
  return useFetch(getArchetypeData, {
    source: 'fallback',
    archetypes: [
      { name: 'Archetype Alpha', signals: ['Non-HTTPS', 'Gov/Edu Domain', 'Password Form'] },
      { name: 'Archetype Beta',  signals: ['HTTPS',     'Gov/Edu Domain', 'Social Linked'] },
      { name: 'Archetype Gamma', signals: ['Non-HTTPS', 'Gov/Edu Domain', 'Social Linked'] },
    ],
  })
}

export function useDatasetStats() {
  return useFetch(getDatasetStats, {
    source: 'fallback',
    dataset: {
      name: 'PhiUSIIL Phishing URL Dataset',
      rows: 235795,
      features: 56,
      missing_values: 0,
      legitimate_pct: 57.19,
      phishing_pct: 42.81,
    },
    overview: [],
  })
}
