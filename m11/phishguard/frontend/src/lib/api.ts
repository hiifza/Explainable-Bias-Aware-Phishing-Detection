import axios from 'axios'
import type { ScanResult } from '@/store'

const BASE = import.meta.env.VITE_API_URL ?? '/api'

export const api = axios.create({
  baseURL: BASE,
  timeout: 30_000,
})

export async function analyzeUrl(url: string): Promise<ScanResult> {
  const { data } = await api.post<ScanResult>('/analyze', { url })
  return data
}

export async function getReports() {
  const { data } = await api.get('/reports')
  return data
}

export async function getShapData() {
  const { data } = await api.get('/intelligence/shap')
  return data
}

export async function getLimeData() {
  const { data } = await api.get('/intelligence/lime')
  return data
}

export async function getBiasData() {
  const { data } = await api.get('/intelligence/bias')
  return data
}

export async function getBlindspotData() {
  const { data } = await api.get('/intelligence/blindspots')
  return data
}

export async function getModelMetrics() {
  const { data } = await api.get('/intelligence/models')
  return data
}

export async function getReliabilityData() {
  const { data } = await api.get('/intelligence/reliability')
  return data
}

export async function getArchetypeData() {
  const { data } = await api.get('/intelligence/archetypes')
  return data
}

export async function getDatasetStats() {
  const { data } = await api.get('/intelligence/dataset')
  return data
}
