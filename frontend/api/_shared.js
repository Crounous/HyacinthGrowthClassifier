import fs from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'

import { createClient } from '@supabase/supabase-js'

export const DEFAULT_LABELS = [
  'No Growth',
  'Low Growth',
  'Moderate Growth',
  'Large Growth',
]

export function getSupabaseClient() {
  const supabaseUrl = process.env.SUPABASE_URL
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_KEY

  if (!supabaseUrl || !supabaseServiceKey) {
    return null
  }

  return createClient(supabaseUrl, supabaseServiceKey, {
    auth: { persistSession: false },
  })
}

export function json(res, statusCode, payload) {
  res.statusCode = statusCode
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.end(JSON.stringify(payload))
}

export function normalizeAuthorityNumber(rawNumber) {
  const digits = String(rawNumber ?? '').replace(/\D/g, '')
  if (!digits) {
    throw new Error('Authority number is required.')
  }

  let normalized = digits
  if (normalized.startsWith('63')) {
    normalized = normalized.slice(2)
  } else if (normalized.startsWith('0')) {
    normalized = normalized.slice(1)
  }

  if (normalized.length !== 10) {
    throw new Error('Authority number must be a Philippine number (+63 followed by 10 digits).')
  }

  return `+63${normalized}`
}

export function normalizeAuthorityEmail(rawEmail) {
  const email = String(rawEmail ?? '').trim().toLowerCase()
  if (!email) {
    throw new Error('Authority email is required.')
  }

  if (!email.includes('@')) {
    throw new Error('Authority email must include @.')
  }

  const [localPart, domain] = email.split('@')
  if (!localPart || !domain || !domain.includes('.')) {
    throw new Error('Authority email must be a valid email address.')
  }

  return email
}

export function getModelConfig() {
  return {
    modelPath: process.env.MODEL_PATH || path.join(os.tmpdir(), 'best_model.onnx'),
    modelUrl: process.env.MODEL_URL || '',
    labelsUrl: process.env.LABELS_URL || '',
  }
}

function getLabelsPathForModelPath(modelPath) {
  return path.format({
    ...path.parse(modelPath),
    base: undefined,
    ext: '.labels.json',
  })
}

function getLabelsUrlForModelUrl(modelUrl) {
  if (!modelUrl) return ''
  if (/\.onnx($|\?)/i.test(modelUrl)) {
    return modelUrl.replace(/\.onnx(?=($|\?))/i, '.labels.json')
  }
  return ''
}

async function downloadToPath(url, outPath, friendlyName) {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(
      `Failed to download ${friendlyName}: ${response.status} ${response.statusText}`
    )
  }

  const arrayBuffer = await response.arrayBuffer()
  await fs.mkdir(path.dirname(outPath), { recursive: true })
  await fs.writeFile(outPath, Buffer.from(arrayBuffer))
}

export async function ensureModelFiles(modelPath, modelUrl, labelsUrl) {
  const labelsPath = getLabelsPathForModelPath(modelPath)

  let modelExists = true
  try {
    await fs.access(modelPath)
  } catch {
    modelExists = false
  }

  let labelsExist = true
  try {
    await fs.access(labelsPath)
  } catch {
    labelsExist = false
  }

  if (modelExists && labelsExist) {
    return
  }

  if (!modelExists) {
    if (!modelUrl) {
      throw new Error(
        'Model file missing and MODEL_URL is not set. ' +
          'Set MODEL_URL to a direct-download .onnx URL.'
      )
    }

    await downloadToPath(modelUrl, modelPath, 'model')
  }

  if (!labelsExist) {
    const resolvedLabelsUrl = labelsUrl || getLabelsUrlForModelUrl(modelUrl)
    if (!resolvedLabelsUrl) {
      // Labels are optional; _model.js will fall back to DEFAULT_LABELS.
      return
    }

    await downloadToPath(resolvedLabelsUrl, labelsPath, 'labels')
  }
}

// Backwards-compat for older imports.
export async function ensureModelFile(modelPath, modelUrl) {
  return ensureModelFiles(modelPath, modelUrl, '')
}
