import fs from 'node:fs/promises'
import path from 'node:path'

import sharp from 'sharp'
import * as ort from 'onnxruntime-node'

import { DEFAULT_LABELS, ensureModelFiles, getModelConfig } from './_shared.js'

let sessionPromise = null

async function loadLabels(modelPath) {
  const labelsPath = path.format({
    ...path.parse(modelPath),
    base: undefined,
    ext: '.labels.json',
  })

  try {
    const text = await fs.readFile(labelsPath, 'utf-8')
    const parsed = JSON.parse(text)
    if (Array.isArray(parsed) && parsed.every((x) => typeof x === 'string')) {
      return parsed
    }
  } catch {
    // ignore
  }

  return DEFAULT_LABELS
}

export async function getSessionAndLabels() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const { modelPath, modelUrl, labelsUrl } = getModelConfig()
      await ensureModelFiles(modelPath, modelUrl, labelsUrl)

      const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['cpu'],
      })

      const inputName = session.inputNames[0]
      const outputName = session.outputNames[0]
      const labels = await loadLabels(modelPath)

      return { session, inputName, outputName, labels, modelPath }
    })()
  }

  return sessionPromise
}

export async function preprocessToNCHWFloat32(imageBuffer) {
  const resized = sharp(imageBuffer)
    .removeAlpha()
    .toColourspace('rgb')
    .resize(256, 256, { fit: 'fill' })
    .extract({ left: 16, top: 16, width: 224, height: 224 })

  const { data, info } = await resized.raw().toBuffer({ resolveWithObject: true })
  if (info.channels !== 3) {
    throw new Error(`Unexpected channel count: ${info.channels}`)
  }

  const mean = [0.485, 0.456, 0.406]
  const std = [0.229, 0.224, 0.225]

  const hw = info.width * info.height
  const out = new Float32Array(3 * hw)

  for (let i = 0; i < hw; i += 1) {
    const r = data[i * 3] / 255
    const g = data[i * 3 + 1] / 255
    const b = data[i * 3 + 2] / 255

    out[i] = (r - mean[0]) / std[0]
    out[hw + i] = (g - mean[1]) / std[1]
    out[2 * hw + i] = (b - mean[2]) / std[2]
  }

  return new ort.Tensor('float32', out, [1, 3, info.height, info.width])
}

export function argmax1d(arrayLike) {
  let bestIndex = 0
  let bestValue = arrayLike[0]
  for (let i = 1; i < arrayLike.length; i += 1) {
    if (arrayLike[i] > bestValue) {
      bestValue = arrayLike[i]
      bestIndex = i
    }
  }
  return bestIndex
}
