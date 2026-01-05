import fs from 'node:fs/promises'
import path from 'node:path'
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'

import jpeg from 'jpeg-js'
import { PNG } from 'pngjs'
import * as ort from 'onnxruntime-web'

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

function isPng(buffer) {
  return (
    buffer?.length >= 8 &&
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47 &&
    buffer[4] === 0x0d &&
    buffer[5] === 0x0a &&
    buffer[6] === 0x1a &&
    buffer[7] === 0x0a
  )
}

function decodeToRgba(imageBuffer) {
  if (isPng(imageBuffer)) {
    const decoded = PNG.sync.read(imageBuffer)
    return { width: decoded.width, height: decoded.height, data: decoded.data }
  }

  const decoded = jpeg.decode(imageBuffer, { useTArray: true })
  if (!decoded?.data || !decoded.width || !decoded.height) {
    throw new Error('Failed to decode image. Only JPEG/PNG supported.')
  }
  return { width: decoded.width, height: decoded.height, data: decoded.data }
}

function rgbaToRgb({ width, height, data }) {
  const rgb = new Uint8Array(width * height * 3)
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j] = data[i]
    rgb[j + 1] = data[i + 1]
    rgb[j + 2] = data[i + 2]
  }
  return { width, height, data: rgb }
}

function resizeBilinearRgb(src, srcW, srcH, dstW, dstH) {
  const dst = new Uint8Array(dstW * dstH * 3)
  const xScale = srcW / dstW
  const yScale = srcH / dstH

  for (let y = 0; y < dstH; y += 1) {
    const sy = (y + 0.5) * yScale - 0.5
    const y0 = Math.max(0, Math.floor(sy))
    const y1 = Math.min(srcH - 1, y0 + 1)
    const wy = sy - y0

    for (let x = 0; x < dstW; x += 1) {
      const sx = (x + 0.5) * xScale - 0.5
      const x0 = Math.max(0, Math.floor(sx))
      const x1 = Math.min(srcW - 1, x0 + 1)
      const wx = sx - x0

      const i00 = (y0 * srcW + x0) * 3
      const i10 = (y0 * srcW + x1) * 3
      const i01 = (y1 * srcW + x0) * 3
      const i11 = (y1 * srcW + x1) * 3

      const o = (y * dstW + x) * 3
      for (let c = 0; c < 3; c += 1) {
        const v00 = src[i00 + c]
        const v10 = src[i10 + c]
        const v01 = src[i01 + c]
        const v11 = src[i11 + c]

        const v0 = v00 + (v10 - v00) * wx
        const v1 = v01 + (v11 - v01) * wx
        const v = v0 + (v1 - v0) * wy

        dst[o + c] = Math.max(0, Math.min(255, Math.round(v)))
      }
    }
  }

  return dst
}

function centerCropRgb(rgb, width, height, cropSize) {
  const left = Math.floor((width - cropSize) / 2)
  const top = Math.floor((height - cropSize) / 2)
  const out = new Uint8Array(cropSize * cropSize * 3)

  for (let y = 0; y < cropSize; y += 1) {
    const sy = top + y
    for (let x = 0; x < cropSize; x += 1) {
      const sx = left + x
      const srcIdx = (sy * width + sx) * 3
      const dstIdx = (y * cropSize + x) * 3
      out[dstIdx] = rgb[srcIdx]
      out[dstIdx + 1] = rgb[srcIdx + 1]
      out[dstIdx + 2] = rgb[srcIdx + 2]
    }
  }

  return out
}

function rgbToNchwFloat32(rgb, width, height) {
  const mean = [0.485, 0.456, 0.406]
  const std = [0.229, 0.224, 0.225]

  const hw = width * height
  const out = new Float32Array(3 * hw)

  for (let i = 0; i < hw; i += 1) {
    const r = rgb[i * 3] / 255
    const g = rgb[i * 3 + 1] / 255
    const b = rgb[i * 3 + 2] / 255

    out[i] = (r - mean[0]) / std[0]
    out[hw + i] = (g - mean[1]) / std[1]
    out[2 * hw + i] = (b - mean[2]) / std[2]
  }

  return out
}

export async function getSessionAndLabels() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const { modelPath, modelUrl, labelsUrl } = getModelConfig()
      await ensureModelFiles(modelPath, modelUrl, labelsUrl)

      // Use WASM runtime for smaller serverless bundles.
      // In Node.js (Vercel Serverless), WASM assets must be loaded from local file paths.
      // Node's ESM loader does not support importing from https:// URLs.
      const require = createRequire(import.meta.url)
      const ortPkgJson = require.resolve('onnxruntime-web/package.json')
      const ortBaseDir = path.dirname(ortPkgJson)
      const ortDistDir = path.join(ortBaseDir, 'dist')
      ort.env.wasm.wasmPaths = pathToFileURL(ortDistDir + path.sep).href
      ort.env.wasm.numThreads = 1

      const modelBytes = await fs.readFile(modelPath)
      const session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
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
  const decoded = decodeToRgba(imageBuffer)
  const rgb = rgbaToRgb(decoded)

  const resizedRgb = resizeBilinearRgb(rgb.data, rgb.width, rgb.height, 256, 256)
  const croppedRgb = centerCropRgb(resizedRgb, 256, 256, 224)
  const nchw = rgbToNchwFloat32(croppedRgb, 224, 224)

  return new ort.Tensor('float32', nchw, [1, 3, 224, 224])
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
