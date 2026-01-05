import Busboy from 'busboy'

import { argmax1d, getSessionAndLabels, preprocessToNCHWFloat32 } from './_model.js'
import { getSupabaseClient, json, normalizeAuthorityEmail, normalizeAuthorityNumber } from './_shared.js'

export const config = {
  api: { bodyParser: false },
}

export default function handler(req, res) {
  if (req.method !== 'POST') {
    return json(res, 405, { detail: 'Method not allowed' })
  }

  const url = new URL(req.url, `http://${req.headers.host}`)
  const source = url.searchParams.get('source') || 'upload'

  const bb = Busboy({ headers: req.headers })

  const fields = {}
  const fileChunks = []
  let filename = null

  bb.on('field', (name, value) => {
    fields[name] = value
  })

  bb.on('file', (_name, file, info) => {
    filename = info?.filename || 'upload'
    file.on('data', (chunk) => fileChunks.push(chunk))
  })

  bb.on('error', (err) => {
    json(res, 400, { detail: `Invalid multipart request: ${err?.message ?? String(err)}` })
  })

  bb.on('finish', async () => {
    try {
      const imageBytes = Buffer.concat(fileChunks)
      if (!imageBytes.length) {
        return json(res, 400, { detail: 'No file uploaded.' })
      }

      let normalizedAuthority = null
      if (fields.authority_number) {
        normalizedAuthority = normalizeAuthorityNumber(fields.authority_number)
      }

      let normalizedEmail = null
      if (fields.authority_email) {
        normalizedEmail = normalizeAuthorityEmail(fields.authority_email)
      }

      const { session, inputName, outputName, labels, modelPath } = await getSessionAndLabels()
      const inputTensor = await preprocessToNCHWFloat32(imageBytes)

      const outputs = await session.run({ [inputName]: inputTensor })
      const logits = outputs[outputName]?.data
      if (!logits) {
        return json(res, 500, { detail: 'Model did not return outputs.' })
      }

      const labelIdx = argmax1d(logits)
      const prediction = labels[labelIdx] ?? labels[0] ?? 'No Growth'

      let status = 'Normal'
      let alert = false
      if (prediction === 'Large Growth') {
        status = 'Warning'
        alert = true
      }

      const supabase = getSupabaseClient()
      if (supabase) {
        const table = process.env.SUPABASE_TABLE || 'prediction_logs'
        const payload = {
          prediction,
          status,
          alert,
          source,
          filename,
          model_path: modelPath,
          file_size: imageBytes.length,
          logged_at: new Date().toISOString(),
          authority_number: normalizedAuthority,
          authority_email: normalizedEmail,
        }

        await supabase.from(table).insert(payload)
      }

      return json(res, 200, { prediction, status, alert })
    } catch (err) {
      return json(res, 500, { detail: err instanceof Error ? err.message : String(err) })
    }
  })

  req.pipe(bb)
}
