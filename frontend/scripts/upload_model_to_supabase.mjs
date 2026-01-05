import fs from 'node:fs/promises'
import path from 'node:path'

import { createClient } from '@supabase/supabase-js'

function requiredEnv(name) {
  const value = process.env[name]
  if (!value) {
    throw new Error(`Missing required env var ${name}`)
  }
  return value
}

async function fileToBlob(filePath) {
  const buf = await fs.readFile(filePath)
  return new Blob([buf], { type: 'application/octet-stream' })
}

async function main() {
  const supabaseUrl = requiredEnv('SUPABASE_URL')
  const supabaseServiceKey = requiredEnv('SUPABASE_SERVICE_KEY')
  const bucket = process.env.SUPABASE_BUCKET || 'models'

  const modelFile = process.env.MODEL_FILE || path.resolve('..', 'best_model.onnx')
  const labelsFile = process.env.LABELS_FILE || path.resolve('..', 'best_model.labels.json')

  const modelObjectPath = process.env.MODEL_OBJECT_PATH || 'best_model.onnx'
  const labelsObjectPath = process.env.LABELS_OBJECT_PATH || 'best_model.labels.json'

  const supabase = createClient(supabaseUrl, supabaseServiceKey, {
    auth: { persistSession: false },
  })

  console.log(`Uploading to bucket '${bucket}'...`)

  const modelBlob = await fileToBlob(modelFile)
  const { error: modelError } = await supabase.storage
    .from(bucket)
    .upload(modelObjectPath, modelBlob, {
      upsert: true,
      contentType: 'application/octet-stream',
    })

  if (modelError) {
    throw new Error(`Model upload failed: ${modelError.message}`)
  }

  const labelsBlob = await fileToBlob(labelsFile)
  const { error: labelsError } = await supabase.storage
    .from(bucket)
    .upload(labelsObjectPath, labelsBlob, {
      upsert: true,
      contentType: 'application/json',
    })

  if (labelsError) {
    throw new Error(`Labels upload failed: ${labelsError.message}`)
  }

  console.log('Upload complete.')

  // For a PUBLIC bucket, this is the direct URL format.
  const projectOrigin = new URL(supabaseUrl).origin
  const modelPublicUrl = `${projectOrigin}/storage/v1/object/public/${bucket}/${modelObjectPath}`
  const labelsPublicUrl = `${projectOrigin}/storage/v1/object/public/${bucket}/${labelsObjectPath}`

  console.log('If your bucket is PUBLIC, use these URLs in Vercel env vars:')
  console.log(`MODEL_URL=${modelPublicUrl}`)
  console.log(`LABELS_URL=${labelsPublicUrl}`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
