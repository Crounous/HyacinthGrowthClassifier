import { getSupabaseClient, json, normalizeAuthorityEmail } from '../_shared.js'

const AUTHORITY_EMAIL_KEY = 'authority_email'

async function fetchSettingValue(supabase, table, key) {
  const { data, error } = await supabase.from(table).select('value').eq('key', key).limit(1)
  if (error) throw new Error(error.message)
  const row = data?.[0]
  return row?.value ?? null
}

async function upsertSettingValue(supabase, table, key, value) {
  const { error } = await supabase.from(table).upsert({ key, value })
  if (error) throw new Error(error.message)
}

async function deleteSettingValue(supabase, table, key) {
  const { error } = await supabase.from(table).delete().eq('key', key)
  if (error) throw new Error(error.message)
}

export default async function handler(req, res) {
  const supabase = getSupabaseClient()
  if (!supabase) {
    return json(res, 503, { detail: 'Supabase logging is not configured' })
  }

  const settingsTable = process.env.SUPABASE_SETTINGS_TABLE || 'settings'

  if (req.method === 'GET') {
    try {
      const authorityEmail = await fetchSettingValue(supabase, settingsTable, AUTHORITY_EMAIL_KEY)
      return json(res, 200, { authority_email: authorityEmail })
    } catch (err) {
      return json(res, 500, { detail: `Failed to fetch authority email: ${err instanceof Error ? err.message : String(err)}` })
    }
  }

  if (req.method === 'POST') {
    try {
      const chunks = []
      for await (const chunk of req) chunks.push(chunk)
      const raw = Buffer.concat(chunks).toString('utf-8')
      const payload = raw ? JSON.parse(raw) : {}

      const trimmed = String(payload.authority_email ?? '').trim()
      if (!trimmed) {
        await deleteSettingValue(supabase, settingsTable, AUTHORITY_EMAIL_KEY)
        return json(res, 200, { authority_email: null })
      }

      const normalized = normalizeAuthorityEmail(trimmed)
      await upsertSettingValue(supabase, settingsTable, AUTHORITY_EMAIL_KEY, normalized)
      return json(res, 200, { authority_email: normalized })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      const status = message.startsWith('Authority ') ? 400 : 500
      return json(res, status, { detail: message })
    }
  }

  return json(res, 405, { detail: 'Method not allowed' })
}
