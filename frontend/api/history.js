import { getSupabaseClient, json } from './_shared.js'

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return json(res, 405, { detail: 'Method not allowed' })
  }

  const supabase = getSupabaseClient()
  if (!supabase) {
    return json(res, 503, { detail: 'Supabase logging is not configured' })
  }

  const url = new URL(req.url, `http://${req.headers.host}`)
  const limitRaw = url.searchParams.get('limit')
  const source = url.searchParams.get('source')

  const limitParsed = Number.parseInt(limitRaw ?? '20', 10)
  const limit = Number.isFinite(limitParsed) ? Math.max(1, Math.min(limitParsed, 100)) : 20

  const table = process.env.SUPABASE_TABLE || 'prediction_logs'

  try {
    let query = supabase
      .from(table)
      .select('*')
      .order('created_at', { ascending: false })
      .limit(limit)

    if (source && source.toLowerCase() !== 'all') {
      query = query.eq('source', source)
    }

    const { data, error } = await query
    if (error) {
      throw new Error(error.message)
    }

    return json(res, 200, { entries: data ?? [] })
  } catch (err) {
    return json(res, 500, { detail: `Failed to fetch history: ${err instanceof Error ? err.message : String(err)}` })
  }
}
