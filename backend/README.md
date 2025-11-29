# Backend Service

FastAPI service that loads the plant classification model and now logs prediction metadata to Supabase for history tracking.

## Required Environment Variables

| Variable | Description |
| --- | --- |
| `MODEL_PATH` | Filesystem path to `best_model.pth`. Optional when using the default path. |
| `MODEL_URL` | Direct download URL for the model if it needs to be fetched dynamically. |
| `SUPABASE_URL` | Supabase project URL (e.g., `https://xyzcompany.supabase.co`). |
| `SUPABASE_SERVICE_KEY` | Service role key with insert/select access to the logging table. |
| `SUPABASE_TABLE` | Optional. Table name for logs (`prediction_logs` by default). |
| `SUPABASE_SETTINGS_TABLE` | Optional. Table storing UI preferences (`settings` by default). |

## Logging Behavior

Every successful `/predict` call records the following metadata:

- prediction label and status
- alert flag
- request source (`upload` or `camera`)
- uploaded filename and file size (bytes)
- model path reference
- optional `authority_number` (+63 format) supplied from the UI

History can be retrieved via `GET /history?limit=25&source=camera` with optional filtering by `source`. When Supabase credentials are not provided the endpoint returns `503` and logging is skipped.

### Shared Authority Contact

Create a simple settings table (default name `settings`) with columns similar to:

| Column | Type | Notes |
| --- | --- | --- |
| `key` | `text primary key` | e.g., `authority_number`. |
| `value` | `text` | Stores the `+63XXXXXXXXXX` number. |
| `updated_at` | `timestamptz` | Optional trigger defaulting to `now()`. |

Endpoints:

- `GET /settings/authority-number` returns `{ "authority_number": "+63..." }` if stored.
- `POST /settings/authority-number` accepts `{ "authority_number": "+63XXXXXXXXXX" }` and upserts the row.

Both endpoints require the Supabase client to be configured; otherwise a `503` is returned.
