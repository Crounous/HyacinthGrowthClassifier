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

## Logging Behavior

Every successful `/predict` call records the following metadata:

- prediction label and status
- alert flag
- request source (`upload` or `camera`)
- uploaded filename and file size (bytes)
- model path reference

History can be retrieved via `GET /history?limit=25&source=camera` with optional filtering by `source`. When Supabase credentials are not provided the endpoint returns `503` and logging is skipped.
