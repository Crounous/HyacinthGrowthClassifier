# River Hyacinth Frontend

React + Vite dashboard that lets operators upload images or run a live camera feed, visualize model output, and manage shared authority contact information stored in Supabase.

## Environment Variables

Create a `.env` (or `.env.local`) inside `frontend/` with the following values:

```
VITE_API_URL=
VITE_EMAILJS_SERVICE_ID=your_service_id
VITE_EMAILJS_TEMPLATE_ID=your_template_id
VITE_EMAILJS_PUBLIC_KEY=your_public_key
```

- `VITE_API_URL` can be left blank when deploying on Vercel because `/predict`, `/history`, and `/settings/*` are served via Vercel functions under `frontend/api/`.
   - For local dev, you can also leave it blank and run `pnpm dev`; Vite will call the same-origin routes.
- The EmailJS variables are optional but required if you want automatic email alerts when the model detects `Moderate Growth` or `Large Growth`.

### Vercel Server Function Env Vars

Set these in Vercel (Project Settings → Environment Variables) for the serverless API routes:

```
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
SUPABASE_TABLE=prediction_logs

MODEL_URL=
LABELS_URL=
MODEL_PATH=
```

- `MODEL_URL` should be a direct-download URL for `best_model.onnx` (for example, a Supabase Storage public URL).
- `LABELS_URL` is optional; if omitted, the API will try to derive it by replacing `.onnx` with `.labels.json`.
- `MODEL_PATH` is optional; default is `/tmp/best_model.onnx` on Vercel.

### Model Inference Notes (Vercel)

The `/api/predict` route runs inference inside a Vercel Serverless Function. To stay under Vercel's serverless bundle size limits, the API uses `onnxruntime-web` (WASM) and decodes images as JPEG/PNG.

## Uploading Large Models to Supabase Storage

The Supabase dashboard upload UI may reject large files (for example >50MB). You can upload your ONNX model from your machine instead.

1. Create a Storage bucket (e.g. `models`). For easiest setup with the Vercel API downloader, make the bucket **public**.
2. From this repo:

```bash
cd frontend
npm install
```

3. Set env vars (PowerShell example) and run the uploader:

```powershell
$env:SUPABASE_URL='https://<project-ref>.supabase.co'
$env:SUPABASE_SERVICE_KEY='<service-role-key>'
$env:SUPABASE_BUCKET='models'
node scripts/upload_model_to_supabase.mjs
```

It prints `MODEL_URL` and `LABELS_URL` values you can paste into Vercel.

## Email Alerts via EmailJS

1. Sign up at [emailjs.com](https://www.emailjs.com/) and add an email service (Gmail, Outlook, custom SMTP, etc.).
2. Create a template containing the placeholders below. The UI passes these fields when a warning alert occurs:
   - `to_email`
   - `prediction`
   - `status`
   - `source`
   - `filename`
   - `timestamp`
3. Grab the Service ID, Template ID, and Public Key from EmailJS and place them in the env vars above.
4. In the app, save an authority email address. Every warning result will trigger `emailjs.send` directly from the browser.

## Development

```bash
cd frontend
pnpm install   # or npm install / yarn
pnpm dev
```

The dev server runs on `http://localhost:5173`. Ensure the backend is reachable via `VITE_API_URL` so API calls succeed.

## Production Build

```bash
pnpm build
```

The compiled assets land in `frontend/dist/` and can be deployed to Vercel, Netlify, or any static host.
