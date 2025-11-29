# River Hyacinth Frontend

React + Vite dashboard that lets operators upload images or run a live camera feed, visualize model output, and manage shared authority contact information stored in Supabase.

## Environment Variables

Create a `.env` (or `.env.local`) inside `frontend/` with the following values:

```
VITE_API_URL=https://your-backend.example.com
VITE_EMAILJS_SERVICE_ID=your_service_id
VITE_EMAILJS_TEMPLATE_ID=your_template_id
VITE_EMAILJS_PUBLIC_KEY=your_public_key
```

- `VITE_API_URL` should point to the FastAPI backend (Railway URL in prod or `http://localhost:8000` locally).
- The EmailJS variables are optional but required if you want automatic email alerts when the model detects `Moderate Growth` or `Large Growth`.

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
