# AltScore React frontend

Vite/React frontend for the repository's real `POST /v1/predict` API. It has no mock prediction path: if the API is unreachable, it displays an error instead of fabricating a result.

## Local development

1. In the repository root, start the backend: `uvicorn api:app --host 127.0.0.1 --port 8000`
2. In this folder: `npm install` then `npm run dev`

## Vercel

Set `VITE_API_BASE_URL` to the HTTPS URL of the separately deployed FastAPI service. On the API host, set `CORS_ALLOWED_ORIGINS` to your Vercel domain, for example `https://your-project.vercel.app`. Build command: `npm run build`; output directory: `dist`.

The backend, model, and React app are all a simulated research demonstration. Never send real applicant information or use outputs for lending decisions.
