# AI - Real-Time Video Intelligence System

An AI-powered real-time video intelligence project that combines a Python backend for video analysis with a React-based frontend for visualization.

## Features

- Real-time video stream processing
- AI-based object/event detection workflow
- Alerts and user data persistence in JSON stores
- Frontend dashboard for monitoring and analytics

## Project Structure

- `real-time-video-ai/backend` - Python backend services and templates
- `real-time-video-ai/requirements.txt` - Backend Python dependencies
- `frontend` - React frontend application
- `package.json` - JavaScript dependencies used in the workspace

## Prerequisites

- Python 3.10+ (recommended)
- Node.js 18+ and npm

## Backend Setup

```bash
cd real-time-video-ai
pip install -r requirements.txt
python backend/app.py
```

If your backend entry point is `main.py` instead, use:

```bash
python backend/main.py
```

## Frontend Setup

```bash
cd frontend
npm install
npm start
```

The frontend usually runs at `http://localhost:3000`.

## Notes

- Large model files (for example `yolov8n.pt`) are currently stored in the repo.
- For long-term maintenance, consider adding a `.gitignore` and excluding generated files like `node_modules` and `__pycache__`.

## Repository

GitHub: [AI---Real-Time-Video-Intelligence-System](https://github.com/sumitdwivedi0640/AI---Real-Time-Video-Intelligence-System)
