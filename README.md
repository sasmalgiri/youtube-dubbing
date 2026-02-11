# YouTube Dubbing Application

This project is a video dubbing application that processes YouTube videos and generates dubbed audio tracks with subtitles. The application utilizes various components for speech-to-text, translation, text-to-speech, and subtitle generation.

## Features

- Download YouTube videos and captions
- Extract audio from videos
- Transcribe audio to text
- Translate transcriptions to different languages
- Synthesize dubbed audio from translated text
- Generate subtitles in SRT and WebVTT formats

## Project Structure

```
youtube-dubbing-app
├── src
│   ├── app.py                # Main entry point of the application
│   ├── cli.py                # Command-line interface for user interaction
│   ├── dubbing               # Dubbing related functionalities
│   ├── io                    # Input/Output operations
│   ├── stt                   # Speech-to-text functionalities
│   ├── mt                    # Machine translation functionalities
│   ├── tts                   # Text-to-speech functionalities
│   ├── subtitles             # Subtitle generation functionalities
│   ├── models                # Data models and types
│   └── utils                 # Utility functions
├── tests                     # Unit tests for the application
├── scripts                   # Scripts for downloading and processing
├── .vscode                   # VS Code configuration files
├── .gitignore                # Git ignore file
├── .env.example              # Example environment variables
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project metadata
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd youtube-dubbing-app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, use the command line interface defined in `src/cli.py`. You can download a YouTube video, transcribe it, translate the transcription, and synthesize the dubbed audio.

## Web UI (Vercel) + Backend (Free)

Vercel can host the **web UI**, but the actual dubbing work (FFmpeg + Whisper + TTS) must run on a separate backend.

### Frontend (Vercel)

The Next.js UI lives in `web/`.

1) Install deps:

```bash
cd web
npm install
```

2) Set environment variable in Vercel:

- `NEXT_PUBLIC_API_BASE_URL` = your backend base URL (example: `https://your-backend.onrender.com`)

3) Deploy the `web/` folder to Vercel.

### Backend (Free tier suggestion: Render)

The FastAPI backend lives in `backend/` and exposes:

- `GET /voices`
- `POST /jobs` (multipart upload)
- `GET /jobs/{id}`
- `GET /jobs/{id}/result`

Local run (after installing backend deps into your Python env):

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then point the frontend to `http://localhost:8000`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.