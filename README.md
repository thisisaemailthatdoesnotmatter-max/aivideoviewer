# AIVideoViewer v2

Local AI video analysis tool. Transcribes and analyzes long videos
chunk-by-chunk using faster-whisper (transcription) and Qwen2.5-VL 7B
via Ollama (vision-language analysis), streaming results into a
localhost web UI. Includes per-video chat to ask questions about a
processed video.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on PATH
- [Ollama](https://ollama.com/) running locally, with the model pulled:
  ```
  ollama pull qwen2.5vl:7b
  ```

## Running

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`.

Or on Windows, just double-click `run.bat` in the project root — it
starts the server and opens the UI in Chrome.

## Notes

- Whisper is forced to CPU (`pipeline/whisper_service.py`) due to a
  `cublas64_12.dll` issue with faster-whisper's CUDA path on Windows.
  The VLM analysis step still uses GPU via Ollama regardless.
- Jobs run as background tasks decoupled from the browser tab —
  closing the tab mid-run does not stop processing. Reattach to a
  running or finished job using its job ID in the UI's "Reconnect"
  section, or `GET /jobs/{job_id}`.
- Per-chunk analysis intentionally does not receive prior-chunk memory
  context by default (see comments in `pipeline/orchestrator.py`) —
  testing showed the model would anchor on and repeat earlier chunks
  instead of describing new content when fed growing context.
