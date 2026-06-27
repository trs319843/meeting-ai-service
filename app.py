from fastapi import FastAPI, UploadFile, File, Request, Header
from faster_whisper import WhisperModel
import tempfile
import os
import time
from pydantic import BaseModel
import requests
import json

app = FastAPI()

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {
        "model": "base",
        "engine": "faster-whisper"
    }

@app.get("/version")
def version():
    return {
        "application": "Meeting AI Service",
        "version": "1.0.0",
        "model": "base",
        "engine": "faster-whisper"
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    temp_path = None

    try:
        start_time = time.time()

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        segments, info = model.transcribe(temp_path)

        transcript_segments = []

        for segment in segments:
            transcript_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })

        CORRECTIONS = {
            "Telek": "Tilak",
            "The luck": "Tilak",
            "Olima": "Ollama",
            "Oracle Apex": "Oracle APEX",
            "Fast API": "FastAPI"
        }

        for segment in transcript_segments:
            for wrong, correct in CORRECTIONS.items():
                segment["text"] = segment["text"].replace(wrong, correct)

        full_transcript = " ".join(
            segment["text"]
            for segment in transcript_segments
        )

        # transcription
        processing_time = round(time.time() - start_time, 2)

        return {
            "filename": file.filename,
            "language": info.language,
            "duration": round(info.duration, 2),
            "transcript": full_transcript,
            "segments": transcript_segments,
            "processing_time": processing_time
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/transcribe-blob")
async def transcribe_blob(
    request: Request,
    x_filename: str = Header(default="recording.m4a")
):
    start_time = time.time()

    file_bytes = await request.body()
    file_ext = os.path.splitext(x_filename)[1] or ".m4a"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        segments, info = model.transcribe(temp_path)

        transcript_segments = []

        for segment in segments:
            transcript_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })

        CORRECTIONS = {
            "Telek": "Tilak",
            "The luck": "Tilak",
            "Olima": "Ollama",
            "Oracle Apex": "Oracle APEX",
            "Fast API": "FastAPI"
        }

        for segment in transcript_segments:
            for wrong, correct in CORRECTIONS.items():
                segment["text"] = segment["text"].replace(wrong, correct)

        full_transcript = " ".join(
            segment["text"] for segment in transcript_segments
        )

        return {
            "filename": x_filename,
            "language": info.language,
            "duration": round(info.duration, 2),
            "processing_time": round(time.time() - start_time, 2),
            "transcript": full_transcript,
            "segments": transcript_segments
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


class SummaryRequest(BaseModel):
    transcript: str

@app.post("/summarize")
def summarize_meeting(request: SummaryRequest):
    prompt = f"""
You are an AI meeting assistant.

Return only valid JSON in this exact format:
{{
  "summary": "...",
  "action_items": [
    {{
      "task": "...",
      "owner": null,
      "due_date": null
    }}
  ],
  "decisions": ["..."]
}}

Rules:
- Action items are future tasks that still need to be done.
- Each action item must be an object with exactly these keys: task, owner, due_date.
- Do not return action_items as plain strings.
- Do not include owner or due date inside the task text.
- If someone agreed to do something, include it in action_items.
- Decisions are agreements, approvals, or conclusions reached during the meeting.
- If no action items exist, return an empty array.
- If no decisions exist, return an empty array.
- Do not invent names or tasks.
- Completed work should be mentioned only in the summary.
- Do not classify completed activities as action items or decisions.
- If owner or due_date are not mentioned, return null.

Action item owner rules:
- If a sentence says "<Person> will do something", set owner to that person.
- If a sentence says "<Person> to do something", set owner to that person.
- If a sentence says "<Person> is responsible for something", set owner to that person.
- Do not leave owner null when a person's name appears directly before "will", "to", "shall", "needs to", or "is responsible for".
- Owner must come from the transcript only.
- Do not copy names from examples.

Example:
Input sentence: "Priya will prepare user acceptance testing scenarios by next Tuesday."
Output action item:
{{
  "task": "Prepare user acceptance testing scenarios",
  "owner": "Priya",
  "due_date": "next Tuesday"
}}

Transcript:
{request.transcript}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=300
    )

    response.raise_for_status()

    result = response.json()
    return json.loads(result["response"])