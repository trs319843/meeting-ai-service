from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Request, Header
from faster_whisper import WhisperModel
import tempfile
import os
import time
from pydantic import BaseModel
import requests
import json

load_dotenv()
MEETINGBAAS_API_KEY = os.getenv("MEETINGBAAS_API_KEY")
BOT_RESULTS = {}

app = FastAPI()

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

CORRECTIONS = {
    "Telek": "Tilak",
    "The luck": "Tilak",
    "Olima": "Ollama",
    "Oracle Apex": "Oracle APEX",
    "Fast API": "FastAPI",
    "meat pilot": "MeetPilot",
    "Meat pilot": "MeetPilot",
    "meat pilot bought": "MeetPilot Bot",
    "Meat pilot bought": "MeetPilot Bot"
}

def apply_corrections(text: str) -> str:
    for wrong, correct in CORRECTIONS.items():
        text = text.replace(wrong, correct)
    return text

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

        for segment in transcript_segments:
            segment["text"] = apply_corrections(segment["text"])

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

        for segment in transcript_segments:
            segment["text"] = apply_corrections(segment["text"])

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

MEETINGBAAS_API_KEY = os.getenv("MEETINGBAAS_API_KEY")

class ScheduleBotRequest(BaseModel):
    meeting_id: int | None = None
    meeting_url: str
    bot_name: str = "MeetPilot Bot"
    webhook_url: str | None = None

@app.post("/schedule-bot")
def schedule_bot(request: ScheduleBotRequest):
    if not MEETINGBAAS_API_KEY:
        return {"success": False, "error": "MEETINGBAAS_API_KEY is not set"}

    payload = {
        "meeting_url": request.meeting_url,
        "bot_name": request.bot_name,
        "speech_to_text": "Gladia",
        "callback_enabled": True,
        "callback_config": {
            "url": request.webhook_url,
            "method": "POST",
            "secret": "meetpilot-dev-secret"
        }
    }

    response = requests.post(
        "https://api.meetingbaas.com/v2/bots",
        headers={
            "x-meeting-baas-api-key": MEETINGBAAS_API_KEY,
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=60
    )

    return {
        "success": response.status_code in (200, 201, 202),
        "meeting_id": request.meeting_id,
        "status_code": response.status_code,
        "response": response.json() if response.text else None
    }


@app.post("/meetingbaas/webhook")
async def meetingbaas_webhook(request: Request):
    payload = await request.json()

    event = payload.get("event")
    data = payload.get("data", {})

    if event != "bot.completed":
        return {
            "success": True,
            "message": "Ignored event",
            "event": event
        }

    bot_id = data.get("bot_id")
    audio_url = data.get("audio")
    duration_seconds = data.get("duration_seconds")

    if not bot_id:
        return {"success": False, "error": "No bot_id received"}

    if not audio_url:
        return {"success": False, "error": "No audio URL received"}

    audio_response = requests.get(audio_url, timeout=300)
    audio_response.raise_for_status()

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as tmp:
            tmp.write(audio_response.content)
            temp_path = tmp.name

        segments, info = model.transcribe(temp_path)

        transcript_segments = []

        for segment in segments:
            text = segment.text.strip()
            text = apply_corrections(text)

            transcript_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text
            })

        full_transcript = " ".join(
            segment["text"] for segment in transcript_segments
        )

        BOT_RESULTS[bot_id] = {
            "bot_id": bot_id,
            "duration_seconds": duration_seconds,
            "language": info.language,
            "transcript": full_transcript,
            "segments": transcript_segments,
            "raw_payload": payload
        }

        print("Bot completed:", bot_id)
        print("Transcript:", full_transcript)

        return {
            "success": True,
            "bot_id": bot_id,
            "message": "Transcript processed"
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/bot-result/{bot_id}")
def get_bot_result(bot_id: str):
    result = BOT_RESULTS.get(bot_id)

    if not result:
        return {
            "success": False,
            "error": "Bot result not found",
            "bot_id": bot_id
        }

    return {
        "success": True,
        "result": result
    }