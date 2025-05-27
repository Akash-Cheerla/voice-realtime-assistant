# MAIN.PY - FastAPI backend

import os
import json
import base64
import asyncio
import tempfile
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from realtime_assistant import process_transcribed_text, form_data, conversation_history
from elevenlabs import generate, play, set_api_key
import whisper

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Init TTS key
set_api_key(os.getenv("ELEVENLABS_API_KEY"))
model = whisper.load_model("base")

class AudioInput(BaseModel):
    audio_base64: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/voice-stream")
async def voice_stream(audio: AudioInput):
    try:
        audio_bytes = base64.b64decode(audio.audio_base64.split(",")[-1])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        result = model.transcribe(temp_audio_path)
        user_text = result['text'].strip()

        print(f"\n🎙️ USER SAID: {user_text}")
        assistant_text = await process_transcribed_text(user_text)

        print(f"🤖 ASSISTANT REPLY: {assistant_text}")

        audio_reply = generate(
            text=assistant_text,
            voice="Rachel",
            model="eleven_monolingual_v1"
        )
        audio_base64 = base64.b64encode(audio_reply).decode("utf-8")

        return JSONResponse({
            "user_text": user_text,
            "assistant_text": assistant_text,
            "assistant_audio_base64": audio_base64
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/form-data")
async def get_form():
    return JSONResponse(form_data)

@app.get("/conversation")
async def get_history():
    return JSONResponse(conversation_history)
