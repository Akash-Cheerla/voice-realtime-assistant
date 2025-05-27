# main.py - FastAPI backend with ElevenLabs TTS and Whisper STT

import os
import json
import base64
import asyncio
import tempfile
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from realtime_assistant import process_transcribed_text, form_data, conversation_history
from fill_pdf_logic import fill_pdf

import whisper
from elevenlabs import generate, set_api_key, Voice, VoiceSettings

# Initialize ElevenLabs
set_api_key(os.getenv("ELEVENLABS_API_KEY"))

# Load Whisper model
model = whisper.load_model("base")

# FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Data model
class AudioInput(BaseModel):
    audio_base64: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/voice-stream")
async def voice_stream(audio: AudioInput):
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio.audio_base64.split(",")[-1])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        # Transcribe with Whisper
        result = model.transcribe(temp_audio_path)
        user_text = result['text'].strip()

        print(f"\n🎙️ USER SAID: {user_text}")
        assistant_text = await process_transcribed_text(user_text)

        print(f"🤖 ASSISTANT REPLY: {assistant_text}")

        # Generate voice reply from ElevenLabs
        audio_reply = generate(
            text=assistant_text,
            voice=Voice(
                voice_id="EXAVITQu4vr4xnSDxMaL",  # Rachel
                settings=VoiceSettings(stability=0.5, similarity_boost=0.75)
            ),
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

@app.post("/confirm")
async def confirm_form(request: Request):
    try:
        body = await request.json()
        if body.get("confirmed"):
            with open("filled_form.json", "w", encoding="utf-8") as f:
                json.dump(form_data, f, indent=2)

            fill_pdf("form_template.pdf", "output_filled.pdf", form_data)
            return {"status": "filled", "download_url": "/download"}
        return {"status": "cancelled"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/download")
async def download_pdf():
    return FileResponse("output_filled.pdf", media_type="application/pdf", filename="Merchant_Form_Filled.pdf")

@app.get("/conversation")
async def get_conversation():
    return JSONResponse(conversation_history)
