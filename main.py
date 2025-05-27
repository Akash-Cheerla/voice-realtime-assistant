# main.py — final version with ElevenLabs, Whisper API, and initial assistant response

import os
import json
import base64
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import openai
from elevenlabs.client import ElevenLabs

from realtime_assistant import process_transcribed_text, form_data, conversation_history
from fill_pdf_logic import fill_pdf

# Initialize OpenAI and ElevenLabs
openai.api_key = os.getenv("OPENAI_API_KEY")
eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class AudioInput(BaseModel):
    audio_base64: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/initial-message")
async def initial_message():
    try:
        initial_text = await process_transcribed_text("[start]")
        audio_reply = eleven_client.text_to_speech.convert(
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model_id="eleven_monolingual_v1",
            text=initial_text
        )
        audio_bytes = b"".join(audio_reply)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return JSONResponse({
            "assistant_text": initial_text,
            "assistant_audio_base64": audio_base64
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/voice-stream")
async def voice_stream(audio: AudioInput):
    try:
        audio_bytes = base64.b64decode(audio.audio_base64.split(",")[-1])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        with open(temp_audio_path, "rb") as audio_file:
            result = openai.Audio.transcribe("whisper-1", audio_file)
        user_text = result["text"].strip()

        print(f"\n🎙️ USER SAID: {user_text}")
        assistant_text = await process_transcribed_text(user_text)
        print(f"🤖 ASSISTANT REPLY: {assistant_text}")

        audio_reply = eleven_client.text_to_speech.convert(
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model_id="eleven_monolingual_v1",
            text=assistant_text
        )
        audio_bytes = b"".join(audio_reply)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return JSONResponse({
            "user_text": user_text,
            "assistant_text": assistant_text,
            "assistant_audio_base64": audio_base64
        })

    except Exception as e:
        print("❌ Exception in /voice-stream:", e)
        traceback.print_exc()
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
