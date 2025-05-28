import os
import base64
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from fill_pdf_logic import fill_pdf
from realtime_assistant import (
    process_transcribed_text,
    form_data,
    conversation_history,
    get_initial_assistant_message
)
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI()
eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and templates setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def serve_index():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.get("/initial-message")
async def initial_message():
    assistant_text = get_initial_assistant_message()
    audio_reply = eleven_client.text_to_speech.convert(
        voice_id="EXAVITQu4vr4xnSDxMaL",
        model_id="eleven_monolingual_v1",
        text=assistant_text
    )
    audio_bytes = b"".join(audio_reply)
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return JSONResponse({
        "assistant_text": assistant_text,
        "assistant_audio_base64": audio_base64
    })


@app.post("/voice-stream")
async def voice_stream(audio: UploadFile = File(...)):
    try:
        contents = await audio.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(contents)
            temp_audio_path = temp_audio.name

        with open(temp_audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )

        user_text = result.text.strip()
        print(f"🎙️ USER SAID: {user_text}")

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
            "audio_base64": audio_base64,
            "form_data": form_data
        })

    except Exception as e:
        print("❌ Error in /voice-stream:", e)
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/form-data")
async def get_form_data():
    return JSONResponse(form_data)


@app.post("/confirm")
async def confirm(request: Request):
    try:
        body = await request.json()
        if body.get("confirmed"):
            filled_pdf_path = fill_pdf(form_data)
            return JSONResponse({"status": "filled"})
        return JSONResponse({"status": "not confirmed"}, status_code=400)
    except Exception as e:
        print("❌ Error in /confirm:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/download")
async def download_pdf():
    return FileResponse("filled_form.pdf", media_type="application/pdf", filename="MerchantForm.pdf")
