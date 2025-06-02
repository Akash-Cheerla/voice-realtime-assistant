import os
import base64
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fill_pdf_logic import fill_pdf
from realtime_assistant import (
    process_transcribed_text,
    form_data,
    conversation_history,
    get_initial_assistant_message,
    end_triggered,
    reset_assistant_state
)
from ws_audio import router as ws_audio_router
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.include_router(ws_audio_router)

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounting
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("templates/index.html")

@app.get("/initial-message")
async def initial_message():
    assistant_text = get_initial_assistant_message()
    return JSONResponse({
        "assistant_text": assistant_text,
        "assistant_audio_base64": None  # Deprecated in WS mode
    })

@app.get("/form-data")
async def get_form_data():
    return JSONResponse(form_data)

@app.post("/confirm")
async def confirm(request: Request):
    try:
        body = await request.json()
        if body.get("confirmed"):
            fill_pdf("form_template.pdf", "filled_form.pdf", form_data)
            return JSONResponse({"status": "filled"})
        return JSONResponse({"status": "not confirmed"}, status_code=400)
    except Exception as e:
        print("❌ Error in /confirm:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/download")
async def download_pdf():
    return FileResponse("filled_form.pdf", media_type="application/pdf", filename="MerchantForm.pdf")

@app.post("/reset")
async def reset():
    reset_assistant_state()
    return JSONResponse({"status": "reset successful"})
