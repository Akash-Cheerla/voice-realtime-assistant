import asyncio
import json
import base64
import os
import tempfile
import wave
import audioop
from fastapi import APIRouter, WebSocket
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import openai
import torch
import whisper
from whisper import Whisper
from realtime_assistant import process_transcribed_text, get_initial_assistant_message

load_dotenv()
router = APIRouter()
model: Whisper = whisper.load_model("base")  # Use "tiny" for faster performance

tts = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

async def handle_assistant_logic(user_text):
    assistant_text = await process_transcribed_text(user_text)
    print("🧫 Assistant:", assistant_text)

    audio_reply = tts.text_to_speech.convert(
        voice_id="EXAVITQu4vr4xnSDxMaL",
        model_id="eleven_monolingual_v1",
        text=assistant_text
    )
    audio_bytes = b"".join(audio_reply)
    return assistant_text, audio_bytes

@router.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = b""

    # ⏱ Initial Assistant Greeting
    try:
        initial_text = get_initial_assistant_message()
        audio_reply = tts.text_to_speech.convert(
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model_id="eleven_monolingual_v1",
            text=initial_text
        )
        audio_bytes = b"".join(audio_reply)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        await websocket.send_text(json.dumps({
            "type": "assistant_reply",
            "text": initial_text,
            "audio_b64": audio_b64
        }))

        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            if data["type"] == "audio_chunk":
                audio_buffer += base64.b64decode(data["data"])

            elif data["type"] == "end_stream":
                # ✅ Resample from 48000 to 16000 using audioop
                resampled = audioop.ratecv(audio_buffer, 2, 1, 48000, 16000, None)[0]

                print("🔊 Received audio bytes:", len(audio_buffer))
                duration_sec = len(audio_buffer) / (2 * 48000)
                print(f"⏱️ Approx. duration: {duration_sec:.2f}s")

                if len(audio_buffer) < 6400:
                    print("⚠️ Skipping: audio too short.")
                    audio_buffer = b""
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                    with wave.open(tmp_audio, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(resampled)
                    tmp_path = tmp_audio.name

                result = model.transcribe(tmp_path)
                transcript = result.get("text", "").strip()
                os.remove(tmp_path)

                print("🎤 User said:", transcript)
                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "text": transcript
                }))

                if not transcript:
                    audio_buffer = b""
                    continue

                assistant_text, audio_bytes = await handle_assistant_logic(transcript)
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

                await websocket.send_text(json.dumps({
                    "type": "assistant_reply",
                    "text": assistant_text,
                    "audio_b64": audio_b64
                }))

                audio_buffer = b""

                if "END OF CONVERSATION" in assistant_text.upper():
                    print("✅ Ending session...")
                    break

    except Exception as e:
        print("❌ WebSocket error:", e)
        await websocket.send_text(json.dumps({ "type": "error", "message": str(e) }))
