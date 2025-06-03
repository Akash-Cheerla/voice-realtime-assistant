# ws_audio.py
import asyncio
import json
import base64
import os
import tempfile
import wave
import audioop
import time
from fastapi import APIRouter, WebSocket
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import openai
import torch
import whisper
from whisper import Whisper
from realtime_assistant import process_transcribed_text, get_initial_assistant_message
from vad import is_speech

load_dotenv()
router = APIRouter()
model: Whisper = whisper.load_model("small")  # Upgraded model for better accuracy

tts = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

currently_playing_audio = None
last_assistant_tts_time = 0
interrupted = False

def generate_tts(assistant_text):
    audio_reply = tts.text_to_speech.convert(
        voice_id="EXAVITQu4vr4xnSDxMaL",
        model_id="eleven_monolingual_v1",
        text=assistant_text
    )
    return b"".join(audio_reply)

@router.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = b""
    global currently_playing_audio, last_assistant_tts_time, interrupted

    try:
        initial_text = get_initial_assistant_message()
        audio_bytes = await asyncio.to_thread(generate_tts, initial_text)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        last_assistant_tts_time = time.time()

        await websocket.send_text(json.dumps({
            "type": "assistant_reply",
            "text": initial_text,
            "audio_b64": audio_b64
        }))

        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            if data["type"] == "audio_chunk":
                interrupted = True
                if currently_playing_audio:
                    currently_playing_audio.cancel()
                    currently_playing_audio = None
                    await websocket.send_text(json.dumps({"type": "interrupt_audio"}))
                    print("⛔️ Assistant TTS interrupted by user.")
                audio_buffer += base64.b64decode(data["data"])

            elif data["type"] == "end_stream":
                await asyncio.sleep(0.2)
                resampled = audioop.ratecv(audio_buffer, 2, 1, 48000, 16000, None)[0]

                duration_seconds = len(resampled) / 32000
                if duration_seconds > 8:
                    print("🛑 Audio too long (>8s), skipping.")
                    audio_buffer = b""
                    continue

                if time.time() - last_assistant_tts_time < 1.0:
                    print("🛑 Skipping input: too soon after TTS.")
                    audio_buffer = b""
                    continue

                if not is_speech(resampled):
                    print("🛑 VAD: No speech detected.")
                    audio_buffer = b""
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                    with wave.open(tmp_audio, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(resampled)
                    tmp_path = tmp_audio.name

                result = await asyncio.to_thread(model.transcribe, tmp_path, language='en', task='transcribe')
                os.remove(tmp_path)
                transcript = result.get("text", "").strip()

                if not transcript or len(transcript.split()) < 2 or transcript.lower().count("sí") > 8:
                    print("🛑 Ignoring hallucinated transcript.")
                    audio_buffer = b""
                    continue

                print("🎤 User said:", transcript)
                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "text": transcript
                }))

                try:
                    assistant_text = await asyncio.wait_for(process_transcribed_text(transcript), timeout=10.0)
                    if not assistant_text:
                        raise ValueError("No assistant response.")
                except Exception as e:
                    print("⚠️ Assistant failed:", e)
                    await websocket.send_text(json.dumps({
                        "type": "assistant_reply",
                        "text": "Sorry, I had trouble processing that. Could you please repeat?"
                    }))
                    audio_buffer = b""
                    continue

                if interrupted:
                    print("🔄 Skipping outdated assistant reply.")
                    interrupted = False
                    audio_buffer = b""
                    continue

                if not assistant_text.strip():
    print("⚠️ Empty assistant message. Skipping TTS.")
    audio_buffer = b""
    continue

print("🤖 Assistant:", assistant_text)
tts_task = asyncio.to_thread(generate_tts, assistant_text)
audio_bytes = await tts_task
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                last_assistant_tts_time = time.time()

                async def send_reply():
                    await websocket.send_text(json.dumps({
                        "type": "assistant_reply",
                        "text": assistant_text,
                        "audio_b64": audio_b64
                    }))

                currently_playing_audio = asyncio.create_task(send_reply())
                await currently_playing_audio
                currently_playing_audio = None
                audio_buffer = b""

                if "END OF CONVERSATION" in assistant_text.upper():
                    print("✅ Ending session...")
                    break

    except Exception as e:
        print("❌ WebSocket error:", e)
        try:
            await websocket.send_text(json.dumps({ "type": "error", "message": str(e) }))
        except RuntimeError:
            print("⚠️ Skipped sending error: client already disconnected.")
