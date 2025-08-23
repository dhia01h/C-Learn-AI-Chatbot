import os
import io
import wave
import httpx
import numpy as np
from openai import AsyncOpenAI
import chainlit as cl

# ------------------ ENV VARIABLES ------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")


openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID or not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY, ELEVENLABS_API_KEY et ELEVENLABS_VOICE_ID doivent être définis"
    )

# Seuil de silence et timeout
SILENCE_THRESHOLD = 500 
SILENCE_TIMEOUT = 1300.0


# ------------------ SPEECH TO TEXT ------------------
@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await openai_client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",  
        file=audio_file
    )
    return response.text


# ------------------ TEXT TO SPEECH ------------------
@cl.step(type="tool")
async def text_to_speech(text: str, mime_type: str):
    CHUNK_SIZE = 1024
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "Accept": mime_type,
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
    }
    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()

        buffer = io.BytesIO()
        buffer.name = f"output_audio.{mime_type.split('/')[1]}"

        async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
            if chunk:
                buffer.write(chunk)

        buffer.seek(0)
        return buffer.name, buffer.read()


# ------------------ GENERATE TEXT ANSWER ------------------
@cl.step(type="tool")
async def generate_text_answer(transcription):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": transcription})

    response = await openai_client.chat.completions.create(
        model="llama3-8b-8192",  # ✅ modèle Groq LLM
        messages=message_history,
        temperature=0.2
    )

    message = response.choices[0].message
    message_history.append(message)

    return message.content


# ------------------ CHAT START ------------------
@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])
    await cl.Message(
        content="🎤 Bienvenue ! Appuie sur `p` pour parler 🚀",
    ).send()


# ------------------ AUDIO HANDLING ------------------
@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")

    if audio_chunks is not None:
        audio_array = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_array)

    # Premier chunk
    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Différence de temps entre chunks
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # ⚡ RMS avec NumPy
    audio_array = np.frombuffer(chunk.data, dtype=np.int16)
    audio_energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

    if audio_energy < SILENCE_THRESHOLD:
        # Silence détecté
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        # Pas de silence
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)


# ------------------ PROCESS AUDIO ------------------
async def process_audio():
    if audio_chunks := cl.user_session.get("audio_chunks"):
        concatenated = np.concatenate(list(audio_chunks))
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(concatenated.tobytes())

        wav_buffer.seek(0)
        cl.user_session.set("audio_chunks", [])

    frames = wav_file.getnframes()
    rate = wav_file.getframerate()
    duration = frames / float(rate)

    if duration <= 1.71:
        print("⏳ Audio trop court, réessaie !")
        return

    audio_buffer = wav_buffer.getvalue()
    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")

    whisper_input = ("audio.wav", audio_buffer, "audio/wav")
    transcription = await speech_to_text(whisper_input)

    await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[input_audio_el],
    ).send()

    answer = await generate_text_answer(transcription)

    output_name, output_audio = await text_to_speech(answer, "audio/wav")
    output_audio_el = cl.Audio(auto_play=True, mime="audio/wav", content=output_audio)

    await cl.Message(content=answer, elements=[output_audio_el]).send()


# ------------------ ON MESSAGE ------------------
@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content=" Ceci est une démo vocale, appuie sur `P` pour commencer !").send()
