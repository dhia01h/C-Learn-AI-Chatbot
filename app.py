# 1️⃣ Importing Required Libraries
import os
from dotenv import load_dotenv
import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import Select, Switch
from rag_pipeline import load_vectorstore
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langdetect import detect 
import aiohttp
import re
from utils import extraire_texte_depuis_image
from langchain.chains import RetrievalQA
from stackoverflow_search import extract_c_error, search_stackoverflow, is_c_error 


import os
import io
import wave
import httpx
import numpy as np

from openai import AsyncOpenAI
import chainlit as cl


# 2️⃣ Load Environment Variables
load_dotenv()


os.environ['CHAINLIT_AUTH_SECRET'] = "6n?yguZPN8xuWC2aZX0Can>%j*X,j8@%5aODVez,Z.R?3BFzePKiwCF6:5hsRe9I"

# 3️⃣ Groq models available
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
     "llama3-8b-8192",
    "whisper-large-v3-turbo",
    "deepseek-r1-distill-llama-70b",
    "compound-beta"
    
]

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# 4️⃣ Authentication
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    return cl.User(identifier=username)

# 5️⃣ Database Layer
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=os.getenv("DATABASE_URL"))


"""commands = [
        {"id": "Audio", "icon": "audio-lines", "button": True , "persistent": True}
    ]
    await cl.context.emitter.set_commands(commands)"""

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")

# 🎤 Client Groq (OpenAI compatible)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=OPENAI_API_BASE)


# Seuil de silence et timeout
SILENCE_THRESHOLD = 500  # Ajuste selon ton micro
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


# 6️⃣ Chat Start
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("message_history", [])
    cl.user_session.set('chat_history', [])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    cl.user_session.set("memory", memory)

    settings = await cl.ChatSettings(
        [
            Select(id="Model", label="Groq Model", values=GROQ_MODELS, initial_index=2),
            Switch(id="Verbose", label="Detailed Response", initial=True)
        ]
    ).send()
    cl.user_session.set("settings", settings)

    await cl.ElementSidebar.set_elements([

        cl.Image(
            url="https://www.esprit.tn/wp-content/uploads/2025/02/Logo_ESPRIT.svg",
            name="esprit_logo",
            display="inline",
            size="medium"
        ),
        
        cl.Text(
            content="""### Les Fonctionnalités 
- Choix du modèle (GPT, Llama, etc.)            
- Correction de code avec explications
- Diagnostic d'erreurs précises
- Accès direct à Stack Overflow
- Quiz personnalisés sur le C
- Support multilingue (FR/EN)
- Export des sessions en PDF
- Supporte les images""",
            name="features_menu"
        )
    ])
    await cl.ElementSidebar.set_title("Fermer la page")

    # Message de bienvenue
    app_user = cl.user_session.get("user")
    welcomeUserStr = app_user.identifier.split('@')[0].replace('.', ' ')
    await cl.Message(f"> Bonjour {welcomeUserStr}. Que veux-tu apprendre aujourd’hui ? ").send()


"""    features = [
    {
        "title": "Génération et correction de code C",
        "content": "Obtenez de l'aide pour vos programmes en C avec corrections détaillées"
    },
    {
        "title": "OCR intégré", 
        "content": "Extraction et analyse de texte depuis des images ou documents"
    },
    {
        "title": "Quiz interactifs",
        "content": "Testez vos connaissances avec des questions personnalisées"
    }
    ]

    cards = [
    cl.Text(
        name=feat["title"],
        content=feat["content"],  # Contenu obligatoire ajouté
        display="inline"
    ) for feat in features
    ]

    await cl.Message(
    content="",
    elements=cards
    ).send()"""


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



# 7️⃣ Resume Chat
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("chat_history", [])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    cl.user_session.set("memory", memory)

    settings = await cl.ChatSettings(
        [
            Select(id="Model", label="Groq Model", values=GROQ_MODELS, initial_index=2),
            Switch(id="Verbose", label="Detailed Response", initial=True)
        ]
    ).send()
    cl.user_session.set("settings", settings)

    for step in thread["steps"]:
        if step["type"] == "user_message":
            memory.chat_memory.add_user_message(step["output"])
        elif step["type"] == "assistant_message":
            memory.chat_memory.add_ai_message(step["output"])

# 8️⃣ Message Handler
@cl.on_message
async def on_message(message: cl.Message):
    settings = cl.user_session.get("settings")
    memory = cl.user_session.get("memory")
    model = settings["Model"]
    verbose = settings["Verbose"]

    llm = ChatOpenAI(
        model_name=model,
        temperature=0.3 if verbose else 0,
        streaming=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )

    from langchain.chains import ConversationalRetrievalChain


    vectordb = load_vectorstore()
    #traitement des images avec OCR
    if message.elements:
        for element in message.elements:
            if element.type == "image":
                # OCR
                image_path = element.path
                texte = extraire_texte_depuis_image(image_path)

                await cl.Message(content=f"```text\n{texte}\n```").send()

                # Continuer le traitement avec le LLM
                model_name = cl.user_session.get("model", "llama3-8b-8192")
                llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=0.3,
                    streaming=True,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_api_base=os.getenv("OPENAI_API_BASE")
                )

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
                    return_source_documents=False
                )

                result = await qa.arun(texte)
                await cl.Message(content=f"{result}").send()
                return

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        memory=memory
    )

    # 🆕 Détection de langue
    try:
        lang = detect(message.content.strip())
    except:
        lang = "fr"  # défaut

    if lang == "fr":
        prefix = "Réponds en français, même s'il y a des fautes de frappe dans la question :\n"
    else:
        prefix = "Answer in English, even if there are typos in the question:\n"

# ✅ RAG + langue
    result = await qa_chain.acall(prefix + message.content.strip())

    await cl.Message(content=result["answer"]).send()
    cl.user_session.get("chat_history").append({
        "role": "user",
        "content": message.content.strip()
    })
    if is_c_error(message.content.strip()):
        link = await search_stackoverflow(message.content.strip())
        if link:
            await cl.Message(content=f" J'ai trouvé une discussion sur Stack Overflow qui pourrait t'aider :\n{link}").send()
            return

#  Gestion du quiz
    if "quiz" in message.content.lower():
        source_docs = result.get("source_documents", [])
        if not source_docs:
            source_docs_text = "Aucun document disponible"
        else:
            source_docs_text = "\n".join([str(doc.page_content) for doc in source_docs])

        quiz_prompt = f"""
À partir des informations suivantes, créez une question Vrai/Faux en français.
Répondez avec ce format exact :
answer: True/False
explanation: ...
Docs: {source_docs_text}
"""
        quiz_result = await llm.agenerate([[HumanMessage(content=quiz_prompt)]])
        quiz_text = quiz_result.generations[0][0].text.strip()

        lines = quiz_text.split("\n")
        question_line = next((l for l in lines if l.lower().startswith("question:")), "")
        answer_line = next((l for l in lines if l.lower().startswith("answer:")), "")
        explanation_line = next((l for l in lines if l.lower().startswith("explanation:")), "")
        correct_answer = "Vrai" if "true" in answer_line.lower() else "Faux"

        msg = await cl.Message(content=question_line).send()
        buttons = [
            cl.Action(label="Vrai", value="Vrai"),
            cl.Action(label="Faux", value="Faux")
        ]
        user_response = await msg.send_buttons(buttons)

        if user_response.value == correct_answer:
            color = "green"
            feedback = " Correct !"
        else:
            color = "red"
            feedback = " Faux !"

        await cl.Message(content=f"{feedback}\n{explanation_line}", color=color).send()
