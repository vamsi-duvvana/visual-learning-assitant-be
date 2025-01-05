# old code
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import openai
from typing import Optional
import base64
import tempfile
import os
from datetime import datetime
from pydantic import BaseModel
import asyncio
import shutil
from pathlib import Path
from dotenv import load_dotenv
import time
import psutil

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Visual Learning Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create temp directory for files
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Models for request/response


class DoubtSession(BaseModel):
    timestamp: str
    image_analysis: str
    context: str
    solution: str


# In-memory session storage
sessions: dict[str, list[DoubtSession]] = {}


async def save_session(user_id: str, doubt_data: dict):
    """Save doubt session to memory"""
    session = DoubtSession(
        timestamp=datetime.now().isoformat(),
        **doubt_data
    )

    if user_id not in sessions:
        sessions[user_id] = []

    sessions[user_id].append(session)


async def get_text_from_audio(audio_path: str) -> Optional[str]:
    """Convert audio to text using Whisper"""
    try:
        with open(audio_path, "rb") as audio:
            transcript = await asyncio.to_thread(
                openai.audio.transcriptions.create,
                model="whisper-1",
                file=audio,
                response_format="text"
            )
        return transcript
    except Exception as e:
        print(f"Error in audio transcription: {e}")
        return None


async def analyze_image(image_data: bytes) -> Optional[str]:
    """Analyze image using OpenAI Vision"""
    try:
        base64_image = base64.b64encode(image_data).decode('utf-8')
        response = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this image and describe what academic doubt or question it contains."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in image analysis: {e}")
        return None


async def generate_solution(image_analysis: str, context: str) -> Optional[str]:
    """Generate solution using GPT-4"""
    try:
        response = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert educational AI assistant specialized in breaking down complex topics into simple steps.

                        Your responses must:
                        - Start with a brief overview of the topic
                        - Provide 3-4 clear, actionable steps
                        - Include relevant examples or analogies
                        - Use concise, clear language suitable for the user's level
                        - Focus on foundational concepts first

                        Your expertise covers:
                        - Academic subjects (math, science, programming)
                        - Study skills and learning strategies
                        - Problem-solving methodologies
                        - Technical concepts and processes

                        Keep responses focused, practical, and under 200 words unless specifically asked for more detail. If a topic is unclear, ask clarifying questions before providing an explanation."""
                },
                {
                    "role": "user",
                    "content": f"""
                        CONTEXT:
                        - Image Analysis: {image_analysis}
                        - Student Background: {context}

                        REQUIREMENTS:
                        - Analyze the image and student context provided above
                        - Break down the solution into 2-4 clear steps
                        - Focus on core concepts relevant to the student's level
                        - Include a brief explanation for each step
                        - If the problem involves calculations, show key steps

                        FORMAT:
                        - Start with a one-line summary
                        - Number each step clearly
                        - Use simple, precise language
                        - Include relevant formulas or key terms
                        - End with a quick check for understanding

                        Provide your solution following these guidelines, keeping it concise and focused."""
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating solution: {e}")
        return None


async def generate_voice_solution(text: str) -> Optional[str]:
    """Generate voice solution using OpenAI TTS"""
    try:
        response = await asyncio.to_thread(
            openai.audio.speech.create,
            model="tts-1",
            voice="alloy",
            input=text
        )

        # Save the audio file
        output_path = TEMP_DIR / f"solution_{datetime.now().timestamp()}.mp3"
        response.stream_to_file(str(output_path))
        return str(output_path)
    except Exception as e:
        print(f"Error generating voice solution: {e}")
        return None


@app.post("/solve_doubt")
async def solve_doubt(
    image: str = Form(...),  # base64 encoded image
    audio: Optional[UploadFile] = File(None),
    text_context: str = Form("")
):
    try:
        # Process image
        image_data = base64.b64decode(image.split(',')[1])

        # Process audio if present
        audio_text = ""
        if audio:
            audio_path = TEMP_DIR / f"audio_{datetime.now().timestamp()}.webm"
            with audio_path.open("wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
            audio_text = await get_text_from_audio(str(audio_path))
            audio_path.unlink()  # Clean up audio file

        # Combine audio and text context
        context = f"{audio_text} {text_context}".strip()

        # Analyze image
        image_analysis = await analyze_image(image_data)
        if not image_analysis:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to analyze image"}
            )

        # Generate solution
        text_solution = await generate_solution(image_analysis, context)
        if not text_solution:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to generate solution"}
            )

        # Generate voice solution
        voice_solution_path = await generate_voice_solution(text_solution)

        # Save session data
        await save_session(
            'user123',  # Replace with actual user ID
            {
                'image_analysis': image_analysis,
                'context': context,
                'solution': text_solution
            }
        )

        return JSONResponse({
            'text_solution': text_solution,
            'voice_solution': voice_solution_path
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/analyze-doubt")
async def analyze_doubt(image: UploadFile = File(...), doubt: str = Form(...)):
    try:
        # Create temp file path
        temp_file = TEMP_DIR / f"upload_{datetime.now().timestamp()}.png"

        # Save uploaded file
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Encode image
        with open(temp_file, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Call Vision API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this image and answer: {doubt}"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=500
        )

        analysis = response.choices[0].message.content

        # Create session record
        session = DoubtSession(
            timestamp=datetime.now().isoformat(),
            image_analysis=analysis,
            context=doubt
        )

        # Cleanup
        temp_file.unlink()

        return JSONResponse(content={"analysis": analysis})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files"""
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Audio file not found"}
        )
    return FileResponse(str(file_path))


@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    """Get all sessions for a user"""
    if user_id not in sessions:
        return []
    return sessions[user_id]

# Cleanup old files periodically


@app.on_event("startup")
async def startup_event():
    async def cleanup_old_files():
        while True:
            # Delete files older than 1 hour
            current_time = datetime.now().timestamp()
            for file in TEMP_DIR.glob("*"):
                if current_time - file.stat().st_mtime > 3600:  # 1 hour
                    file.unlink()
            await asyncio.sleep(3600)  # Check every hour

    asyncio.create_task(cleanup_old_files())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    START_TIME = time.time()
    return {
        "status": "healthy",
        "uptime": int(time.time() - START_TIME),
        "timestamp": datetime.now().isoformat(),
        "memory_usage": f"{psutil.Process().memory_info().rss / 1024 / 1024:.2f}MB"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
