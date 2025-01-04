from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import openai
from datetime import datetime
import base64
import tempfile
import os
import asyncio
import shutil
from pathlib import Path
import logging
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Visual Learning Assistant API",
    description="API for processing and solving student doubts using AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Create temp directory for files
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Pydantic models


class DoubtSession(BaseModel):
    session_id: str
    timestamp: str
    image_analysis: str
    context: str
    solution: str
    steps: List[str]


class SolutionResponse(BaseModel):
    text_solution: str
    steps: List[str]
    voice_solution_steps: List[str]


# In-memory session storage
sessions: dict[str, list[DoubtSession]] = {}


class AIAssistant:
    """Class to handle AI-related operations"""

    @staticmethod
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
                                "text": "Please analyze this image and identify the academic question or doubt it contains. Focus on understanding the key concepts and problem areas."
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
            logger.error(f"Error in image analysis: {str(e)}")
            return None

    @staticmethod
    async def transcribe_audio(audio_path: str) -> Optional[str]:
        """Transcribe audio using Whisper"""
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
            logger.error(f"Error in audio transcription: {str(e)}")
            return None

    @staticmethod
    async def generate_solution(image_analysis: str, context: str) -> tuple[list[str], str]:
        """Generate step-by-step solution using GPT-4"""
        try:
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Provide a clear 5-step solution in this exact format:
                        Step 1: [Initial concept/setup]
                        Step 2: [Key process/calculation]
                        Step 3: [Core solution steps]
                        Step 4: [Implementation/solving]
                        Step 5: [Verification/conclusion]
                    
                        Keep each step brief but complete. Format as one continuous narrative."""
                    },
                    {
                        "role": "user",
                        "content": f"Problem: {image_analysis}\nContext: {context}"
                    }
                ],
                max_tokens=1000,
                temperature=0.5
            )

            solution_text = response.choices[0].message.content

            # Split into steps while preserving markdown formatting
            steps = [step.strip()
                     for step in solution_text.split('Step') if step.strip()]
            steps = steps[:5]  # Ensure exactly 5 steps

            audio_response = await openai.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=solution_text
            )

            audio_file_path = f"solution_complete_{int(datetime.now().timestamp())}.mp3"
            audio_response.stream_to_file(audio_file_path)

            return steps, audio_file_path
        except Exception as e:
            logger.error(f"Error generating solution: {str(e)}")
            return [], ""

    @staticmethod
    async def generate_voice_solutions(steps: list[str]) -> list[str]:
        """Generate voice solutions for each step using OpenAI TTS"""
        voice_files = []
        try:
            for i, step in enumerate(steps):
                response = await asyncio.to_thread(
                    openai.audio.speech.create,
                    model="tts-1",
                    voice="nova",  # Using Nova voice for clear educational content
                    input=step,
                    speed=0.9,  # Slightly slower for better comprehension
                    response_format="mp3"
                )

                filename = f"solution_step_{i}_{datetime.now().timestamp()}.mp3"
                output_path = TEMP_DIR / filename
                response.stream_to_file(str(output_path))
                voice_files.append(filename)

            return voice_files
        except Exception as e:
            logger.error(f"Error generating voice solutions: {str(e)}")
            return []


class SessionManager:
    """Class to handle session management"""

    @staticmethod
    async def save_session(session_data: dict) -> str:
        """Save session data and return session ID"""
        session_id = f"session_{datetime.now().timestamp()}"
        session = DoubtSession(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            **session_data
        )

        if session.session_id not in sessions:
            sessions[session.session_id] = []

        sessions[session.session_id].append(session)
        return session_id

    @staticmethod
    async def get_session(session_id: str) -> Optional[DoubtSession]:
        """Retrieve session data"""
        return sessions.get(session_id, None)


class FileManager:
    """Class to handle file operations"""

    @staticmethod
    async def save_temp_file(file: UploadFile, prefix: str) -> Path:
        """Save uploaded file to temp directory"""
        file_path = TEMP_DIR / \
            f"{prefix}_{datetime.now().timestamp()}{Path(file.filename).suffix}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return file_path

    @staticmethod
    async def cleanup_old_files():
        """Clean up files older than 1 hour"""
        current_time = datetime.now().timestamp()
        for file in TEMP_DIR.glob("*"):
            if current_time - file.stat().st_mtime > 3600:  # 1 hour
                file.unlink()

# API Routes


@app.post("/solve_doubt", response_model=SolutionResponse)
async def solve_doubt(
    image: str = Form(...),
    audio: Optional[UploadFile] = File(None),
    text_context: str = Form("")
):
    try:
        # Process image
        image_data = base64.b64decode(image.split(',')[1])

        # Process audio if present
        audio_text = ""
        if audio:
            audio_path = await FileManager.save_temp_file(audio, "audio")
            audio_text = await AIAssistant.transcribe_audio(str(audio_path))
            audio_path.unlink()

        # Combine audio and text context
        context = f"{audio_text} {text_context}".strip()

        # Analyze image
        image_analysis = await AIAssistant.analyze_image(image_data)
        if not image_analysis:
            raise HTTPException(
                status_code=500, detail="Failed to analyze image")

        # Generate solution
        steps, full_solution = await AIAssistant.generate_solution(image_analysis, context)
        if not steps:
            raise HTTPException(
                status_code=500, detail="Failed to generate solution")

        # Generate voice solutions
        voice_solution_files = await AIAssistant.generate_voice_solutions(steps)

        # Save session
        await SessionManager.save_session({
            'image_analysis': image_analysis,
            'context': context,
            'solution': full_solution,
            'steps': steps
        })

        return SolutionResponse(
            text_solution=full_solution,
            steps=steps,
            voice_solution_steps=voice_solution_files
        )

    except Exception as e:
        logger.error(f"Error in solve_doubt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio files"""
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(file_path))


@app.get("/sessions/{session_id}")
async def get_session_data(session_id: str):
    """Retrieve session data"""
    session = await SessionManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

# Startup and shutdown events


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    TEMP_DIR.mkdir(exist_ok=True)

    # Start cleanup task
    async def cleanup_task():
        while True:
            await FileManager.cleanup_old_files()
            await asyncio.sleep(3600)  # Run every hour

    asyncio.create_task(cleanup_task())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
