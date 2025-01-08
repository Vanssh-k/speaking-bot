# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2AuthorizationCodeBearer
from starlette.middleware.sessions import SessionMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
import os
from moviepy.editor import VideoFileClip
import assemblyai as aai
from openai import OpenAI
from dotenv import load_dotenv
from httpx import AsyncClient

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI client
openAIClient = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# MongoDB connection
mongoClient = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
db = mongoClient['test']

aai.settings.api_key = os.getenv('ASSEMBLY_API_KEY')
transcriber = aai.Transcriber()

# User model
class User(BaseModel):
    email: str
    name: str
    picture: str
    videos: list = []

# Add session middleware for storing user session
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY"))

# Dependency to check if user is authenticated
def get_current_user(request: Request):
    user = request.session.get("user")
    if not user:
        print("User not authenticated, redirecting to login.")
        raise HTTPException(status_code=307, detail="Redirecting to login", headers={"Location": "/login"})
    return user

# OAuth2 configuration
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/auth",
    tokenUrl="https://oauth2.googleapis.com/token",
)

# Function to get Google login URL
def get_google_login_url():
    return (
        "https://accounts.google.com/o/oauth2/auth?response_type=code"
        "&client_id={}&redirect_uri={}&scope=openid%20email%20profile"
    ).format(os.getenv("GOOGLE_CLIENT_ID"), os.getenv("REDIRECT_URI"))

# Function to check if a file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}

# Function to extract audio from video
def extract_audio_from_video(video_file, audio_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(audio_file)
    audio.close()
    video.close()

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    result = transcriber.transcribe(audio_file)
    return result.text

async def analyze_text(email, transcription):

    previous_analysis = await db.users.find_one(
        {"email": email},
        {"videos": {"$slice": -2}}
    )
    
    previous_analysis = previous_analysis.get("videos", []) if previous_analysis else []

    context = "\n".join([analysis["analysis"] for analysis in previous_analysis])

    print(context)

    response = openAIClient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "system",
            "content": "You are an AI assistant designed to help users enhance their spoken language proficiency. The user has recorded a video of themselves speaking, from which the audio has been extracted and transcribed into text. Your task is to analyze this text, considering it originates from spoken language. Provide a comprehensive analysis that identifies areas for improvement, incorrect usage, and suggests alternative word choices to enhance their speech. Analyze the current transcription considering the analysis of the last two transcripts if available."
          },
            {"role": "user", "content": f"Please analyze the following transcription of my speech. Previous analysis: {context}\n\nCurrent transcription: {transcription}"}
        ]
    )
    
    return response.choices[0].message.content

# Function to generate a report from transcription
def generate_report(transcription, analysis):
    report = {
        "total_words": len(transcription.split()),
        "analysis": analysis
    }
    return report

# Function to save video info to MongoDB
async def save_video_info(email, video_info):
    await db.users.update_one(
        {"email": email},
        {"$push": {"videos": video_info}}
    )

# Route to handle Google login
@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "login_url": get_google_login_url()})

# Route to handle Google callback
@app.get("/auth")
async def auth(code: str, request: Request):
    async with AsyncClient() as client:
        response = await client.post("https://oauth2.googleapis.com/token", data={
            "code": code,
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "redirect_uri": os.getenv("REDIRECT_URI"),
            "grant_type": "authorization_code",
        })
        token_data = response.json()
        access_token = token_data['access_token']
        resp = await client.get("https://www.googleapis.com/oauth2/v1/userinfo", headers={"Authorization": f"Bearer {access_token}"})
        user_info = resp.json()
        
        request.session["user"] = user_info

        existing_user = await db.users.find_one({"email": user_info["email"]})
        if not existing_user:
            await db.users.insert_one({"email": user_info["email"], "name": user_info["name"], "picture": user_info["picture"], "videos": []})

        return RedirectResponse(url="/")

# Route to upload files
@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")

    os.makedirs('uploads', exist_ok=True)
    
    video_path = os.path.join('uploads', file.filename)
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    audio_file_path = os.path.join('uploads', 'extracted_audio.wav')
    extract_audio_from_video(video_path, audio_file_path)

    transcription = transcribe_audio(audio_file_path)
    analysis = await analyze_text(current_user["email"], transcription)
    report = generate_report(transcription, analysis)

    # Save video info to user's videos list
    video_info = {
        "name": file.filename,
        "transcript": transcription,
        "analysis": analysis
    }
    await save_video_info(current_user["email"], video_info)

    # Clean up: remove video and audio files after processing
    os.remove(video_path)
    os.remove(audio_file_path)

    return templates.TemplateResponse("report.html", {"request": request, "report": report, "current_user": current_user})

# Route to render the upload page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, current_user: dict = Depends(get_current_user)):
    return templates.TemplateResponse("upload.html", {"request": request, "current_user": current_user})

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="info")