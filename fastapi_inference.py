import os
import torch
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from io import BytesIO
import numpy as np
from pydub import AudioSegment
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import subprocess
import soundfile as sf

# Define the Inferencer class
class Inferencer:
    def __init__(self, device, huggingface_folder, model_path) -> None:
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(huggingface_folder)
        self.model = Wav2Vec2ForCTC.from_pretrained(huggingface_folder).to(self.device)
        if model_path is not None:
            self.preload_model(model_path)

    def preload_model(self, model_path) -> None:
        assert os.path.exists(model_path), f"The file {model_path} does not exist. Please check the path."
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=True)
        else:
            self.model.load_state_dict(checkpoint, strict=True)
        
        print(f"Model preloaded successfully from {model_path}.")

    def transcribe(self, wav) -> str:
        # Process and transcribe the audio
        input_values = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_values
        logits = self.model(input_values.to(self.device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_transcript = self.processor.batch_decode(pred_ids)[0]
        return pred_transcript

# FastAPI setup
app = FastAPI()

# Add CORS middleware to allow cross-origin requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all domains, you can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load model paths (ensure these paths are correct for your setup)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"C:\ASR-Wav2vec-Finetune\saved\ASR\checkpoints\pytorch_model.bin"
huggingface_folder = r"C:\ASR-Wav2vec-Finetune\saved\ASR\checkpoints"

inferencer = Inferencer(device=device, huggingface_folder=huggingface_folder, model_path=model_path)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def convert_audio_to_wav(audio_bytes: BytesIO, file_ext: str) -> BytesIO:
    """Convert any audio format to WAV using pydub and return as BytesIO"""
    try:
        audio = AudioSegment.from_file(audio_bytes, format=file_ext)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Resample to 16kHz

        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)  # Reset pointer for reading
        return wav_io
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error converting audio: {str(e)}")

# def process_audio(file_bytes: BytesIO, file_ext: str) -> str:
#     """Process any audio file format and return transcription"""
#     try:
#         wav_io = convert_audio_to_wav(file_bytes, file_ext)
#         wav, sr = librosa.load(wav_io, sr=16000)  # Resample to 16kHz
#         print(f"Audio loaded: Sample Rate: {sr}, Audio shape: {wav.shape}")
#     except Exception as e:
#         raise HTTPException(status_code=422, detail=f"Error loading audio: {str(e)}")
    
#     try:
#         transcription = inferencer.transcribe(wav)
#         print(f"Transcription result: {transcription}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")
    
#     return transcription

def process_audio(file_bytes: BytesIO, file_ext: str) -> str:
    """Process any audio file format and return transcription"""
    try:
        # Convert to WAV first
        wav_io = convert_audio_to_wav(file_bytes, file_ext)

        # Load the WAV file using librosa
        wav, sr = librosa.load(wav_io, sr=16000)  # Resample to 16kHz
        print(f"Audio loaded: Sample Rate: {sr}, Audio shape: {wav.shape}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error loading audio: {str(e)}")
    
    # Transcribe the audio
    try:
        transcription = inferencer.transcribe(wav)
        print(f"Transcription result: {transcription}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")
    
    return transcription


# Helper function to process audio and return transcription
# def process_audio(file_bytes: BytesIO) -> str:
#     try:
#         # Load audio from bytes and resample to 16kHz
#         wav, sr = librosa.load(file_bytes, sr=16000)  # Always resample to 16kHz for consistency
#         print(f"Audio loaded: Sample Rate: {sr}, Audio shape: {wav.shape}")
#     except Exception as e:
#         raise HTTPException(status_code=422, detail=f"Error loading audio: {str(e)}")
    
#     # Transcribe the audio
#     try:
#         transcription = inferencer.transcribe(wav)
#         print(f"Transcription result: {transcription}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")
    
#     return transcription





@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split(".")[-1].lower()
        if 'audio' not in file.content_type:
            raise HTTPException(status_code=415, detail="File is not an audio file")
        
        audio_bytes = await file.read()
        audio_file = BytesIO(audio_bytes)
        transcription = process_audio(audio_file, file_ext)
        
        return {"transcription": transcription}
    except HTTPException as e:
        print(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
app.mount("/static", StaticFiles(directory="src"), name="static")

@app.get("/upload-page")
async def serve_upload_page():
    file_path = Path("src/upload.html")
    if file_path.exists():
        return FileResponse(file_path)
    return {"error": "upload.html not found"}



@app.post("/record/")
async def process_recorded_audio(audio: UploadFile = File(...)):
    try:
        file_ext = audio.filename.split(".")[-1].lower()
        if file_ext not in ["wav", "mp3", "flac", "ogg", "m4a", "webm"]:
            raise HTTPException(status_code=415, detail="Unsupported audio format")
        
        audio_bytes = await audio.read()
        audio_file = BytesIO(audio_bytes)

        # Convert webm to wav if needed
        if file_ext == "webm":
            webm_path = "temp_audio.webm"
            wav_path = "temp_audio.wav"
            with open(webm_path, "wb") as f:
                f.write(audio_bytes)

            # Convert to WAV using FFmpeg
            # ffmpeg_cmd = ["ffmpeg", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path, "-y"]
            ffmpeg_cmd = ["ffmpeg", "-i", webm_path, "-ar", "16000", "-ac", "1", "-af", "highpass=200, lowpass=3000", wav_path, "-y"]
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
            audio_file = BytesIO(audio_bytes)

        transcription = process_audio(audio_file, "wav")
        
        return JSONResponse(content={"message": "Audio recorded successfully", "transcription": transcription})
    except Exception as e:
        print(f"Audio Processing Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")




# @app.post("/upload/")
# async def upload_file(file: UploadFile = File(...)):
#     """Handles file uploads"""
#     try:
#         file_location = os.path.join(UPLOAD_DIR, file.filename)
#         with open(file_location, "wb") as buffer:
#             buffer.write(await file.read())

#         return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})
#     except Exception as e:
#         print(f"Upload Error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handles file uploads"""
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})
    except Exception as e:
        print(f"Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


