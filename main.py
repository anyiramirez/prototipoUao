from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline

app = FastAPI()

# Cargar el pipeline de análisis de sentimientos
sentiment_pipeline = pipeline("sentiment-analysis")


def transcribe_audio(file_path):
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Open the audio file
    with sr.AudioFile(file_path) as source:
        # Record the audio data from the file
        audio_data = recognizer.record(source)
        
        # Recognize (convert from speech to text) using Google Web Speech API
        try:
            text = recognizer.recognize_google(audio_data, language="es-ES")
            return text
        except sr.UnknownValueError:
            return "Google Web Speech API could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Web Speech API; {e}"

def analyze_text(text):
    # Analizar el texto transcrito
    analysis = sentiment_pipeline(text)
    return analysis

@app.get("/hola")
async def hola():
    return {"message": "¡Hola desde FastAPI!"}

@app.post("/transcribe_and_analyze")
async def transcribe_and_analyze(file: UploadFile = File(...)):
    try:
        # Guardar el archivo subido temporalmente
        file_location = f"temp.{file.filename.split('.')[-1]}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        
        # Transcribir el audio
        transcription = transcribe_audio(file_location)
        
        # Analizar el texto transcrito
        analysis = analyze_text(transcription)
        
        # Devolver la respuesta en formato JSON
        response = {
            "transcription": transcription,
            "analysis": analysis
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)