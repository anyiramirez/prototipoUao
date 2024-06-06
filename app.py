import streamlit as st
import speech_recognition as sr
from transformers import pipeline

# Función para transcribir audio
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="es-ES")
            return text
    except sr.UnknownValueError:
        return "Google Web Speech API no pudo entender el audio"
    except sr.RequestError as e:
        return f"No se pudieron solicitar resultados de la API de Google Web Speech; {e}"
    except Exception as e:
        return f"Ocurrió un error: {e}"

# Interfaz de Streamlit
st.title("Transcripción de Audio a Texto y Análisis de Sentimientos")
st.header("Carga un archivo de audio en formato WAV para transcribir y analizar")

# Mensaje de carga para el pipeline de análisis de sentimientos
with st.spinner("Cargando el modelo de análisis de sentimientos..."):
    sentiment_pipeline = pipeline("sentiment-analysis")
st.success("Modelo de análisis de sentimientos cargado")

uploaded_file = st.file_uploader("Elige un archivo de audio...", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Iniciar Transcripción y Análisis"):
        # Guardar el archivo subido temporalmente
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Transcribir el audio
        with st.spinner("Transcribiendo el audio..."):
            transcription = transcribe_audio("temp_audio.wav")
        st.success("Transcripción completada")
        st.header("Transcripción")
        st.write(transcription)
        
        if transcription:
            # Analizar el texto transcrito
            with st.spinner("Realizando análisis de sentimientos..."):
                analysis = sentiment_pipeline(transcription)
            st.success("Análisis de sentimientos completado")
            st.header("Análisis de Sentimientos")
            
            # Mostrar el análisis de sentimientos con estilo
               # Mostrar el análisis de sentimientos con estilo y traducción
            for result in analysis:
                label = result['label']
                score = result['score']
                if label == "NEGATIVE":
                    label = "NEGATIVO"
                    st.markdown(f"**El audio expresa un sentimiento:** :red[{label}]")
                else:
                    label = "POSITIVO"
                    st.markdown(f"**El audio expresa un sentimiento:** :green[{label}]")
                st.markdown(f"**Confianza:** {score:.2%}")

