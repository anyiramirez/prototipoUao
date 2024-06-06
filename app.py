import streamlit as st
import requests
import json

st.title("Transcripción y Análisis de Sentimientos de Audio/Video")

uploaded_file = st.file_uploader("Elija un archivo de audio o video", type=["mp3", "mp4", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Transcribir y Analizar"):
        with st.spinner("Transcribiendo y Analizando..."):
            try:
                # Enviar el archivo a la API
                files = {'file': uploaded_file.getvalue()}
                response = requests.post("http://localhost:8000/transcribe_and_analyze", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    transcription = result.get("transcription", "")
                    analysis = result.get("analysis", [])
                    
                    st.subheader("Transcripción")
                    st.write(transcription)
                    
                    st.subheader("Análisis de Sentimientos")
                    for item in analysis:
                        st.write(f"Sentimiento: {item['label']} - Puntaje: {item['score']:.2f}")
                else:
                    st.error(f"Error al analizar el archivo: {response.content.decode()}")
            except Exception as e:
                st.error(f"Se produjo un error: {str(e)}")
