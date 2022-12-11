import streamlit as st
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
from playsound import playsound
import  speech_recognition as sr
from deep_translator import GoogleTranslator
from pydub import AudioSegment, silence
from transformers import pipeline

st.markdown(
    "<h1 style='text-align: center;'>Sentiment Analysis App</h1>", unsafe_allow_html=True)
   

st.markdown("---", unsafe_allow_html=True) 

#record voice and extract text from voice
r=sr.Recognizer()
audio = audio_recorder()

if audio:
   
    audio_segment=AudioSegment(audio)
    chunks=silence.split_on_silence(audio_segment, min_silence_len=500, silence_thresh=audio_segment.dBFS-20, keep_silence=100)
    for index,chunk in enumerate(chunks):
        chunk.export(str(index)+".wav", format="wav")
        with sr.AudioFile(str(index)+".wav") as src:
              audio=r.record(src)
        try:
                
                t=r.recognize_google(audio,language='en-US')
                st.write(t)
                obj=gTTS(text=t,lang='en',slow=False)
                obj.save('c:/files/inputtext.mp3')
                audio_file = open('c:/files/inputtext.mp3', 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/ogg')
                
        except sr.UnknownValueError as U:
             print(U)
        except sr.RequestError as R:
             print(R)
#translate text
    translated = GoogleTranslator(source='auto', target='ar').translate(t) 
    st.write(translated)
    obj=gTTS(text=translated,lang='ar',slow=False)
    obj.save('c:/files/outputtext.mp3')
    audio_file = open('c:/files/outputtext.mp3', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')

#sentiment analysis   with distilbert-base-uncased-finetuned-sst-2-english
    classifier = pipeline("sentiment-analysis")    
    result = classifier(t)[0]    
    label = result['label']    
    score = result['score']
    classifier = pipeline("sentiment-analysis")
    result = classifier(t)[0]
    label = result['label']
    score = result['score']
    if label == 'POSITIVE':
        st.success(f'{label} sentiment (score: {score})')
    else:
        st.error(f'{label} sentiment (score: {score})')
    