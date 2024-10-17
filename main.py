import streamlit as st
import tempfile
import os
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from moviepy.editor import VideoFileClip, AudioFileClip
from openai import AzureOpenAI
import soundfile as sf
from moviepy.video.fx import all as vfx

# Google Cloud credentials
gcp_credentials = st.secrets["gcp"]["credentials"]

# Azure OpenAI credentials
azure_openai_api_key = st.secrets["azure"]["openai_api_key"]
client = AzureOpenAI(
    api_key=azure_openai_api_key,
    api_version="2024-08-01-preview",
    azure_endpoint="https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
)

def convert_to_mono(audio_path):
    audio_data, sample_rate = sf.read(audio_path)

    # Check if the audio is stereo, and convert to mono if necessary
    if audio_data.ndim == 2 and audio_data.shape[1] == 2:
        # Downmix stereo to mono
        mono_audio = audio_data.mean(axis=1)
        mono_audio_path = "mono_audio.wav"
        # Save as mono
        sf.write(mono_audio_path, mono_audio, sample_rate)
        return mono_audio_path
    else:
        # Return original if already mono
        return audio_path


def transcribe_audio(audio_file):
    client = speech.SpeechClient()

    # Open audio file in binary mode
    with open(audio_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    return " ".join([result.alternatives[0].transcript for result in response.results])


def correct_text(text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that corrects grammar and removes filler words."},
            {"role": "user",
             "content": f"Please correct the following text, removing grammatical mistakes and filler words: {text}"}
        ]
    )
    return response.choices[0].message.content


def text_to_speech(text):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Journey-F"
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response.audio_content


def replace_audio(video_path, audio_content):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(audio_content)
        temp_audio_path = temp_audio.name

    video = VideoFileClip(video_path)
    audio = AudioFileClip(temp_audio_path)

    # Adjust audio duration to match video duration
    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)
    elif audio.duration < video.duration:
        # Ensure audio matches video duration
        audio = audio.fx(vfx.loop, duration=video.duration)

    final_clip = video.set_audio(audio)

    output_path = "output_video.mp4"
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    os.unlink(temp_audio_path)
    return output_path


st.title("Video Audio Replacement PoC")

# File upload widget for video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    # Button to trigger the video processing
    if st.button("Process Video"):
        with st.spinner("Processing..."):
            # Extract audio from video
            video = VideoFileClip(video_path)
            audio = video.audio
            audio_path = "temp_audio.wav"
            audio.write_audiofile(audio_path)

            # Convert audio to mono if needed
            mono_audio_path = convert_to_mono(audio_path)

            # Transcribe audio using Google Speech-to-Text
            transcription = transcribe_audio(mono_audio_path)
            st.text("Original Transcription:")
            st.write(transcription)

            # Correct the transcription text using Azure GPT-4
            corrected_text = correct_text(transcription)
            st.text("Corrected Transcription:")
            st.write(corrected_text)

            # Convert corrected text back to speech
            new_audio_content = text_to_speech(corrected_text)

            # Replace the original audio in the video with the new audio
            output_path = replace_audio(video_path, new_audio_content)

            st.success("Video processing complete!")
            st.video(output_path)

        # Clean up temporary files
        os.unlink(video_path)
        os.unlink(audio_path)
        os.unlink(mono_audio_path)
        os.unlink(output_path)
