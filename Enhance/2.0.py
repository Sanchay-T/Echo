import os
import requests
import json
import time
import shutil
import sounddevice as sd
import mimetypes
import tempfile
import numpy as np
from scipy.io.wavfile import write
import gradio as gr
import assemblyai as aai
import openai  # Import OpenAI library

from elevenlabs import clone, generate, play, stream, set_api_key

set_api_key("cedcbf1991539f9c825a9346e1b7b708")
import mimetypes
from gradio.components import Audio, Textbox, Radio
from gradio.components import Audio as AudioInput
from gradio.components import Audio as AudioOutput
from gradio.components import Textbox as TextboxOutput

APP_KEY = "6lWL15cmmm5y5hLYU8-MvQ=="
APP_SECRET = "xoXvx_qwuD5HczjnEYOC9OJj6HGCZDFZBHKHEegigHA="

aai.settings.api_key = "6c7f4d60028e4df9b889b93acb8ed698"

openai.api_key = (
    "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"  # Set OpenAI API key
)


def transcribe_audio_openai(file_path):
    # Transcribe audio using OpenAI's Whisper ASR system
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


def clone_and_stream_voice(name, description, labels, text, model):
    voice = clone(
        name=name, description=description, files=["output.wav"], labels=labels
    )

    audio = generate(
        text=text,
        voice=voice,
        model=model,
        stream=True,
        stream_chunk_size=2048,
        latency=1,
    )

    stream(audio)


def get_access_token():
    payload = {"grant_type": "client_credentials", "expires_in": 1800}
    response = requests.post(
        "https://api.dolby.io/v1/auth/token",
        data=payload,
        auth=requests.auth.HTTPBasicAuth(APP_KEY, APP_SECRET),
    )
    return response.json()["access_token"]


def upload_media(file_path, headers):
    upload_url = "https://api.dolby.com/media/input"
    upload_body = {"url": f"dlb://in/{os.path.basename(file_path)}"}
    response = requests.post(upload_url, json=upload_body, headers=headers)
    response.raise_for_status()
    presigned_url = response.json()["url"]

    with open(file_path, "rb") as input_file:
        requests.put(presigned_url, data=input_file)


def create_enhancement_job(file_path, output_path, headers, audio_type):
    enhance_url = "https://api.dolby.com/media/enhance"
    enhance_body = {
        "input": f"dlb://in/{os.path.basename(file_path)}",
        "output": f"dlb://out/{os.path.basename(output_path)}",
        "content": {"type": audio_type},
    }
    response = requests.post(enhance_url, json=enhance_body, headers=headers)
    response.raise_for_status()
    return response.json()["job_id"]


def check_job_status(job_id, headers):
    status_url = "https://api.dolby.com/media/enhance"
    params = {"job_id": job_id}
    while True:
        response = requests.get(status_url, params=params, headers=headers)
        response.raise_for_status()
        status = response.json()["status"]
        if status == "Success":
            break
        print(f"Job status: {status}, progress: {response.json()['progress']}%")
        time.sleep(5)


def download_enhanced_file(output_path, headers):
    download_url = "https://api.dolby.com/media/output"
    args = {"url": f"dlb://out/{os.path.basename(output_path)}"}
    with requests.get(
        download_url, params=args, headers=headers, stream=True
    ) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        print(f"Downloading from {response.url} into {output_path}")
        with open(output_path, "wb") as output_file:
            shutil.copyfileobj(response.raw, output_file)


def dolby_process(input_file, output_file, audio_type):
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    upload_media(input_file, headers)
    job_id = create_enhancement_job(input_file, output_file, headers, audio_type)
    check_job_status(job_id, headers)
    download_enhanced_file(output_file, headers)


def enhance_audio(recording, upload, audio_type):
    audio_type = audio_type_mapping[audio_type]
    if recording is not None:
        rate, data = recording
        temp_input_file = "input.wav"
    elif upload is not None:
        rate, data = upload
        if rate not in [44100, 48000] or data.dtype not in [np.int16, np.int32]:
            return None, None, "Invalid file type. Please upload an MP3 file."
        temp_input_file = "input.mp3"
    else:
        return (
            None,
            None,
            "Invalid input. Please record some audio or upload an audio file.",
        )

    write(temp_input_file, rate, data)

    temp_output_file = "output.wav"
    dolby_process(
        temp_input_file, temp_output_file, audio_type
    )  # Pass the audio type to the Dolby processing function

    return temp_input_file, temp_output_file, "Processing complete!"


def clone_voice(temp_output_file):
    # Your voice cloning logic goes here
    cloned_voice_file = "cloned_voice.wav"
    return cloned_voice_file, "Voice cloning complete!"


audio_type_mapping = {
    "Conference": "conference",
    "Interview": "interview",
    "Lecture": "lecture",
    "Meeting": "meeting",
    "Mobile Phone": "mobile_phone",
    "Music": "music",
    "Podcast": "podcast",
    "Studio": "studio",
    "Voice Over": "voice_over",
}

from gradio import Checkbox


def combined_function(
    recording,
    upload,
    audio_type,
    proceed_to_clone,
    name,
    description,
    accent,
    age,
    gender,
    use_case,
    model,
):
    labels = {
        "accent": accent,
        "description": description,
        "age": age,
        "gender": gender,
        "use case": use_case,
    }
    input_file, output_file, status1 = enhance_audio(recording, upload, audio_type)
    status1 = "Enhancement complete!"
    transcript = transcribe_audio_openai(output_file)
    if proceed_to_clone:
        clone_and_stream_voice(name, description, labels, transcript, model)
        status2 = "Cloning complete!"
    else:
        status2 = "Voice cloning not performed."
    return input_file, output_file, status1, transcript, status2


def main():
    iface = gr.Interface(
        fn=combined_function,
        inputs=[
            Audio(source="microphone", label="Recorded Audio"),
            Audio(source="upload", label="Uploaded Audio"),
            Radio(choices=list(audio_type_mapping.keys()), label="Audio Type"),
            Checkbox(label="Proceed to Clone Voice"),
            Textbox(label="Name"),
            Textbox(label="Description"),
            Textbox(label="Accent"),
            Textbox(label="Age"),
            Textbox(label="Gender"),
            Textbox(label="Use Case"),
            Radio(
                choices=["eleven_monolingual_v1", "eleven_multilingual_v1"],
                label="Model",
            ),
        ],
        outputs=[
            Audio(type="filepath", label="Original Audio"),
            Audio(type="filepath", label="Processed Audio"),
            Textbox(label="Enhancement Status"),
            Textbox(label="Transcript"),
            Textbox(label="Cloning Status"),
        ],
        title="Audio Enhancer, Transcriber and Voice Cloner",
        description="Enhance your audio, transcribe it and clone voices using the Dolby API",
        allow_flagging="never",
    )

    iface.launch(inbrowser=True, share=True)


if __name__ == "__main__":
    main()
