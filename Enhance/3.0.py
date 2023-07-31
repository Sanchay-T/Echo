# Standard library imports
import os
import time
import shutil
import tempfile
from dotenv import load_dotenv

# Third party imports
import requests
import numpy as np
import gradio as gr
import assemblyai as aai
import sounddevice as sd
import openai
from scipy.io.wavfile import write
from gradio.components import Audio, Textbox, Radio, Checkbox
from elevenlabs import clone, generate, play, stream, set_api_key

# Local application imports
load_dotenv()
# Set API keys
set_api_key(os.getenv("ELEVENLABS_API_KEY"))
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
APP_KEY = os.getenv("DOLBY_APP_KEY")
APP_SECRET = os.getenv("DOLBY_APP_SECRET")
# Define constants
AUDIO_TYPE_MAPPING = {
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


cloned_voice = None


class ChatWrapper:
    def __init__(self, generate_speech, generate_text):
        self.lock = Lock()
        self.generate_speech = generate_speech
        self.generate_text = generate_text
        self.s2t_processor_ref = bentoml.models.get("whisper_processor:latest")
        self.processor = bentoml.transformers.load_model(self.s2t_processor_ref)

    def __call__(
        self,
        api_key: str,
        audio_path: str,
        text_message: str,
        history: Optional[Tuple[str, str]],
        chain: Optional[ConversationChain],
    ):
        """Execute the chat functionality."""
        self.lock.acquire()

        print(f"audio_path : {audio_path} ({type(audio_path)})")
        print(f"text_message : {text_message} ({type(text_message)})")

        try:
            if audio_path is None and text_message is not None:
                transcription = text_message
            elif audio_path is not None and text_message in [None, ""]:
                audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column(
                    "audio",
                    Audio(sampling_rate=16000),
                )
                sample = audio_dataset[0]["audio"]

                if sample is not None:
                    input_features = self.processor(
                        sample["array"],
                        sampling_rate=sample["sampling_rate"],
                        return_tensors="pt",
                    ).input_features

                    transcription = self.generate_text(input_features)
                else:
                    transcription = None
                    speech = None

            if transcription is not None:
                history = history or []
                # If chain is None, that is because no API key was provided.
                if chain is None:
                    response = "Please paste your Open AI key."
                    history.append((transcription, response))
                    speech = (PLAYBACK_SAMPLE_RATE, self.generate_speech(response))
                    return history, history, speech, None, None
                # Set OpenAI key
                import openai

                openai.api_key = api_key
                # Run chain and append input.
                output = chain.run(input=transcription)
                speech = (PLAYBACK_SAMPLE_RATE, self.generate_speech(output))
                history.append((transcription, output))

        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history, speech, None, None


chat = ChatWrapper(generate_speech, generate_text)


def clone_and_stream_voice(name, description, labels):
    voice = clone(
        name=name, description=description, files=["output.wav"], labels=labels
    )

    return voice


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
    input_file, output_file, _ = enhance_audio(recording, upload, audio_type)
    if proceed_to_clone:
        voice = clone_and_stream_voice(name, description, labels, model)
    else:
        voice = "Voice cloning not performed."

    cloned_voice = voice
    return voice


def user_text():
    pass


def user_audio():
    pass


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
        outputs=Textbox(label="Cloned Voice"),
        title="Audio Enhancer, Transcriber and Voice Cloner",
        description="Enhance your audio, transcribe it and clone voices using the Dolby API",
        allow_flagging="never",
    )

    # iface.launch(inbrowser=True, share=True)

    # iface_2 = gr.Interface(
    #     fn=combined_function,
    #     inputs=[gr.Microphone(label="Speak Your Query")],
    #     outputs=Textbox(label="Cloned Voice"),
    #     title="Agent Vinod",
    #     description="Enhance your audio, transcribe it and clone voices using the Dolby API",
    #     allow_flagging="never",
    # )
    block = gr.Blocks(css=".gradio-container")

    with block:
        with gr.Row():
            gr.Markdown("<h3><center>BentoML LangChain Demo</center></h3>")

            openai_api_key_textbox = gr.Textbox(
                placeholder="Paste your OpenAI API key (sk-...)",
                show_label=False,
                lines=1,
                type="password",
            )

        chatbot = gr.Chatbot()

        audio = gr.Audio(label="Chatbot Voice", elem_id="chatbox_voice")

        with gr.Row():
            audio_message = gr.Audio(
                label="User voice message",
                source="microphone",
                type="filepath",
            )

            text_message = gr.Text(
                label="User text message",
                placeholder="Give me 5 gift ideas for my mother",
            )

        gr.HTML("Demo BentoML application of a LangChain chain.")

        gr.HTML(
            "<center>Powered by <a href='https://github.com/bentoml/BentoML'>BentoML 🍱</a> and <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
        )

        state = gr.State()
        agent_state = gr.State()

        audio_message.change(
            user_audio,
            inputs=[
                openai_api_key_textbox,
                audio_message,
                text_message,
                state,
                agent_state,
            ],
            outputs=[chatbot, state, audio, audio_message, text_message],
            show_progress=False,
        )

        text_message.submit(
            user_text,
            inputs=[
                openai_api_key_textbox,
                audio_message,
                text_message,
                state,
                agent_state,
            ],
            outputs=[chatbot, state, audio, audio_message, text_message],
            show_progress=False,
        )

        # openai_api_key_textbox.change(
        #     set_openai_api_key,
        #     inputs=[openai_api_key_textbox],
        #     outputs=[agent_state],
        #     show_progress=False,
        # )

    iface_2.launch(inbrowser=True, share=True)

    demo = gr.TabbedInterface([iface, block], ["Text-to-speech", "Agent Vinod"])
    demo.launch(share=True)


if __name__ == "__main__":
    main()
