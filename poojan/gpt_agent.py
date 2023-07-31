import keyboard
import os
import tempfile

import numpy as np
import openai
import sounddevice as sd
import soundfile as sf
import tweepy

from elevenlabs import generate, play, set_api_key
from langchain.agents import initialize_agent, load_tools
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.utilities.zapier import ZapierNLAWrapper

stop_flag = True


def set_stop_flag(e):
    global stop_flag
    stop_flag = True


keyboard.on_press_key("c", set_stop_flag)

set_api_key("dcb21c9f4a8176f2f19148f63cde21e4")
openai.api_key = "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"

duration = 15  # duration of each recording in seconds
fs = 44100  # sample rate
channels = 1  # number of channels


def record_audio(duration, fs, channels):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    print("Finished recording.")
    return recording


def transcribe_audio(recording, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, recording, fs)
        temp_audio.close()
        with open(temp_audio.name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        os.remove(temp_audio.name)
    return transcript["text"].strip()


def play_generated_audio(text, voice="Bella", model="eleven_monolingual_v1"):
    audio = generate(text=text, voice=voice, model=model)
    play(audio)


if __name__ == "__main__":
    llm = OpenAI(
        openai_api_key="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",
        temperature=0,
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    zapier = ZapierNLAWrapper(zapier_nla_api_key="sk-ak-jed9SzIRg7R2F5sio6orTRGX5D")
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

    tools = toolkit.get_tools() + load_tools(["human"])

    agent = initialize_agent(
        tools,
        llm,
        memory=memory,
        agent="conversational-react-description",
        verbose=True,
    )

    while not stop_flag:  # checking the stop_flag in loop condition
        print("Press spacebar to start recording.")
        keyboard.wait("space")  # wait for spacebar to be pressed
        recorded_audio = record_audio(duration, fs, channels)
        message = transcribe_audio(recorded_audio, fs)
        print(f"You: {message}")
        assistant_message = agent.run(message)
        play_generated_audio(assistant_message)
