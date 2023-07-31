from flask import Flask
from flask_socketio import SocketIO, send
import openai
import os
from elevenlabs import clone, generate, play, stream, set_api_key

app = Flask(__name__)
socketio = SocketIO(app)

set_api_key("cedcbf1991539f9c825a9346e1b7b708")
openai.api_key = "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"


@socketio.on("message")
def handle_message(data):
    # Convert client audio data to format suitable for OpenAI and Eleven Labs

    # Transcribe audio using OpenAI
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # Process text with GPT-3
    response = openai.Completion.create(
        model="text-davinci-003", prompt=transcript["text"], max_tokens=7, temperature=0
    )

    # Convert response text to speech with Eleven Labs
    audio_stream = generate(text=response["choices"][0]["text"], stream=False)
    play(audio_stream)

    # Send audio response to client
    send(audio_stream)


if __name__ == "__main__":
    socketio.run(app)
