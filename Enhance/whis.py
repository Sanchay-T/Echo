import os
import openai

openai.api_key = "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"
audio_file = open("New Recording 21.m4a", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript["text"])
