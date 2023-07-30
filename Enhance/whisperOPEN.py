from elevenlabs import voices, generate
from elevenlabs import set_api_key

set_api_key("cedcbf1991539f9c825a9346e1b7b708")
voices = voices()

audio = generate(text="Hello there!", voice=voices[0])

print(voices)
