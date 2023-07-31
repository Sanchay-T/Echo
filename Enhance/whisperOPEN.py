from elevenlabs import clone, generate, play, stream, set_api_key

set_api_key("cedcbf1991539f9c825a9346e1b7b708")

# speech to text
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
audio_file = open("audio.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)

[response= {
  "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger. This is a place where you can get to do that."
}]

## gpt LLM text to text
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Completion.create(
    model="text-davinci-003", prompt="Say this is a test", max_tokens=7, temperature=0
)

[response= {
  "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
  "object": "text_completion",
  "created": 1589478378,
  "model": "text-davinci-003",
  "choices": [
    {
      "text": "\n\nThis is indeed a test",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12
  }
}
]

# text to speech
from elevenlabs import generate, stream

audio_stream = generate(text="This is a... streaming voice!!", stream=True)

stream(audio_stream)
