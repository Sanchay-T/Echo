import assemblyai as aai

aai.settings.api_key = "6c7f4d60028e4df9b889b93acb8ed698"
transcriber = aai.Transcriber()

transcript = transcriber.transcribe("output.wav")
# transcript = transcriber.transcribe("./my-local-audio-file.wav")

print(transcript.text)
