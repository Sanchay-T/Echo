from elevenlabs import clone, generate, stream, set_api_key


def clone_voice(output_file, name, description, labels, text, model):
    set_api_key("cedcbf1991539f9c825a9346e1b7b708")

    voice = clone(
        name=name,
        description=description,  # Optional
        files=[output_file],
        labels=labels,
    )

    audio = generate(
        text=text,
        voice=voice,
        model=model,
        stream=True,  # If True, returns a generator streaming bytes
        stream_chunk_size=2048,  # Size of each chunk when stream=True
        latency=1,
    )  # Either a model name or Model object

    stream(audio)

    cloned_voice_file = "cloned_voice.wav"
    return cloned_voice_file, "Voice cloning complete!"
