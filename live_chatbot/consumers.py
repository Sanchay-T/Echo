import json

from channels.generic.websocket import WebsocketConsumer
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import os
from .ai import *


print("Inside Consumers")
class ChatConsumer(WebsocketConsumer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded_chunks = []  # List to accumulate the received audio chunks
        self.sample_rate = 44100  # Set the desired sample rate (you can adjust it as needed)
        self.recording_active = False  # Set the desired sample rate (you can adjust it as needed)

    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data=None , bytes_data=None):

        audio_data = None  # Initialize audio_data to None

        if bytes_data is not None:

            print("Received binary data length:", len(bytes_data))


            # Append the audio segment to the list of recorded chunks
            if len(bytes_data) % 2 == 0:  # For int16 data, each element is 2 bytes
                    audio_data = np.frombuffer(bytes_data, dtype=np.int16)
                    self.recorded_chunks.append(audio_data)
            else:
                self.incomplete_chunk += bytes_data


        else:
            try:
                data = json.loads(text_data)


                if 'event' in data:
                    # Check for WebSocket events
                    event_name = data['event']


                    if event_name == 'start_recording':
                        # Start recording event received
                        self.recording_active = True
                        self.incomplete_chunk = b''
                        self.recorded_chunks = []  

                    elif event_name == 'stop_recording':
                        self.recording_active = False

                        if self.incomplete_chunk:
                            audio_data = np.frombuffer(self.incomplete_chunk, dtype=np.int16)
                            self.recorded_chunks.append(audio_data)
                        print("Received text data:", data)
                        self.create_final_audio()  
            

                else:
                    pass

            except json.JSONDecodeError:
                # Handle JSON decoding errors if necessary
                pass

                    
        # text_data_json = json.loads(text_data)
        # print(text_data_json)
        # audio_chunk = text_data_json["audio_chunk"]
        # index = text_data_json["audio_index"]
        # print("\n")
        # print(f"Audio Chunk {index} -> {audio_chunk}")
        # print("\n")


        # self.send(text_data=json.dumps({"message": audio_chunk}))

    def create_final_audio(self):
        if not self.recorded_chunks:
            print("No audio chunks to create the final audio.")
            return
        # Concatenate the audio segments to create the final audio
        final_audio = np.concatenate(self.recorded_chunks)

        audio_directory = 'audio'
        audio_filename = 'file.wav'
        audio_file_path = os.path.join(audio_directory, audio_filename)

        os.makedirs(audio_directory, exist_ok=True)

        wavfile.write(audio_file_path, self.sample_rate, final_audio.astype(np.int16))

        print("Final audio file saved:", audio_file_path)
        self.recorded_chunks.clear()
        self.recording_active = False
        self.speech_to_text()

    def speech_to_text(self):
        audio_directory = 'audio'
        audio_filename = 'file.wav'
        audio_file_path = os.path.join(audio_directory, audio_filename)
        transcribed_text = stt_function(audio_file_path)
        print(transcribed_text)
        llm_response = llm_run_query(transcribed_text['text'])
        # llm_response.
        self.send(text_data=json.dumps({"message": llm_response}))
