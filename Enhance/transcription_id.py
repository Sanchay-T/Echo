import requests
import json

filename = "New Recording 21.m4a"

# Read file data
with open(filename, "rb") as f:
    audio_data = f.read()

# Define headers
headers = {
    "authorization": "6c7f4d60028e4df9b889b93acb8ed698",
}

# Define data
data = {
    "audio_data": audio_data,
    "auto_detect_language": True,  # Enable automatic language detection
}

# Make the API request
response = requests.post(
    "https://api.assemblyai.com/v2/transcript",
    headers=headers,
    data=data,
)

# Parse the JSON response
transcription_results = json.loads(response.text)

# Print the transcription text
print(transcription_results["text"])
