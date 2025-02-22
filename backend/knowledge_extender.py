
import os
from dotenv import load_dotenv
import requests

load_dotenv()

agent_id = os.getenv("AGENT_ID")
api_key = os.getenv("ELEVENLABS_API_KEY")

# 1. Get Agent Details
url = f"https://api.elevenlabs.io/v1/convai/agents/{agent_id}"
headers = {"xi-api-key": api_key}
response = requests.get(url, headers=headers)

agent_details = response.json()


# 2. Upload document to knowledge base
file_name = "transcript.txt"

url = "https://api.elevenlabs.io/v1/convai/knowledge-base"

with open('transcript.txt', 'rb') as file:
    files = { 
        "file": (file_name, file, "text/plain") 
    }
    
    headers = {"xi-api-key": api_key}

    response = requests.post(url, files=files, headers=headers)


document = response.json()
document['name'] = file_name
document['type'] = "file"


# 3. Update agent

if isinstance(agent_details['conversation_config']['agent']['prompt'].get('knowledge_base', None), list):
    # Append the new document_id to the list
    agent_details['conversation_config']['agent']['prompt']['knowledge_base'].append(document)
else:
    # If the list doesn't exist, initialize it and append the document_id
    agent_details['conversation_config']['agent']['prompt']['knowledge_base'] = [document]


url = f"https://api.elevenlabs.io/v1/convai/agents/{agent_id}"


headers = {
    "xi-api-key": api_key,
    "Content-Type": "application/json"
}
response = requests.patch(url, json=agent_details, headers=headers)
print(response.json())
