import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Function to update agent with a file
def update_agent_with_file(file_name: str):
    # Fetch API keys and agent ID from environment variables
    agent_id = os.getenv("AGENT_ID")
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not agent_id or not api_key:
        raise ValueError("Agent ID or API key is not set in the environment variables.")

    # 1. Get Agent Details
    url = f"https://api.elevenlabs.io/v1/convai/agents/{agent_id}"
    headers = {"xi-api-key": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch agent details. Status Code: {response.status_code}, Response: {response.text}")

    agent_details = response.json()

    # 2. Upload document to knowledge base
    upload_url = "https://api.elevenlabs.io/v1/convai/knowledge-base"

    with open(file_name, 'rb') as file:
        files = { 
            "file": (file_name, file, "text/plain")  # Adjust MIME type if needed
        }
        
        response = requests.post(upload_url, files=files, headers={"xi-api-key": api_key})

    if response.status_code != 200:
        raise Exception(f"Failed to upload file. Status Code: {response.status_code}, Response: {response.text}")

    document = response.json()

    # Add necessary fields to the document
    document['name'] = file_name
    document['type'] = "file"

    # 3. Update agent with the new document
    # If the list doesn't exist, initialize it and append the document
    agent_details['conversation_config']['agent']['prompt']['knowledge_base'] = [document]

    # 4. Send the PATCH request to update the agent
    patch_url = f"https://api.elevenlabs.io/v1/convai/agents/{agent_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    response = requests.patch(patch_url, json=agent_details, headers=headers)

    if response.status_code == 200:
        print(f"Agent updated successfully with the document '{file_name}'.")
        print(response.json())
    else:
        raise Exception(f"Failed to update agent. Status Code: {response.status_code}, Response: {response.text}")

