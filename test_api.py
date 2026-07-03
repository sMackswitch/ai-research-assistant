import os
from google import genai

# 1. Initialize the official Google GenAI client
# Make sure you've set your environment variable, or paste your key directly: Client(api_key="YOUR_KEY")
client = genai.Client()

# 2. Test a basic generation call
print("Sending test request to Gemini...")
response = client.models.generate_content(
    model='gemini-3.5-flash',
    contents='Tell me one joke about programming.',
)

print("\nResult from API:")
print(response.text)