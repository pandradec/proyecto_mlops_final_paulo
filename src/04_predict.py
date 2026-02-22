import requests
import json

# URL of the running API
url = "http://127.0.0.1:5000/predict"

# Sample input data (same structure as original dataset)
input_data = {
    "Age": 35,
    "Sex": "male",
    "Job": 2,
    "Housing": "own",
    "Saving accounts": "little",
    "Checking account": "moderate",
    "Credit amount": 3000,
    "Duration": 24,
    "Purpose": "car"
}

print("Sending request to model API...")

# Send POST request
response = requests.post(url, json=input_data)

# Print response
if response.status_code == 200:
    print("Prediction successful!")
    print("Response:")
    print(json.dumps(response.json(), indent=4))
else:
    print("Error:", response.status_code)
    print(response.text)