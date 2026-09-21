import requests

response = requests.get("https://catfact.ninja/fact")
data = response.json()

print("Status code:", response.status_code)
print("Cat fact:", data["fact"])