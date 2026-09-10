import requests

url = "http://localhost:8000/api/v1/repositories/index"
data = {
    "repository_url": "https://github.com/PransuVadsmiya/movie-reservation-system"
}
response = requests.post(url, json=data)
print(response.status_code)
print(response.json())
