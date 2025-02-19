import requests

url = "https://huggingface.co/spaces/gauthamk/access-system/resolve/main/embedding_keys.csv" 
response = requests.get(url)

with open("downloaded_file.csv", "wb") as f:
    f.write(response.content)

print("Download complete!")
