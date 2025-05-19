import requests
from urllib.parse import quote
from pathlib import Path

Path("tests/biodivine_db").mkdir(parents=True, exist_ok=True)

url = "https://api.github.com/repos/sybila/biodivine-boolean-models/contents/models"

response = requests.get(url)

names = [quote(el['name']) for el in response.json()]

urls = ["https://github.com/sybila/biodivine-boolean-models/blob/main/{}/model.sbml".format(name) for name in names]

for i in range(len(urls)):
    response = requests.get(urls[i])
    with open("tests/biodivine_db/model{}.sbml".format(i), 'w') as f:
        f.write(response.text)