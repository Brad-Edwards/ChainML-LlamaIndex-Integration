import os
import requests

import dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader

dotenv.load_dotenv()
print(os.getenv("OPENAI_API_KEY", None) is not None)

url = (
    "https://github.com/jerryjliu/llama_index/blob/main/examples/gatsby/gatsby_full.txt"
)
filename = url.split("/")[-1]

os.makedirs("gatsby_download", exist_ok=True)

response = requests.get(url)
with open(os.path.join("gatsby_download", filename), "wb") as f:
    f.write(response.content)

documents = SimpleDirectoryReader("gatsby_download").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("Where do Gatsby and Daisy meet?")
print(response)
