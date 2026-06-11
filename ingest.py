from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os

model = SentenceTransformer("BAAI/bge-m3")

client = chromadb.HttpClient(
    host="localhost",
    port=8001
)

collection = client.get_or_create_collection(
    name="faculdade"
)

DATA_DIR = "data"

documents = []

for root, dirs, files in os.walk(DATA_DIR):

    for file in files:

        if file.endswith(".pdf"):

            path = os.path.join(root, file)

            reader = PdfReader(path)

            text = ""

            for page in reader.pages:
                text += page.extract_text()

            documents.append(
                {
                    "source": file,
                    "content": text
                }
            )

for idx, doc in enumerate(documents):

    embedding = model.encode(
        doc["content"]
    ).tolist()

    collection.add(
        ids=[str(idx)],
        documents=[doc["content"]],
        metadatas=[
            {
                "source": doc["source"]
            }
        ],
        embeddings=[embedding]
    )

print("Documentos indexados.")