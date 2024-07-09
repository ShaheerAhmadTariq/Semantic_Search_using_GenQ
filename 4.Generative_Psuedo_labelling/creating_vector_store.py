import pandas as pd
df = pd.read_csv('../Data_ETL/dataset.csv', dtype=object)
# save the document col to a list
descriptions = df['document'].tolist()

ids = [str(n) for n in range(len(descriptions))]

# # Loading Embedding model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('aditeyabaral/sentencetransformer-bert-base-cased')
embeddings = model.encode(descriptions, show_progress_bar=True)
print("Embeddings generated successfully")

# initialize the vector store
import chromadb
chroma_client = chromadb.PersistentClient(path="./db")
collection_name = "property_collection"
collection = chroma_client.create_collection(name=collection_name)

collection.upsert(
    documents=descriptions,
    embeddings=embeddings,
    ids=ids
)

print("Vector store created successfully")