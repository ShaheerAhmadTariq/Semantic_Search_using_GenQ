from pymilvus import MilvusClient, DataType
from pymilvus import model

# 1. Set up a Milvus client
client = MilvusClient(
    uri="http://localhost:19530"
)
collection_name = "property_listing"

client.load_collection(
    collection_name=collection_name,
    replica_number=1 # Number of replicas to create on query nodes. Max value is 1 for Milvus Standalone, and no greater than `queryNode.replicas` for Milvus Cluster.
)

res = client.get_load_state(
    collection_name=collection_name
)
sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='../5.finetuning_SBERT/sbert_test_mnr2', # Specify the model name
    device='cuda:0' # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)
def get_query_vector(query):
    return sentence_transformer_ef.encode_queries([query])[0]

def search_vector(query, top_k=5):
    query_vector = get_query_vector(query)
    # 2. Search for similar vectors
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        top_k=top_k,
        params={"nprobe": 16},
        output_fields=["doc_id", "document"]
    )
    return results