import os
import ollama
from dotenv import load_dotenv
load_dotenv()

LOCAL_LLM = os.environ.get('LOCAL_LLM')
def generate_queries(document):
    """ Generate five queries for a given document using the LLaMA model. """
    # Initialize the LLaMA client

    # Construct the prompt to specifically ask for five queries
    prompt = f"Given the following document, generate five distinct questions that one might ask to understand it better:\n\nDocument: {document}"

    # Send the document to the model and request five queries
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ],
        stream=False,)
    queries = response['message']['content'].split('\n')  # Extract the first five queries
    queries = queries[1:]
    # remove empty str from queries
    queries = [q for q in queries if q]
    return queries


from llm import generate_queries
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('./dataset.csv', dtype=object)
# Iterate over df rows with a status bar from tqdm
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Documents"):
    document_str = row['document']
    try:
        queries_list = generate_queries(document_str)  # Your function to generate queries
        count = 1
        for query in queries_list:
            if count > 5:
                break
            df.at[index, f"query{count}"] = query
            count += 1
    except Exception as e:
        print(f"Error processing document ID {row['id']}: {e}")

df.to_csv('./dataset_Q_D.csv', index=False)