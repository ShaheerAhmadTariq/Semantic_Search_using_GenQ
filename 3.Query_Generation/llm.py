import os
from ollama import Client
from dotenv import load_dotenv
load_dotenv()

LOCAL_LLM = os.environ.get('LOCAL_LLM')
def generate_queries(document):
    """ Generate five queries for a given document using the LLaMA model. """
    # Initialize the LLaMA client
    client = Client(host=LOCAL_LLM)

    # Construct the prompt to specifically ask for five queries
    prompt = f"Given the following document, generate five distinct questions that one might ask to understand it better:\n\nDocument: {document}"

    # Send the document to the model and request five queries
    response = client.chat(model='llama2', messages=[
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