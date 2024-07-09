from vector_search import search_vector
import pandas as pd

df = pd.read_csv('../5.finetuning_SBERT/dataset_D_Q.csv')
#  create a dict with df['id] as key and df['document] as value
# if two df['id'] are the same, the value will be a str of documents
doc_dict = {}
for idx, row in df.iterrows():
    if row['id'] in doc_dict:
        # Append the new document to the existing string with a space or other delimiter
        doc_dict[row['id']] += " " + row['document']
    else:
        # Create a new entry in the dictionary
        doc_dict[row['id']] = row['document']


def get_relavent_docs(query):
    result = search_vector(query)
    doc_ids = []
    for row in result[0]:
        doc_ids.append(row['entity']['doc_id'])
    docs = []
    for doc in doc_ids:
        # print(doc_dict[int(doc)])
        docs.append(doc_dict[int(doc)])
    # return docs
    df = pd.DataFrame(docs, columns=['document'])
    return df