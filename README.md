# Semantic_Search_using_GenQ
Training embedding model to generate better sematic search engine

## Generative query network (GenQ)

is a training approach used in Natural Language Processing (NLP) and information retrieval. The core concept of GenQ involves the creation of query-document pairs where each query is algorithmically generated and associated with a relevant document. This pairing forms the training dataset.

The training process leverages a technique called Multiple Negatives Ranking Loss (MNRL). MNRL is used to optimize the model by effectively ranking the relevance of multiple documents against a given query. The idea is to improve the model's ability to discern and rank the most relevant document higher compared to less relevant ones.

Additionally, GenQ involves the fine-tuning of the SBERT model, a variant of the BERT model optimized for sentence-level tasks. By fine-tuning SBERT using MNRL, the model becomes better at understanding the nuances of language used in the queries and documents, enhancing its performance in tasks like document retrieval or question answering.

## Methods and implementation

### Data Preparation

Handle the initial processing of property listings from a database in Spain. The process involves these steps:

1. **Data Retrieval**: We start by extracting all the property listings from our database.
2. **DataFrame Creation**: The data is organized into a DataFrame that includes an identifier and the property descriptions, along with five additional columns for queries related to each property.
3. **Document Splitting**: For text entries longer than 1024 characters, the documents are split into smaller segments of 100 characters. This ensures that the data remains manageable and each segment retains its association with the original property ID.
4. **Data Export**: The processed data is compiled into a new DataFrame and exported as a CSV file for future use.

### Generating Labeled Data Using LLaMA 3 Model

This phase involves using the LLaMA 3 model to enhance our dataset with generated queries:

1. **Model Initialization**: We start by setting up the LLaMA 3 model, ensuring it's ready to generate queries.
2. **Query Generation**: For each document, the model generates five relevant queries aimed at better understanding the document's content.
3. **Data Association**: These queries are then linked back to their respective documents, effectively labeling the dataset.
4. **Data Export**: Finally, this labeled dataset is saved, making it available for further analysis or training processes.

### TSDAE (Transformer-based Denoising AutoEncoder) Training

This section details the training of a Transformer-based Denoising AutoEncoder (TSDAE) using the BERT model architecture. The process unfolds through several key steps:

1. **Data Preparation**: We begin by loading the dataset and extracting document texts into a list, which serves as our training data.
2. **Model Setup**: A BERT model (`bert-base-uncased`) is configured with CLS pooling to capture the essence of each sentence. This setup includes integrating a word embedding model and a pooling layer.
3. **Dataset Creation**: We generate a specialized denoising dataset that introduces noise into the training data on-the-fly, preparing it for denoising tasks.
4. **Dataloader Configuration**: The DataLoader batches the data, allowing for efficient training through shuffled mini-batches.
5. **Loss Function Setup**: A denoising autoencoder loss is utilized, specifically designed to train the model by reconstructing the original data from its noised version.
6. **Model Training**: The model is trained for a set number of epochs, using specific training parameters such as learning rate and weight decay, with real-time progress displayed.
7. **Model Saving**: After training, the model is saved for later use, encapsulating all learned parameters and configurations.

This training regimen effectively teaches the model to denoise text data, enhancing its capability for downstream tasks like sentence embedding and document similarity assessment.

### Finetuning SBERT with MNR

This section covers the fine-tuning of the Sentence-BERT (SBERT) model using Multiple Negatives Ranking (MNR) loss. The process includes:

- **Data Preparation**: Loading the dataset and creating document-query pairs for training.
- **Creating Training Samples**: Transforming document-query pairs into `InputExample` objects and compiling them into a training dataset.
- **Model and DataLoader Configuration**: Initializing the BERT transformer model, adding a pooling layer for sentence embeddings, and setting up a data loader for efficient batching.
- **Training the Model**: Fine-tuning the model with MNR loss to improve its ranking capability, setting training parameters, and saving the fine-tuned model for future use.