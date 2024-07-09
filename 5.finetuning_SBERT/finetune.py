import datasets
dataset = datasets.load_dataset("csv",data_files='./dataset_D_Q.csv', split='train')
from sentence_transformers import InputExample
from tqdm.auto import tqdm  # so we see progress bar

train_samples = []
for row in tqdm(dataset):
    train_samples.append(InputExample(
        texts=[row['document'], row['query']]
    ))

from sentence_transformers import datasets

batch_size = 32

loader = datasets.NoDuplicatesDataLoader(
    train_samples, batch_size=batch_size)

from sentence_transformers import models, SentenceTransformer

bert = models.Transformer('bert-base-uncased')
pooler = models.Pooling(
    bert.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

model = SentenceTransformer(modules=[bert, pooler])

from sentence_transformers import losses

loss = losses.MultipleNegativesRankingLoss(model)



epochs = 10
warmup_steps = int(len(loader) * epochs * 0.1)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path='./sbert_test_mnr2',
    show_progress_bar=True
)

