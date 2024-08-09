import json
import pandas as pd
import tqdm
from torch.utils.data import DataLoader, Dataset
#from datasets import Dataset

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models, util

import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

with open('triplets.json', 'r') as f:
  data = json.load(f)

data = pd.DataFrame(data)
data = data.explode('negatives')
data[['positive1', 'positive2']]  = pd.DataFrame(data['positive'].tolist(), index= data.index)
data[['negative1', 'negative2']] = pd.DataFrame(data['negatives'].tolist(), index= data.index)
data = data.drop('positive', axis=1)
data = data.drop('negatives', axis=1)
data = data.drop_duplicates()
data = data.reset_index(drop=True)

train_batch_size = 64
max_seq_length = 300
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
max_passages = 0
num_epochs = 10
pooling = "mean"
negs_to_use = None
warmup_steps = 1000
lr = 2e-5
num_negs_per_system = 5
use_pre_trained_model = False
use_all_queries = False
ce_score_margin = 3.0
model_save_path = 'output/training_triplets'

model = SentenceTransformer(model_name)
model.max_seq_length = max_seq_length

class MSMARCODataset(Dataset):
    def __init__(self, data, tokenizer):
      self.data = data.reset_index(drop=True)
      self.tokenizer = tokenizer

    def __getitem__(self, item):
      row = self.data.iloc[item]
      query = row.anchor
      pos_text = row.positive1 + self.tokenizer.sep_token + row.positive2
      neg_text = row.negative1 + self.tokenizer.sep_token + row.negative2
      return InputExample(texts=[query, pos_text, neg_text])

    def __len__(self):
        return len(self.data)


train_dataset = MSMARCODataset(data, model.tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Training the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    use_amp=True,
    checkpoint_path=model_save_path,
    checkpoint_save_steps=len(train_dataloader),
    optimizer_params={"lr": lr},
)

# Saving the model
model.save(model_save_path)
