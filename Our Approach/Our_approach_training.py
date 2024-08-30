import json
import tqdm
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models, util

class config:
  TRIPLETS_PATH = 'paired_triplets.csv'
  TEST_SIZE = 0.2
  TRAIN_BATCH_SIZE = 512
  MAX_SEQ_LENGTH = 300
  MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
  NUM_EPOCHS = 10
  WARMUP_STEPS = 1000
  LEARNING_RATE = 2e-5
  MODEL_SAVE_PATH = 'output/our_approach_training_triplets'

data = pd.read_csv(config.TRIPLETS_PATH)
model = SentenceTransformer(config.MODEL_NAME)
model.max_seq_length = config.MAX_SEQ_LENGTH

class OURAPPROACHDataset(Dataset):
    def __init__(self, data, tokenizer):
      self.data = data.reset_index(drop=True)
      self.tokenizer = tokenizer

    def __getitem__(self, item):
        row = self.data.iloc[item]
        query = row.query
        pos_text = row.positive
        neg_text = row.negative
        return InputExample(texts=[query, pos_text, neg_text])

    def __len__(self):
        return len(self.data)

train_dataset = OURAPPROACHDataset(data, model.tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Training the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=config.NUM_EPOCHS,
    warmup_steps=config.WARMUP_STEPS,
    use_amp=True,
    checkpoint_path=config.MODEL_SAVE_PATH,
    checkpoint_save_steps=len(train_dataloader),
    optimizer_params={"lr": config.LEARNING_RATE},
)

# Saving the model
model.save(config.MODEL_SAVE_PATH)