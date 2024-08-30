import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import KDTree
from itertools import combinations

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
pd.set_option('display.max_rows', 1000)

# Function to create KDTree from training dataset
def get_kdtree(dataset):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    train_sentences = dataset['review'].tolist()
    train_embeddings = sbert_model.encode(train_sentences)
    tree = KDTree(train_embeddings)
    return sbert_model, tree, train_embeddings

# Function to predict compatibility scores
def predict_compatibility(model, query, pair):
    query_embedding = model.encode(query, convert_to_tensor=True)
    pair_embeddings = model.encode(pair, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(query_embedding, pair_embeddings)
    return cos_sim.mean().item()


# Function to generate all possible pairs
def generate_pairs_baseline(sentence_model, query, neighbors):
    pairs = list(combinations(neighbors, 2))
    scores = []
    for pair in pairs:
        score = predict_compatibility(sentence_model, query, list(pair))
        scores.append((pair, score))
    sorted_pairs = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_pairs

def generate_pairs_approach2(query, neighbors, trained_sentence_model_path):
  pairs = list(combinations(neighbors, 2))
  model = SentenceTransformer(trained_sentence_model_path)
  tokenizer = model.tokenizer
  scores = []
  for pair in pairs:
    break
  return None

def get_top_k_examples(num_examples,sentence_model,query,sorted_pairs):

  if num_examples == 1:
    top_pair, top_score = sorted_pairs[0]
    similarity_scores = util.pytorch_cos_sim(
        sentence_model.encode(query, convert_to_tensor=True),
        sentence_model.encode(list(top_pair), convert_to_tensor=True)
    )
    most_similar_idx = similarity_scores.argmax().item()
    selected_examples = [top_pair[most_similar_idx]]
  else:

      selected_examples = []
      used_sentences = set()
      for pair, score in sorted_pairs:
          if len(selected_examples) >= num_examples:
              break
          new_sentences = [sent for sent in pair if sent not in used_sentences]
          selected_examples.extend(new_sentences)
          used_sentences.update(pair)

      selected_examples = selected_examples[:num_examples]
  return selected_examples

def multi_shot_prediction(test_example, examples, sentiments, prompt_prefix, prompt_suffix, tokenizer, model, class_idx, classes):
    prompt = prompt_prefix
    for example, sentiment in zip(examples, sentiments):
        prompt += f'Review: {example}\nSentiment: {sentiment}\n\n'
    prompt += f'Review: {test_example}\n'
    prompt += prompt_suffix
    enc = tokenizer.encode_plus(prompt, return_tensors='pt', padding='longest')
    enc = {k: v.to('cuda') for k, v in enc.items()}
    out = llama_model(**enc).logits[0,-1,class_idx]
    pred = out.argmax().item()

    confidence = out[pred].item()
    return pred, confidence

def baseline(test_data, train_data, sentence_model, llama_model, llama_tokenizer, k=10, num_examples=1):
    results = []

    print("Encoding test reviews...")
    test_embeddings = sentence_model.encode(test_data['review'].tolist(), show_progress_bar=True)

    print("Creating KDTree for training data...")
    train_sbert_model, kd_tree, train_embeddings = get_kdtree(train_data)

    prompt_prefix = 'Your task is to judge whether the sentiment of a movie review is positive or negative.\n'
    prompt_suffix = 'Sentiment: '
    classes = ['negative', 'positive']
    class_idx = tuple([llama_tokenizer.encode(clas, add_special_tokens=False)[0] for clas in ['negative', 'positive']])

    print(f"Processing test examples (using top-{num_examples})...")
    all_preds = []
    all_labels = []
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        query = row['review']
        query_embedding = sentence_model.encode(query)
        _, neighbor_indices = kd_tree.query(query_embedding, k=k)
        neighbors = train_data.iloc[neighbor_indices]['review'].tolist()

        # we generated pairs for every query with baseline approach
        sorted_pairs = generate_pairs_baseline(sentence_model, query, neighbors)
        selected_examples = get_top_k_examples(num_examples,sentence_model,query,sorted_pairs)

        top_score = sorted_pairs[0][1]  # Score of the top pair

        # Get sentiments for selected examples
        sentiments = [train_data.loc[train_data['review'] == ex, 'prediction'].iloc[0] for ex in selected_examples]

        # Perform prediction using the selected examples
        pred, confidence = multi_shot_prediction(
            query,
            selected_examples, sentiments,
            prompt_prefix,
            prompt_suffix,
            llama_tokenizer,
            llama_model,
            class_idx,
            classes
        )

        results.append({
            'query': query,
            'original_prediction': row['prediction'],
            'selected_examples': selected_examples,
            'compatibility_score': top_score,
            'prediction': pred,
            'confidence': confidence
        })

        all_preds.append(pred)
        all_labels.append(row['prediction'])

    return pd.DataFrame(results), all_preds, all_labels

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the LLaMA-2 model for prediction
llama_model_name = 'meta-llama/llama-2-7b-hf'
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, torch_dtype=torch.float16, device_map = "cuda")
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
llama_model.resize_token_embeddings(len(llama_tokenizer))
llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
llama_model.to('cuda')
llama_model.eval()

num_examples = 1

# Define the column names
column_names = ['review', 'prediction']

# Load the training data and test data with specified column names
train_data = pd.read_csv('data/train.tsv', sep='\t', names=column_names)
test_data = pd.read_csv('data/test.tsv', sep='\t', names=column_names)

train_data.prediction = train_data.prediction.map({ 0: "negative", 1: "positive"})
test_data.prediction = test_data.prediction.map({ 0: "negative", 1: "positive"})

results_df, all_preds, all_labels = baseline(
        test_data, train_data, sentence_model, llama_model, llama_tokenizer, k=10, num_examples=num_examples)

label_map = {'positive': 1, "negative": 0}
all_labels = [label_map[label] for label in all_labels]

drop_idx = np.where(np.array(all_preds) == -1)[0]
print("Total Unparsable Predictions = ", len(drop_idx))

all_preds = np.delete(all_preds, drop_idx)
all_labels = np.delete(all_labels, drop_idx)

label_map = {'positive': 1, "negative": 0}
results_df['label'] =  results_df.original_prediction.map(label_map)
results_df['correct'] = results_df['label'] == results_df['prediction']
results_df['correct'] = results_df['correct'].astype(bool)
results_df['example'] = results_df.selected_examples.apply(lambda x: x[0])

def generate_triplets(df):
    triplets = []
    correct_examples = df[df['correct'] == 1]
    incorrect_examples = df[df['correct'] == 0]
    for _, correct_row in correct_examples.iterrows():
        query = correct_row['query']
        positive_text = correct_row['example']
        
        for _, incorrect_row in incorrect_examples.iterrows():
            negative_text = incorrect_row['example']
            
            triplets.append({
                'query': query,
                'positive_text': positive_text,
                'negative_text': negative_text
            })
    
    return pd.DataFrame(triplets)

triplets_df = generate_triplets(results_df)

print(f"Total number of triplets: {len(triplets_df)}")
triplets_df.to_csv('triplets.csv', index=False)
