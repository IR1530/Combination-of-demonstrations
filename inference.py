import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import KDTree
from itertools import combinations
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
from sklearn.metrics import classification_report

# Function to create KDTree from training dataset
def get_kdtree(dataset):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    train_sentences = dataset['review'].tolist()
    train_embeddings = sbert_model.encode(train_sentences)
    tree = KDTree(train_embeddings)
    return sbert_model, tree, train_embeddings

# Function to generate all possible pairs
def generate_pairs(neighbors):
    return list(combinations(neighbors, 2))

# Function to predict compatibility scores
def predict_compatibility(model, query, pair):
    query_embedding = model.encode(query, convert_to_tensor=True)
    pair_embeddings = model.encode(pair, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(query_embedding, pair_embeddings)
    return cos_sim.mean().item()

# Function for multi-shot prediction
def multi_shot_prediction(test_example, examples, sentiments, prompt_prefix, prompt_suffix, tokenizer, model, class_idx, classes):
    prompt = prompt_prefix
    for example, sentiment in zip(examples, sentiments):
        prompt += f'Review: {example}\nSentiment: {sentiment}\n\n'
    prompt += f'Review: {test_example}\n'
    
    enc = tokenizer.encode_plus(prompt, return_tensors='pt', padding='longest')
    enc = {k: v.to('cuda') for k, v in enc.items()}
    
    with torch.no_grad():
        result = model(**enc).logits
    
    result = result[:, -1, class_idx]
    result = F.softmax(result, dim=1)
    pred = torch.argmax(result, dim=-1)
    confidence = result[0][pred].item()
    
    return pred.cpu().item(), confidence

def test_model_our_method(test_data, train_data, sentence_model, llama_model, llama_tokenizer, k=10, num_examples=1):
    results = []
    
    print("Encoding test reviews...")
    test_embeddings = sentence_model.encode(test_data['review'].tolist(), show_progress_bar=True)
    
    print("Creating KDTree for training data...")
    train_sbert_model, kd_tree, train_embeddings = get_kdtree(train_data)
    
    prompt_prefix = 'Your task is to judge whether the sentiment of a movie review is positive or negative.\n'
    prompt_suffix = 'Sentiment: '
    classes = ['negative', 'positive']
    class_idx = tuple([llama_tokenizer.encode(clas, add_special_tokens=False)[0] for clas in classes])
    
    print(f"Processing test examples (using top-{num_examples})...")
    all_preds = []
    all_labels = []
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        query = row['review']
        query_embedding = sentence_model.encode(query)
        _, neighbor_indices = kd_tree.query(query_embedding, k=k)
        neighbors = train_data.iloc[neighbor_indices]['review'].tolist()
        
        pairs = generate_pairs(neighbors)
        scores = []
        for pair in pairs:
            score = predict_compatibility(sentence_model, query, list(pair))
            scores.append((pair, score))
        
        sorted_pairs = sorted(scores, key=lambda x: x[1], reverse=True)
        
        if num_examples == 1:
            # For top-1, find the most similar sentence in the top pair
            top_pair, top_score = sorted_pairs[0]
            similarity_scores = util.pytorch_cos_sim(
                sentence_model.encode(query, convert_to_tensor=True),
                sentence_model.encode(list(top_pair), convert_to_tensor=True)
            )
            most_similar_idx = similarity_scores.argmax().item()
            selected_examples = [top_pair[most_similar_idx]]
        else:
            # For top-5 and top-10, consider pairs and avoid repetition
            selected_examples = []
            used_sentences = set()
            for pair, score in sorted_pairs:
                if len(selected_examples) >= num_examples:
                    break
                new_sentences = [sent for sent in pair if sent not in used_sentences]
                selected_examples.extend(new_sentences)
                used_sentences.update(pair)
            
            # Trim to exact number of examples if we've selected too many
            selected_examples = selected_examples[:num_examples]
        
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
        all_labels.append(int(row['prediction']))
    
    return pd.DataFrame(results), all_preds, all_labels

# Main execution
if __name__ == "__main__":
    # Load the trained SentenceTransformer model
    sentence_model = SentenceTransformer('output/training_triplets')

    # Load the LLaMA-2 model for prediction
    llama_model_name = 'meta-llama/llama-2-7b-hf'
    llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, torch_dtype=torch.float16)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
    llama_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    llama_model.resize_token_embeddings(len(llama_tokenizer))
    llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
    llama_model.to('cuda')
    llama_model.eval()

    # Define the column names
    column_names = ['review', 'prediction']

    # Load the training data and test data with specified column names
    train_data = pd.read_csv('train.tsv', sep='\t', names=column_names)
    test_data = pd.read_csv('test.tsv', sep='\t', names=column_names)

    # Run experiments for different numbers of examples
    for num_examples in [1, 5, 10]:
        print(f"\nRunning our method with top-{num_examples} examples...")
        results_df, all_preds, all_labels = test_model_our_method(
            test_data, train_data, sentence_model, llama_model, llama_tokenizer, k=10, num_examples=num_examples
        )
        
        # Save results
        results_df.to_csv(f'test_results_our_method_top{num_examples}.csv', index=False)
        
        # Generate and print classification report
        print(f"\nClassification Report (Our Method, top-{num_examples}):")
        report = classification_report(all_labels, all_preds, target_names=['negative', 'positive'])
        print(report)
        
        # Calculate and print accuracy
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        print(f"Accuracy (Our Method, top-{num_examples}): {accuracy:.4f}")
        
        # Print some statistics
        print(f"\nOur Method (top-{num_examples}) - Top 5 most compatible examples:")
        print(results_df.nlargest(5, 'compatibility_score')[['query', 'selected_examples', 'compatibility_score', 'prediction', 'confidence']])
        
        print(f"\nOur Method (top-{num_examples}) - Average compatibility score: {results_df['compatibility_score'].mean():.4f}")
        print(f"Our Method (top-{num_examples}) - Average confidence: {results_df['confidence'].mean():.4f}")