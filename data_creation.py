import random
import numpy as np
import torch
import json
import os
import copy
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.neighbors import KDTree

# Function to load the model and tokenizer
def load_model_tokenizer(model_name, single_precision):
    if model_name == 'meta-llama/llama-2-7b-hf':
        if not single_precision:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

# Class to represent input examples
class InputExample(object):
    def __init__(self, guid=None, text_a="", text_b="", label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

# Class to process SST2 data
class SST2Processor():
    def __init__(self, classes_in_data):
        self.labels = classes_in_data
        self.label_mapping = {k: i for (i, k) in enumerate(self.labels)}

    def get_examples(self, file_path):
        examples = []
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
                label = linelist[1]
                guid = f"{file_path}-{idx}"
                example = InputExample(guid=guid, text_a=text_a, label=self.label_mapping[label])
                examples.append(example)
        return examples

# Function to load dataset
def load_dataset(train_path, test_path, classes_in_data):
    dataset_dict = dict()
    processor = SST2Processor(classes_in_data)
    dataset_dict['train'] = processor.get_examples(train_path)
    dataset_dict['test'] = processor.get_examples(test_path)
    print("Length of train set: ", len(dataset_dict['train']))
    print("Length of test set", len(dataset_dict['test']))
    print("Train example at 0th index: ", dataset_dict['train'][0])
    return dataset_dict

# Function to create KDTree for nearest neighbor search
def get_kdtree(dataset):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    train_sentences = [ex.text_a for ex in dataset['train']]
    train_embeddings = sbert_model.encode(train_sentences)
    tree = KDTree(train_embeddings)
    return sbert_model, tree

# Function to get top k examples using KDTree
def get_top_k_examples(test_example, sbert_model, tree, dataset_dict, k=10):
    test_embedding = sbert_model.encode([test_example.text_a])
    _, top_k_indices = tree.query(test_embedding, k=k)
    top_k_examples = [dataset_dict['train'][idx] for idx in top_k_indices[0]]
    return top_k_examples

# Function to perform one-shot prediction
def one_shot_prediction(test_example, one_shot_example, prompt_prefix, prompt_suffix, tokenizer, max_rem_len, prompts, model, class_idx, classes, device):
    # Construct the prompt
    prompt = f'{prompt_prefix}Review: {one_shot_example.text_a}\nSentiment: {classes[one_shot_example.label]}\n\nReview: {test_example.text_a}\n'
    enc = tokenizer.encode_plus(prompt, return_tensors='pt', padding='longest')
    
    # Truncate and pad the input
    for key, enc_value in list(enc.items()):
        enc_value = enc_value[:, :max_rem_len]
        enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1)
    
    seq_len = enc['input_ids'].shape[1]
    enc = {ky: v.to(device) for ky, v in enc.items()}
    
    # Get model prediction
    with torch.no_grad():
        result = model(**enc).logits
    
    result = result[:, -1, class_idx]
    result = F.softmax(result, dim=1)
    preds = torch.argmax(result, dim=-1)
    confidence = result[0][preds].item()
    
    return seq_len, test_example.label, preds.cpu().item(), confidence

# Function to perform two-shot prediction
def two_shot_prediction(test_example, one_shot_example1, one_shot_example2, prompt_prefix, prompt_suffix, tokenizer, max_rem_len, prompts, model, class_idx, classes, device):
    # Construct the prompt
    prompt = f'{prompt_prefix}Review: {one_shot_example1.text_a}\nSentiment: {classes[one_shot_example1.label]}\n\n'
    prompt += f'Review: {one_shot_example2.text_a}\nSentiment: {classes[one_shot_example2.label]}\n\n'
    prompt += f'Review: {test_example.text_a}\n'
    
    enc = tokenizer.encode_plus(prompt, return_tensors='pt', padding='longest')
    
    # Truncate and pad the input
    for key, enc_value in list(enc.items()):
        enc_value = enc_value[:, :max_rem_len]
        enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1)
    
    seq_len = enc['input_ids'].shape[1]
    enc = {ky: v.to(device) for ky, v in enc.items()}
    
    # Get model prediction
    with torch.no_grad():
        result = model(**enc).logits
    
    result = result[:, -1, class_idx]
    result = F.softmax(result, dim=1)
    preds = torch.argmax(result, dim=-1)
    confidence = result[0][preds].item()
    
    return seq_len, test_example.label, preds.cpu().item(), confidence

# Function to process a batch of predictions
def pred_batch(splt, prompt_prefix, prompt_suffix, tokenizer, indexes, dataset_dict, sbert_model, tree, max_rem_len, prompts, batch_size, model, class_idx, classes, device):
    test_examples = [dataset_dict[splt][i] for i in indexes if i < len(dataset_dict[splt])]

    all_preds_one_shot = []
    all_labels_one_shot = []
    results_one_shot = []
    seq_lens_one_shot = []

    all_preds_two_shot = []
    all_labels_two_shot = []
    results_two_shot = []
    seq_lens_two_shot = []

    for test_example in test_examples:
        # Get top k similar examples
        top_k_examples = get_top_k_examples(test_example, sbert_model, tree, dataset_dict, k=10)
        
        # One-shot predictions
        for one_shot_example in top_k_examples:
            seq_len, labels, preds, confidence = one_shot_prediction(
                test_example, one_shot_example, prompt_prefix, prompt_suffix, tokenizer, max_rem_len, prompts, model, class_idx, classes, device
            )
            all_labels_one_shot.append(labels)
            all_preds_one_shot.append(preds)
            seq_lens_one_shot.append(seq_len)

            results_one_shot.append({
                'test_instance': test_example.text_a,
                'one_shot_example': one_shot_example.text_a,
                'prediction': preds,
                'correct': 1 if preds == labels else 0,
                'predicted_probability': confidence,
                'sequence_length': seq_len
            })
        
        # Two-shot predictions
        for i in range(len(top_k_examples)):
            for j in range(i+1, len(top_k_examples)):
                one_shot_example1 = top_k_examples[i]
                one_shot_example2 = top_k_examples[j]
                seq_len, labels, preds, confidence = two_shot_prediction(
                    test_example, one_shot_example1, one_shot_example2, prompt_prefix, prompt_suffix, tokenizer, max_rem_len, prompts, model, class_idx, classes, device
                )
                all_labels_two_shot.append(labels)
                all_preds_two_shot.append(preds)
                seq_lens_two_shot.append(seq_len)

                results_two_shot.append({
                    'test_instance': test_example.text_a,
                    'one_shot_example1': one_shot_example1.text_a,
                    'one_shot_example2': one_shot_example2.text_a,
                    'prediction': preds,
                    'correct': 1 if preds == labels else 0,
                    'predicted_probability': confidence,
                    'sequence_length': seq_len
                })

    return all_labels_one_shot, all_preds_one_shot, results_one_shot, all_labels_two_shot, all_preds_two_shot, results_two_shot, seq_lens_one_shot, seq_lens_two_shot

# Function to calculate compatibility
def calculate_compatibility(one_shot_results, two_shot_results):
    compatibility_results = []
    for two_shot in two_shot_results:
        test_instance = two_shot['test_instance']
        example1 = two_shot['one_shot_example1']
        example2 = two_shot['one_shot_example2']
        
        # Find individual one-shot predictions
        pred1 = next(r['correct'] for r in one_shot_results if r['test_instance'] == test_instance and r['one_shot_example'] == example1)
        pred2 = next(r['correct'] for r in one_shot_results if r['test_instance'] == test_instance and r['one_shot_example'] == example2)
        
        # Calculate compatibility
        compatibility = 1 if pred1 == 1 and pred2 == 1 and two_shot['correct'] == 1 else 0
        
        compatibility_results.append({
            'test_instance': test_instance,
            'example1': example1,
            'example2': example2,
            'pred1': pred1,
            'pred2': pred2,
            'pair_pred_AB': two_shot['correct'],
            'compatibility': compatibility
        })
    
    return compatibility_results

# Function to generate triplets with multiple negatives
def generate_triplets_with_multiple_negatives(compatibility_results, num_negatives=5):
    triplets = []
    
    # Group compatibility results by test instance
    grouped_results = {}
    all_examples = set()
    for comp in compatibility_results:
        if comp['test_instance'] not in grouped_results:
            grouped_results[comp['test_instance']] = {'compatible': [], 'incompatible': []}
        
        if comp['compatibility'] == 1:
            grouped_results[comp['test_instance']]['compatible'].append((comp['example1'], comp['example2']))
        else:
            grouped_results[comp['test_instance']]['incompatible'].append((comp['example1'], comp['example2']))
        
        all_examples.add(comp['example1'])
        all_examples.add(comp['example2'])
    
    all_examples = list(all_examples)

    for test_instance, results in grouped_results.items():
        for positive in results['compatible']:
            negatives = []
            
            # First, try to get negatives from incompatible examples
            negatives.extend(random.sample(results['incompatible'], min(num_negatives, len(results['incompatible']))))
            
            # If we don't have enough, add random examples from other queries
            while len(negatives) < num_negatives:
                random_example1 = random.choice(all_examples)
                random_example2 = random.choice(all_examples)
                if (random_example1, random_example2) not in results['compatible'] and (random_example1, random_example2) not in negatives:
                    negatives.append((random_example1, random_example2))
            
            triplets.append({
                'anchor': test_instance,
                'positive': positive,
                'negatives': negatives
            })
    
    return triplets

def main():
    # Configurations
    dataset = 'sst2'
    model_name = 'meta-llama/llama-2-7b-hf'
    single_precision = True
    gpu_id = 0
    train_datapath = 'train.tsv'
    test_datapath = 'test.tsv'
    classes = ['negative', 'positive']
    classes_in_data = ['0', '1']
    prompt_prefix = 'Your task is to judge whether the sentiment of a movie review is positive or negative.\n'
    prompt_suffix = 'Sentiment: '
    batch_size = 10
    
    # Seeds and device setup
    random.seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.set_device(gpu_id)
    device = torch.device('cuda:'+str(gpu_id) if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(model_name, single_precision)
    model.to(device)
    model.eval()

    # Get indices of class label words in the vocab
    class_idx = tuple([tokenizer.encode(clas, add_special_tokens=False)[0] for clas in classes])

    # Load dataset
    dataset_dict = load_dataset(train_datapath, test_datapath, classes_in_data)

    # Model for getting similar demonstrations
    sbert_model, tree = get_kdtree(dataset_dict)

    # Prepare prompts
    prompts = tokenizer.batch_encode_plus([prompt_suffix for _ in range(batch_size)], return_tensors='pt', padding='longest', add_special_tokens=False)
    max_rem_len = model.config.max_position_embeddings - prompts['input_ids'].shape[1]

    all_preds_one_shot = list()
    all_labels_one_shot = list()
    all_results_one_shot = list()
    all_seq_lens_one_shot = list()

    all_preds_two_shot = list()
    all_labels_two_shot = list()
    all_results_two_shot = list()
    all_seq_lens_two_shot = list()

    num_examples = len(dataset_dict['test'])
    for start_idx in tqdm(range(0, num_examples, batch_size)):
        end_idx = min(start_idx + batch_size, num_examples)
        indexes = range(start_idx, end_idx)
        labels_one_shot, preds_one_shot, results_one_shot, labels_two_shot, preds_two_shot, results_two_shot, seq_lens_one_shot, seq_lens_two_shot = pred_batch(
            'test', prompt_prefix, prompt_suffix, tokenizer, indexes, dataset_dict, sbert_model, tree, max_rem_len, prompts, batch_size, model, class_idx, classes, device
        )
        
        all_preds_one_shot.extend(preds_one_shot)
        all_labels_one_shot.extend(labels_one_shot)
        all_results_one_shot.extend(results_one_shot)
        all_seq_lens_one_shot.extend(seq_lens_one_shot)

        all_preds_two_shot.extend(preds_two_shot)
        all_labels_two_shot.extend(labels_two_shot)
        all_results_two_shot.extend(results_two_shot)
        all_seq_lens_two_shot.extend(seq_lens_two_shot)

    # One-shot results
    report_one_shot = classification_report(all_labels_one_shot, all_preds_one_shot, digits=4)
    print('One-shot Classification Report:')
    print(report_one_shot)

    results_df_one_shot = pd.DataFrame(all_results_one_shot)
    results_df_one_shot.to_csv('one_shot_results.csv', index=False)

    # Two-shot results
    report_two_shot = classification_report(all_labels_two_shot, all_preds_two_shot, digits=4)
    print('Two-shot Classification Report:')
    print(report_two_shot)

    results_df_two_shot = pd.DataFrame(all_results_two_shot)
    results_df_two_shot.to_csv('two_shot_results.csv', index=False)

    # Calculate sequence length statistics
    all_seq_lens = all_seq_lens_one_shot + all_seq_lens_two_shot
    max_seq_len = max(all_seq_lens)
    avg_seq_len = sum(all_seq_lens) / len(all_seq_lens)
    percentage_greater_than_avg = sum(1 for x in all_seq_lens if x > avg_seq_len) / len(all_seq_lens) * 100

    print(f"Max sequence length: {max_seq_len}")
    print(f"Average sequence length: {avg_seq_len:.2f}")
    print(f"Percentage of sequences greater than average length: {percentage_greater_than_avg:.2f}%")

    # Calculate compatibility
    compatibility_results = calculate_compatibility(all_results_one_shot, all_results_two_shot)

    # Save compatibility results to a CSV file
    compatibility_df = pd.DataFrame(compatibility_results)
    compatibility_df.to_csv('compatibility_results.csv', index=False)
    triplets = generate_triplets_with_multiple_negatives(compatibility_results, num_negatives=5)

    print(f"Generated {len(triplets)} triplets")

    # Save triplets to a JSON file
    with open('triplets.json', 'w') as f:
        json.dump(triplets, f)

    # Print a sample triplet
    if triplets:
        print("Sample triplet:")
        print(json.dumps(triplets[0], indent=2))
    else:
        print("No triplets generated.")

if __name__ == "__main__":
    main()