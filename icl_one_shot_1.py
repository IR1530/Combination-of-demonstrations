import random
import numpy as np
import torch
import os
import json
import copy
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.neighbors import KDTree

def load_model_tokenizer(model_name, single_precision):
    if model_name == 'meta-llama/llama-2-7b-hf':
        if not single_precision:
            model = LlamaForCausalLM.from_pretrained(model_name, cache_dir="/scratch/ishan/hf_cache/")
        else:
            model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir="/scratch/ishan/hf_cache/")
        tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

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

class SST2Processor():
    def __init__(self, classes_in_data):
        self.labels = classes_in_data
        self.label_mapping = {k: i for (i, k) in enumerate(self.labels)}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[0:]):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
                label = linelist[1]
                guid = "%s-%s" % (split, idx)
                example = InputExample(guid=guid, text_a=text_a, label=self.label_mapping[label])
                examples.append(example)
        return examples

def load_dataset(dataset, train_datapath, test_datapath, classes_in_data, tokenizer):
    dataset_dict = dict()
    if dataset == 'sst2':
        dataset_dict['train'] = SST2Processor(classes_in_data).get_examples(train_datapath, 'train')
        dataset_dict['test'] = SST2Processor(classes_in_data).get_examples(test_datapath, 'test')
    print("Length of train set: ", len(dataset_dict['train']))
    print("Length of test set", len(dataset_dict['test']))
    print("Train example at 0th index: ", dataset_dict['train'][0])
    return dataset_dict

def get_kdtree(dataset):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    train_sentences = [ex.text_a for ex in dataset['train']]
    train_embeddings = sbert_model.encode(train_sentences)
    tree = KDTree(train_embeddings)
    return sbert_model, tree

def get_top_k_examples(test_example, sbert_model, tree, dataset_dict, k=5):
    test_embedding = sbert_model.encode([test_example.text_a])
    _, top_k_indices = tree.query(test_embedding, k=k)
    top_k_examples = [dataset_dict['train'][idx] for idx in top_k_indices[0]]
    return top_k_examples

def pred_batch(splt, prompt_prefix, prompt_suffix, tokenizer, indexes, dataset_dict, sbert_model, tree, max_rem_len, prompts, batch_size, model, class_idx, classes, device):
    test_examples = dataset_dict[splt][indexes]
    
    all_preds = []
    all_labels = []
    
    for test_example in test_examples:
        top_k_examples = get_top_k_examples(test_example, sbert_model, tree, dataset_dict, k=5)
        
        for one_shot_example in top_k_examples:
            prompt = f'{prompt_prefix}Review: {one_shot_example.text_a}\nSentiment: {classes[one_shot_example.label]}\n\nReview: {test_example.text_a}\n'
            enc = tokenizer.encode_plus(prompt, return_tensors='pt', padding='longest')
            
            for key, enc_value in list(enc.items()):
                enc_value = enc_value[:, :max_rem_len]
                enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1)
            
            seq_len = enc['input_ids'].shape[1]
            enc = {ky: v.to(device) for ky, v in enc.items()}
            
            with torch.no_grad():
                result = model(**enc).logits
            
            result = result[:, -1, class_idx]
            result = F.softmax(result, dim=1)
            labels = test_example.label
            preds = torch.argmax(result, dim=-1)
            confidence = result[0][labels].item()
            
            all_labels.append(labels)
            all_preds.append(preds.cpu().item())
    
    return seq_len, all_labels, all_preds, confidence

def main():
    # Configurations
    dataset = 'sst2'
    K_max = 1  # For one-shot prediction
    model_name = 'meta-llama/llama-2-7b-hf'
    single_precision = True
    gpu_id = 0
    train_datapath = 'data/SST2'
    test_datapath = 'data/SST2'
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
    dataset_dict = load_dataset(dataset, train_datapath, test_datapath, classes_in_data, tokenizer)

    # Model for getting similar demonstrations
    sbert_model, tree = get_kdtree(dataset_dict)

    # Prepare prompts
    prompts = tokenizer.batch_encode_plus([prompt_suffix for _ in range(batch_size)], return_tensors='pt', padding='longest', add_special_tokens=False)
    max_rem_len = model.config.max_position_embeddings - prompts['input_ids'].shape[1]

    all_preds = list()
    all_labels = list()
    seq_lens = list()
    for start_idx in tqdm(range(0, len(dataset_dict['test']), batch_size)):
        indexes = slice(start_idx, start_idx + batch_size)
        seq_len, labels, preds, confidence = pred_batch('test', prompt_prefix, prompt_suffix, tokenizer, indexes, dataset_dict, sbert_model, tree, max_rem_len, prompts, batch_size, model, class_idx, classes, device)
        seq_lens.append(seq_len)
        all_preds.extend(preds)
        all_labels.extend(labels)
    seq_lens = np.array(seq_lens)
    print("Mean Sequence length: ", seq_lens.mean())
    print("Min Sequence length: ", seq_lens.min())
    print("Max Sequence length: ", seq_lens.max())
    print("95th percentile: ", np.percentile(seq_lens, 95))
    print("99th percentile: ", np.percentile(seq_lens, 99))
    print("99.9th percentile: ", np.percentile(seq_lens, 99.9))

    report = classification_report(all_labels, all_preds, digits=4)
    print('Classification Report:')
    print(report)

if __name__ == "__main__":
    main()
