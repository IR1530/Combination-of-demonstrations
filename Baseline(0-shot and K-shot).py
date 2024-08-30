import random
import numpy as np
import torch
import os
import json
import copy
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KDTree
from collections import defaultdict


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

class InputExample(object):
    def __init__(self,
                 guid = None,
                 text_a = "",
                 text_b = "",
                 label = None,
                ):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        r"""Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
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
    print("Train example at 0th  index: ", dataset_dict['train'][0])

    return dataset_dict

def get_kdtree(dataset):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load SentenceTransformer model
    train_sentences = [ex.text_a for ex in dataset['train']]  # Get sentence embeddings
    train_embeddings = sbert_model.encode(train_sentences)
    tree = KDTree(train_embeddings)
    return sbert_model, tree

def pred_batch(splt, dataset, prompt_prefix, tokenizer, indexes, dataset_dict, k, max_rem_len, prompts, tree, batch_size, sbert_model, model, class_idx, classes, device):
    if dataset == 'sst2':
        enc = tokenizer.batch_encode_plus(
            [f'{prompt_prefix}Review: {test_example.text_a}\n' for test_example in dataset_dict[splt][indexes]],
            return_tensors='pt', padding='longest')

    if k == 0:
        for key, enc_value in list(enc.items()):
            enc_value = enc_value[:, :max_rem_len]  # truncate any tokens that will not fit once the prompt suffix is added
            enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1)
    else:
        test_sentences = [f'{test_example.text_a}' for test_example in dataset_dict[splt][indexes]]
        test_embeddings = sbert_model.encode(test_sentences)  # Get the embedding of the test_sentence
        _, top_k_indices = tree.query(test_embeddings, k=k)  # find the top k most similar train sentences
        K_examples_all = list()
        for i in top_k_indices:
            K_examples_all.append([dataset_dict['train'][j] for j in i])
        demonstrations = list()
        for K_examples in K_examples_all:
            if dataset == 'sst2':
                demonstrations.append(
                    ''.join([f'Review: {example.text_a}\nSentiment: {classes[example.label]}\n' for example in K_examples]))
        if dataset == 'sst2':
            enc = tokenizer.batch_encode_plus(
                [f'{prompt_prefix}{demonstrations[indx]}Review: {test_example.text_a}\n' for indx, test_example in
                 enumerate(dataset_dict[splt][indexes])], return_tensors='pt', padding='longest')
    for key, enc_value in list(enc.items()):
        enc_value = enc_value[:, :max_rem_len]
        enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1)
    seq_len = enc['input_ids'].shape[1]
    enc = {ky: v.to(device) for ky, v in enc.items()}
    with torch.no_grad():
        result = model(**enc).logits
    result = result[:, -1, class_idx]
    result = F.softmax(result, dim=1)
    labels = [test_example.label for test_example in dataset_dict[splt][indexes]]
    preds = torch.argmax(result, dim=-1)
    confidence = result[0][labels[0]].item()
    return seq_len, labels, preds, confidence

def aggregate_classification_reports(reports, accuracies):
    """
    Aggregates multiple classification reports (dictionaries) by calculating the mean for each metric
    and also calculates the mean accuracy.
    """
    aggregated_report = defaultdict(lambda: defaultdict(list))

    for report in reports:
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    aggregated_report[label][metric].append(value)

    # Calculate the mean for each metric
    mean_report = {label: {metric: np.mean(values) for metric, values in metrics.items()} for label, metrics in aggregated_report.items()}
    mean_accuracy = np.mean(accuracies)
    return mean_report, mean_accuracy

def main():
    # Configs
    k_max_values = [0, 1, 3, 5, 10]
    model_name = 'meta-llama/llama-2-7b-hf'
    single_precision = True  # False #True
    gpu_id = 0
    train_datapath = 'data/'
    test_datapath = 'data/'
    classes = ['negative', 'positive']
    classes_in_data = ['0', '1']
    prompt_suffix = 'Sentiment: '
    batch_size = 10

    # Seeds and device setup
    random.seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.set_device(gpu_id)
    device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(model_name, single_precision)
    model.to(device)
    model.eval()

    # Get indices of class label words in the vocab
    class_idx = tuple([tokenizer.encode(clas, add_special_tokens=False)[0] for clas in classes])

    # Load dataset
    dataset_dict = load_dataset('sst2', train_datapath, test_datapath, classes_in_data, tokenizer)

    # Model for getting similar demonstrations
    sbert_model, tree = get_kdtree(dataset_dict)

    prompts = tokenizer.batch_encode_plus([prompt_suffix for _ in range(batch_size)], return_tensors='pt', padding='longest', add_special_tokens=False)
    max_rem_len = model.config.max_position_embeddings - prompts['input_ids'].shape[1]

    for k_max in k_max_values:
        print(f"Running for k_max = {k_max}")
        all_runs_preds = []
        all_runs_labels = []
        reports = []
        accuracies = []

        if k_max == 0:
            prompt_prefix = 'Your task is to judge whether the sentiment of a movie review is positive or negative.\n'
        else:
            prompt_prefix = 'Your task is to judge whether the sentiment of a movie review is positive or negative. Here are a few examples:\n'

        for run in range(5):  # Run 5 times for each k_max
            print(f"Run {run + 1} for k_max = {k_max}")
            all_preds = list()
            all_labels = list()

            for start_idx in tqdm(range(0, len(dataset_dict['test']), batch_size)):
                indexes = slice(start_idx, start_idx + batch_size)
                seq_len, labels, preds, confidence = pred_batch('test', 'sst2', prompt_prefix, tokenizer, indexes, dataset_dict, k_max, max_rem_len, prompts, tree, batch_size, sbert_model, model, class_idx, classes, device)
                all_preds.extend(preds)
                all_labels.extend(labels)

            # Store predictions and labels for the current run
            all_preds = [pred.item() for pred in all_preds]
            all_runs_preds.append(all_preds)
            all_runs_labels.append(all_labels)

            # Calculate and store the accuracy for this run
            accuracy = accuracy_score(all_labels, all_preds)
            accuracies.append(accuracy)

            # Generate and store the classification report for this run
            report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
            reports.append(report)

        # Calculate the mean classification report and mean accuracy over 5 runs
        mean_report, mean_accuracy = aggregate_classification_reports(reports, accuracies)

        print(f"\nMean Classification Report for k_max = {k_max}:")
        for label, metrics in mean_report.items():
            print(f"{label}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        # Print mean accuracy
        print(f"\nMean Accuracy for k_max = {k_max}: {mean_accuracy:.4f}")
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
