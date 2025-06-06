import os
import pdb
import sys
import json
import torch
import numpy as np
import warnings
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

warnings.filterwarnings('ignore')

project_path = os.path.abspath('')
if project_path not in sys.path:
    sys.path.append(project_path)
import backend_model_info
from dataloader import DataManager
from model import ModelWiseCNNClassifier, ModelWiseTransformerClassifier, TransformerOnlyClassifier


class SupervisedTrainer:
    def __init__(self, data, model, en_labels, id2label, args):
        self.data = data
        # pdb.set_trace()
        self.model = model
        self.en_labels = en_labels
        self.id2label = id2label

        self.seq_len = args.seq_len
        self.num_train_epochs = args.num_train_epochs
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.warm_up_ratio = args.warm_up_ratio

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model.to(self.device)
        self._create_optimizer_and_scheduler()

    def _create_optimizer_and_scheduler(self):
        num_training_steps = len(
            self.data.train_dataloader) * self.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]

        named_parameters = self.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in named_parameters
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                    self.weight_decay,
            },
            {
                "params": [
                    p for n, p in named_parameters
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                    0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_ratio * num_training_steps,
            num_training_steps=num_training_steps)

    def train(self, ckpt_name='CNN_linear_en.pt'):
        for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_steps = 0
            # train
            for step, inputs in enumerate(
                    tqdm(self.data.train_dataloader, desc="Iteration")):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.set_grad_enabled(True):
                    labels = inputs['labels']
                    # pdb.set_trace()
                    output = self.model(inputs['features'], inputs['labels'])

                    logits = output['logits']
                    loss = output['loss']
                    # print(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            # pdb.set_trace()
            print(f'epoch {epoch + 1}: train_loss {loss}')
            # test
            # pdb.set_trace()
            self.test()
            print('*' * 120)
            torch.save(self.model.cpu(), ckpt_name)
            self.model.to(self.device)

        torch.save(self.model.cpu(), ckpt_name)
        saved_model = torch.load(ckpt_name)
        self.model.load_state_dict(saved_model.state_dict())
        return

    def test(self):
        self.model.eval()
        texts = []
        true_labels = []
        pred_labels = []
        total_logits = []
        for step, inputs in enumerate(
                tqdm(self.data.test_dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                labels = inputs['labels']
                output = self.model(inputs['features'], inputs['labels'])
                logits = output['logits']
                preds = output['preds']

                texts.extend(inputs['text'])
                pred_labels.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
                total_logits.extend(logits.cpu().tolist())


        print("*" * 8, "Evalation", "*" * 8)
        result = self.text_level_eval(texts, true_labels, pred_labels)

    def text_level_eval(self, texts, true_labels, pred_labels):
        """
        """
        true_text_labels = []
        pred_text_labels = []

        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_text_label = self.get_text_label(text, true_label)
            pred_text_label = self.get_text_label(text, pred_label)
            true_text_labels.extend(true_text_label)
            pred_text_labels.extend(pred_text_label)

        true_text_labels = [self.en_labels[label] for label in true_text_labels]
        pred_text_labels = [self.en_labels[label] for label in pred_text_labels]
        result = self._get_precision_recall_acc_macrof1(true_text_labels, pred_text_labels)
        return result

    def get_text_label(self, text, label):
        import nltk
        sent_separator = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_separator.tokenize(text)
        # pdb.set_trace()
        offset = 0
        sent_label = []
        for sent in sents:
            start = text[offset:].find(sent) + offset
            end = start + len(sent)
            offset = end

            split_sentence = self.data.split_sentence
            end_word_idx = len(split_sentence(text[:end]))
            if end_word_idx > self.seq_len:
                break
            word_num = len(split_sentence(text[start:end]))
            start_word_idx = end_word_idx - word_num
            tags = label[start_word_idx:end_word_idx]
            most_common_tag = self._get_most_common_tag(tags)
            sent_label.append(most_common_tag[0])
        # pdb.set_trace()

        return sent_label

    def _get_most_common_tag(self, tags):
        """most_common_tag is a tuple: (tag, times)"""
        from collections import Counter

        tags = [self.id2label[tag] for tag in tags]
        tags = [tag.split('-')[-1] for tag in tags]
        tag_counts = Counter(tags)
        most_common_tag = tag_counts.most_common(1)[0]

        return most_common_tag

    def _get_precision_recall_acc_macrof1(self, true_labels, pred_labels):
        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        print("Accuracy: {:.1f}".format(accuracy * 100))
        print("Macro F1 Score: {:.1f}".format(macro_f1 * 100))

        precision = precision_score(true_labels, pred_labels, average=None)
        recall = recall_score(true_labels, pred_labels, average=None)
        print("Precision/Recall per class: ")
        precision_recall = ' '.join(["{:.1f}/{:.1f}".format(p * 100, r * 100) for p, r in zip(precision, recall)])
        print(precision_recall)

        result = {"precision": precision, "recall": recall, "accuracy": accuracy, "macro_f1": macro_f1}
        return result


def construct_bmes_labels(labels):
    prefix = ['B-', 'M-', 'E-', 'S-']
    id2label = {}
    counter = 0

    for label, id in labels.items():
        for pre in prefix:
            id2label[counter] = pre + label
            counter += 1

    return id2label


def split_dataset(data_path, train_path, test_path, train_ratio=0.8):
    file_names = [file_name for file_name in os.listdir(data_path) if file_name.endswith('.jsonl')]
    print('*' * 32)
    print('The overall data sources:')
    print(file_names)
    file_paths = [os.path.join(data_path, file_name) for file_name in file_names]

    total_samples = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            samples = [json.loads(line) for line in f]
            total_samples.extend(samples)

    import random
    random.seed(0)
    random.shuffle(total_samples)

    split_index = int(len(total_samples) * train_ratio)
    train_data = total_samples[:split_index]
    test_data = total_samples[split_index:]

    def save_dataset(fpath, data_samples):
        with open(fpath, 'w', encoding='utf-8') as f:
            for sample in tqdm(data_samples):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    save_dataset(train_path, train_data)
    save_dataset(test_path, test_data)
    print()
    print("The number of train dataset:", len(train_data))
    print("The number of test  dataset:", len(test_data))
    print('*' * 32)
    pass


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Transformer')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--train_mode', type=str, default='classify')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=1024)

    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--split_dataset', action='store_true')
    parser.add_argument('--data_path', type=str, default='/home/wangchi/LIFE/dataset/output')
    parser.add_argument('--train_path', type=str, default='/home/wangchi/LIFE/dataset/train')
    parser.add_argument('--test_path', type=str, default='/home/wangchi/LIFET/dataset/test')

    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)

    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--test_content', action='store_true')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.split_dataset:
        print("Log INFO: split dataset...")
        split_dataset(data_path=args.data_path, train_path=args.train_path, test_path=args.test_path,
                      train_ratio=args.train_ratio)

    en_labels = {
        # 'gpt2_fake': 0,
        # 'gptneo_fake': 1,
        # 'gptj_fake': 2,
        # 'llama_fake': 3,
        # 'human_fake': 4,
        # 'gpt2_true': 5,
        # 'gptneo_true': 6,
        # 'gptj_true': 7,
        # 'llama_true': 8,
        # 'human_true': 9,
        'gpt3.5_fake': 10,
        'gpt3.5_true': 11
    }

    id2label = construct_bmes_labels(en_labels)
    label2id = {v: k for k, v in id2label.items()}

    # pdb.set_trace()
    data = DataManager(train_path=args.train_path, test_path=args.test_path, batch_size=args.batch_size,
                       max_len=args.seq_len, human_label='human', id2label=id2label)

    """linear classify"""
    if args.train_mode == 'classify':
        print('-' * 32 + 'classify' + '-' * 32)
        if args.model == 'CNN':
            print('-' * 32 + "CNN" + '-' * 32)
            classifier = ModelWiseCNNClassifier(id2labels=id2label)
            ckpt_name = 'CNN_linear_en.pt'
        elif args.model == 'RNN':
            print('-' * 32 + "RNN" + '-' * 32)
            classifier = TransformerOnlyClassifier(id2labels=id2label, seq_len=args.seq_len)
            ckpt_name = 'RNN_linear_en.pt'
        else:
            classifier = ModelWiseTransformerClassifier(id2labels=id2label, seq_len=args.seq_len)
            ckpt_name = 'linear_en.pt'

        trainer = SupervisedTrainer(data, classifier, en_labels, id2label, args)

        if args.do_test:
            print("Log INFO: do test...")
            saved_model = torch.load(ckpt_name)
            trainer.model.load_state_dict(saved_model.state_dict())
            trainer.test(content_level_eval=args.test_content)
        else:
            print("Log INFO: do train...")
            trainer.train(ckpt_name=ckpt_name)
