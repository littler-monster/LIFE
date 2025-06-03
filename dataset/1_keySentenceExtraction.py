import os
import json
import pdb

import torch
import nltk
from tqdm import tqdm  # <<== 新增
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
from itertools import combinations

# 配置参数
DATA_DIR = ''  #
MODEL_PATH = 'gpt3.5_bert_model.pt'
BERT_MODEL_NAME = 'bert-base-uncased'


def load_data_from_folder(folder_path: str) -> Tuple[List[Dict], List[Dict]]:
    mf_data = []
    mr_data = []

    for filename in os.listdir(folder_path):
        if not filename.endswith('.jsonl'):
            continue

        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                if filename.split('_')[-1].startswith('fake.jsonl'):
                    mf_data.append(data)
                elif filename.split('_')[-1].startswith('true.jsonl'):
                    mr_data.append(data)

    return mf_data, mr_data


def prepare_texts_and_labels(mf_data: List[Dict], mr_data: List[Dict]) -> Tuple[List[str], List[int]]:
    texts = [item['text'] for item in mf_data] + [item['text'] for item in mr_data]
    labels = [1] * len(mf_data) + [0] * len(mr_data)
    return texts, labels


def tokenize_data(texts: List[str], labels: List[int], tokenizer: BertTokenizer) -> Dataset:
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    return Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })


def train_or_load_model(train_dataset: Dataset) -> BertForSequenceClassification:
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
    else:
        training_args = TrainingArguments(
            output_dir='./results',
            per_device_train_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            report_to="none"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )
        trainer.train()
        torch.save(model.state_dict(), MODEL_PATH)

    return model


def classify_news(text: str, model: BertForSequenceClassification, tokenizer: BertTokenizer,
                  device: torch.device) -> float:
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    return probs[0][1].item()


def split_sentences(text: str) -> List[str]:
    return nltk.tokenize.sent_tokenize(text)


def find_important_sentences(text: str, model: BertForSequenceClassification, tokenizer: BertTokenizer,
                              device: torch.device, top_k: int = 20) -> Tuple[List[str], List[int]]:
    sentences = split_sentences(text)
    original_prob = classify_news(text, model, tokenizer, device)

    impacts = []

    for idx, _ in enumerate(sentences):
        modified_text = ' '.join([s for j, s in enumerate(sentences) if j != idx])
        modified_prob = classify_news(modified_text, model, tokenizer, device)
        change = original_prob - modified_prob
        impacts.append((change, idx))

    # 按照影响从大到小排序，选出 top_k 个索引
    topk_indices = [idx for _, idx in sorted(impacts, reverse=True)[:top_k]]
    topk_indices.sort()

    important_sentences = [sentences[i] for i in topk_indices]
    return important_sentences, topk_indices


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mf_data, mr_data = load_data_from_folder(DATA_DIR)
    texts, labels = prepare_texts_and_labels(mf_data, mr_data)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_texts, _, train_labels, _ = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_dataset = tokenize_data(train_texts, train_labels, tokenizer)

    model = train_or_load_model(train_dataset)
    model.to(device)

    all_data = mf_data + mr_data
    results = []

    # 使用 tqdm 显示进度条
    for entry in tqdm(all_data, desc="Processing news items"):
        # pdb.set_trace()
        news_text = entry['text']
        news_id = entry.get('id', 'unknown_id')
        news_label = entry['label']

        key_sentence, key_index = find_important_sentences(news_text, model, tokenizer, device)

        results.append({
            'id': news_id,
            'sentence': key_sentence,
            'index': key_index,
            'label': news_label
        })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"关键句子提取完成，共处理{len(results)}条新闻，结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
