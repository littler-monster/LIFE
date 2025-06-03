import pdb
import random
import httpx
import msgpack
import threading
import time
import os
import argparse
import json
import scipy
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm


def access_api(text, api_url, sentence=None, do_generate=False):
    """

    :param text: input text
    :param api_url: api
    :param do_generate: whether generate or not
    :return:
    """
    with httpx.Client(timeout=None) as client:
        post_data = {

            "text": text,
            "sentence": sentence,
            "do_generate": do_generate,
        }
        prediction = client.post(api_url,
                                 data=msgpack.packb(post_data),
                                 timeout=None)
    if prediction.status_code == 200:
        content = msgpack.unpackb(prediction.content)
    else:
        content = None
    return content


def get_features(input_file, output_file):
    """
    get [losses, begin_idx_list, sentence_ll_tokens_list, label_int, label] based on raw lines
    """

    en_model_names = ['gpt_2', 'gpt_neo', 'gpt_J', 'llama']
    cn_model_names = ['wenzhong', 'sky_text', 'damo', 'chatglm']

    gpt_2_api = 'http://0.0.0.0:6006/inference'
    gpt_neo_api = 'http://0.0.0.0:6007/inference'
    gpt_J_api = 'http://0.0.0.0:6008/inference'
    llama_api = 'http://0.0.0.0:6009/inference'

    # en_model_apis = [gpt_2_api, gpt_neo_api, gpt_J_api, llama_api], choose one to use inference
    en_model_apis = [gpt_2_api]

    en_labels = {
        'gpt2_fake': 0,
        'gptneo_fake': 1,
        'gptj_fake': 2,
        'llama_fake': 3,
        'human_fake': 4,
        'gpt2_true': 5,
        'gptneo_true': 6,
        'gptj_true': 7,
        'llama_true': 8,
        'human_true': 9,
        'gpt3.5_fake': 10,
        'gpt3.5_true': 11
    }

    # line = {'text': '', 'label': ''}
    with open(input_file, 'r') as f:
        lines = [json.loads(line) for line in f]

    print('input file:{}, length:{}'.format(input_file, len(lines)))

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in tqdm(lines):
            line = data['text']
            label = data['label']

            sentence = data['sentence']

            losses = []
            begin_idx_list = []
            ll_tokens_list = []
            sentence_ll_tokens_list = []

            model_apis = en_model_apis
            label_dict = en_labels

            label_int = label_dict[label]

            error_flag = False
            for api in model_apis:
                try:
                    loss, begin_word_idx, ll_tokens, sentence_ll = access_api(line, api, sentence)

                except TypeError:
                    print("return NoneType, probably gpu OOM, discard this sample")
                    error_flag = True
                    break
                losses.append(loss)
                begin_idx_list.append(begin_word_idx)
                ll_tokens_list.append(ll_tokens)
                sentence_ll_tokens_list.append(sentence_ll)
            # if oom, discard this sample
            if error_flag:
                continue

            result = {
                'losses': losses,
                'begin_idx_list': begin_idx_list,
                'sentence_ll_tokens_list':sentence_ll_tokens_list,
                'label_int': label_int,
                'label': label,
                'text': line
            }

            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input file")
    parser.add_argument("--output_file", type=str, help="output file")

    parser.add_argument("--get_en_features", action="store_true", help="generate en logits and losses")

    parser.add_argument("--do_normalize", action="store_true", help="normalize the features")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    get_features(input_file=args.input_file, output_file=args.output_file)

