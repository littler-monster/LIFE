import os
import json
import pdb
# 文件夹路径
folder_path = ''  # 放置原新闻的文件夹
important_sentences_file = 'keySentence/MF/important_sentences_top20.jsonl'  # 重要句子的 JSONL 文件路径

important_sentences = {}
with open(important_sentences_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        key = (data['id'], data['label'])  # 使用 (id, label) 作为组合键
        important_sentences[key] = data['sentence']

# 遍历文件夹中的所有 .jsonl 文件，更新每个文件
for filename in os.listdir(folder_path):
    if filename.endswith('.jsonl'):  # 确保是 JSONL 文件
        file_path = os.path.join(folder_path, filename)

        # 存储更新后的数据
        updated_lines = []

        # 逐行读取 JSONL 文件内容
        with open(file_path, 'r') as f:
            for line in f:
                file_data = json.loads(line)

                # 获取文件中的 id 和 label
                file_id = file_data.get('id')
                file_label = file_data.get('label')

                # 根据 id 和 label 组合键查找对应的句子，并添加到文件中
                if file_id and file_label:
                    key = (file_id, file_label)
                    if key in important_sentences:
                        file_data['sentence'] = important_sentences[key]

                # 将更新后的行添加到更新列表
                updated_lines.append(file_data)

        # 将更新后的数据写回文件
        with open(file_path, 'w') as f:
            for updated_line in updated_lines:
                f.write(json.dumps(updated_line) + '\n')

print("所有 .jsonl 文件已成功更新。")