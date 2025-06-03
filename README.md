This is the official code of paper "Prompt-Induced Linguistic Fingerprints for LLM-Generated Fake
News Detection"

Firstly, enter the dataset folder and run 1_keySentenceExtraction.py to retrieve the key sentences of the news. Next, run 2_concatenate.py to concatenate the key sentences with the original news. Finally, run 3_gen_features-py to obtain the inference probability features of the key sentences.

Then, enter the LIFE_train folder and run train.py

The LLM generated dataset we use is GossipCop++, and the manual dataset is a LUN. The links are as follows:

https://github.com/mbzuai-nlp/Fakenews-dataset  https://github.com/jiayingwu19/SheepDog/tree/main/data/news_articles