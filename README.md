This is the official code of the paper "Prompt-Induced Linguistic Fingerprints for LLM-Generated Fake
News Detection"

Firstly, enter the dataset folder and run 1_keySentenceExtraction.py to retrieve the key sentences of the news. Next, run 2_concatenate.py to concatenate the key sentences with the original news. Finally, run 3_gen_features-py to obtain the inference probability features of the key sentences.

Then, enter the LIFE_train folder and run train.py

The LLM-generated datasets we used are Politifact++, GossipCop++, and the human-LLM mixed dataset we used is VLFPN. The links are as follows:


https://github.com/mbzuai-nlp/Fakenews-dataset  https://www.dropbox.com/scl/fo/1kf2up2ge0v13izbr7z2e/h?rlkey=xzhm0dbmqevee8f76asz5cyuw&e=1&dl=0
