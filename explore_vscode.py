
#%%
with open("data/msr/msr-para-train.tsv", "r") as f:
    lines = f.readlines()


#%%
lines[0].split("\t")


#%%
from typing import List
class MSRexample(object):
    
    def __init__(self, line: List[str]):
        self.label = int(line[0].strip())
        self.text1_id = line[1].strip()
        self.text2_id = line[2].strip()
        self.text1 = line[3].strip()
        self.text2 = line[4].strip()
        


#%%
dataset = []
for line in lines[1:]:
    l = line.split('\t')
    if len(l) != 5:
        continue
    
    dataset.append(MSRexample(l))
    


#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
import spacy 


nlp = spacy.load('en')

a = "That cat is cool."
b = "The cats are cool."

def strip_lemmatize(text: str) -> str:
    text = text if text[-1] not in punctuation else text[:-1]
    lemmas = " ".join([tok.lemma_ for tok in nlp(text) if tok.text != "-PRON-"])
    return lemmas

def simple_baseline(text1: str, text2: str, threshold = .85) -> bool:
    text1 = strip_lemmatize(text1)
    text2 = strip_lemmatize(text2)
    
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf = tfidf.fit_transform([text1,text2])
    sim_score = (tfidf * tfidf.T).toarray()[0,1]
    return sim_score > threshold, sim_score

simple_baseline(a,b)


#%%
baseline_res = []
for ex in dataset:
    res, score = simple_baseline(ex.text1, ex.text2)
    
    baseline_res.append(int(res) == ex.label)


#%%
import numpy as np

np.mean(baseline_res)


#%%
from InferSent.models import InferSent
import torch 

model_version = 2
MODEL_PATH = "InferSent/encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))


#%%
use_cuda = False
model = model.cuda() if use_cuda else model

W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'InferSent/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)


#%%
model.build_vocab_k_words(1000)


#%%
from sklearn.metrics.pairwise import paired_distances

def infersent_sim(text1, text2, model, threshold = .85):
    
    text1_embed = model.encode([a])
    text2_embed = model.encode([b])
    score = 1 - paired_distances(text1_embed, text2_embed, metric="cosine")[0]
    
    return score > threshold, score


#%%
infersent_res = []
for ex in dataset:
    res, score = infersent_sim(ex.text1, ex.text2, model)
    
    infersent_res.append(int(res) == ex.label)
    
np.mean(infersent_res)


#%%



#%%



