import argparse
from string import punctuation

import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import paired_distances
import spacy 
from typing import List, Tuple

from InferSent.models import InferSent

nlp = spacy.load('en')

class MSRexample(object):
    
    def __init__(self, line: List[str]):
        self.label = line[0].strip()
        self.text1_id = line[1].strip()
        self.text2_id = line[2].strip()
        self.text1 = line[3].strip()
        self.text2 = line[4].strip()

def read_data(data_loc: str) -> List[MSRexample]:
    
    # 1. Read file into memory
    with open(data_loc, "r") as f:
        lines = f.readlines()
    
    # 2. Loop through
    dataset = []
    for line in lines[1:]:
        l = line.split('\t')
        if len(l) != 5:
            continue
        
        dataset.append(MSRexample(l))
    
    return dataset

def strip_lemmatize(text: str) -> str:
    text = text if text[-1] not in punctuation else text[:-1]
    lemmas = " ".join([tok.lemma_ for tok in nlp(text) if tok.text != "-PRON-"])
    return lemmas

def simple_baseline(text1: str, text2: str, threshold = .85) -> Tuple[bool, float]:
    text1 = strip_lemmatize(text1)
    text2 = strip_lemmatize(text2)
    
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf = tfidf.fit_transform([text1,text2])
    sim_score = (tfidf * tfidf.T).toarray()[0,1]
    return sim_score > threshold, sim_score


def load_infersent( model_version: int,
                    vec_path: str, 
                    encoder_path: str, 
                    top_k_vocab: int = 1000, 
                    use_vocab: bool = False,
                    train_sents: List[str] = None) -> object:
   
    # Standard model params
    params_model = {'bsize': 64, 
                    'word_emb_dim': 300,
                    'enc_lstm_dim': 2048,
                    'pool_type': 'max',
                    'dpout_model': 0.0,
                    'version': model_version}

    # Load model
    model = InferSent(params_model)
    model.load_state_dict(torch.load(encoder_path))
    model.set_w2v_path(vec_path)

    # Set to cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Build vocabulary
    if use_vocab == True and train_sents != None:
    	model.build_vocab(train_sents, tokenize=True)
    else:
    	model.build_vocab_k_words(top_k_vocab)

    return model

    
def infersent_sim(text1, text2, model, threshold = .85) -> Tuple[bool, float]:
	text1_embed = model.encode([text1])
	text2_embed = model.encode([text2])

	score = 1 - paired_distances(text1_embed, text2_embed, metric = "cosine")[0]
	return score > threshold, score


def run_baseline_experiment(dataset: List[MSRexample]) -> List[int]:
	baseline_scores  = []
	baseline_results = []

	for ex in dataset:
		res, score = simple_baseline(ex.text1, ex.text2)

		baseline_results.append(int(res))

		baseline_scores.append({"text1": ex.text1,
								"text2": ex.text2,
				 				"score": score, 
				 				"pred": int(res),
				 				"true_label": ex.label})
	pd.DataFrame(baseline_scores).to_csv("cosine_baseline.csv", index=False)

	return baseline_results


def run_infersent_experiment(dataset: List[MSRexample],
							 model_version: int,
							 vec_path: str, 
							 encoder_path: str, 
							 top_k_vocab: int = 1000, 
							 use_vocab: bool = False,
							 train_sents: List[str] = None) -> List[int]:

	# 1. Load model
	model = load_infersent(model_version, vec_path, encoder_path, 
						   top_k_vocab, use_vocab, train_sents)

	# 2. Loop through DataSet and evaluate
	infersent_results = []
	infersent_scores  = []

	for ex in dataset:
		res, score = infersent_sim(ex.text1, ex.text2, model)

		infersent_results.append(int(res))

		infersent_scores.append({"text1": ex.text1,
								 "text2": ex.text2,
				 				 "score": score, 
				 				 "pred": int(res),
				 				 "true_label": ex.label,
				 				 "top_k_vocab": top_k_vocab,
				 				 "use_vocab": use_vocab	})

	pd.DataFrame(infersent_scores).to_csv(f"infersent_vocab_{top_k_vocab}_use_vocab_{use_vocab}.csv", 
										  index=False)
	return infersent_results


def main():
	eval_data_loc = "data/msr/msr-para-test.tsv"
	encoder_loc = "InferSent/encoder/infersent2.pkl"
	vec_loc = "InferSent/fastText/crawl-300d-2M.vec"
	model_version = 2	

	print("loading dataset")
	# 1. Load dataset
	dataset = read_data(eval_data_loc)
	
	y_true = [int(ex.label) for ex in dataset]
	all_sents = []
	for ex in dataset:
		all_sents.append(ex.text1)
		all_sents.append(ex.text2)


	# 2. Run baseline experiment
	print("running baseline ......")
	results = []
	baseline_res = run_baseline_experiment(dataset)
	results.append({"model": "baseline cosine",
					"accuracy": accuracy_score(baseline_res, y_true),
					"f1": f1_score(baseline_res, y_true),
					"top_k_vocab": None,
					"use_vocab": None})
	print("finised running baseline ......")


	print("running infersent experiments")
	# 3. Run Infersent experiments
	experiment_params = [(1000, False, None), 
						 (10000, False, None), 
						 (50000, False, None),
						 (100000, False, None),
						 (500000, False, None), 
						 (None, True, all_sents)]

	for param in experiment_params:
		infersent_results = run_infersent_experiment(dataset = dataset,
													 model_version = model_version,
													 vec_path = vec_loc,
													 encoder_path =  encoder_loc, 
													 top_k_vocab = param[0],
													 use_vocab = param[1],
													 train_sents = param[2])
		results.append({"model": "infersent",
						"accuracy": accuracy_score(infersent_results, y_true),
						"f1": f1_score(infersent_results, y_true),
						 "top_k_vocab": param[0],
						 "use_vocab": param[1]})

	print("finished running experiments")
	res_tbl = pd.DataFrame(results)

	res_tbl.to_csv("sent_sim_results.csv", index = False)
	print(res_tbl)


if __name__ == "__main__":
	main()