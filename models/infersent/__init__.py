from typing import List, Tuple
from .models import InferSent
from ..test_interface import ModelInterface

from sklearn.metrics.pairwise import paired_distances
import torch


GLOVE_LOC = "models/infersent/weights/glove.840B.300d.txt"
FASTEXT_LOC = "models/infersent/weights/crawl-300d-2M.vec"
GLOVE_ENCODER = "models/infersent/weights/infersent1.pkl"
FASTEXT_ENCODER = "models/infersent/weights/infersent2.pkl"

MODEL_PARAMS = {'bsize': 64,
                'word_emb_dim': 300,
                'enc_lstm_dim': 2048,
                'pool_type': 'max',
                'dpout_model': 0.0}


class InferSentModel(ModelInterface):

    def __init__(self,
                 embedding_type: str,
                 vocab_size: int = 10000,
                 use_train_sents: bool = False,
                 train_sent_loc: str = None,
                 threshold:float = .85):
        super().__init__("InferSent")
        self.embedding_type = embedding_type
        self.vocab_size = vocab_size
        self.use_train_sents = use_train_sents
        self.threshold = threshold

        if embedding_type == "glove":
            encoder_path = GLOVE_ENCODER
            vec_path = GLOVE_LOC
        else:
            encoder_path = FASTEXT_ENCODER
            vec_path = FASTEXT_LOC

        self.train_sents = list()
        if train_sent_loc is not None:
            with open(train_sent_loc, "r") as f:
                self.train_sents.append(f.read())

        self.model = InferSent(MODEL_PARAMS)
        self.model.load_state_dict(torch.load(encoder_path))
        self.model.set_w2v_path(vec_path)

        # Set to cuda if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Build vocabulary
        if use_train_sents and len(self.train_sents) > 0:
            self.model.build_vocab(self.train_sents, tokenize=True)
        else:
            self.model.build_vocab_k_words(vocab_size)

    def predict_batch(self,
                      inputs: List[Tuple[str, str]]) -> List[Tuple[int, float]]:

        predictions = list()
        for input in inputs:
            predictions.append(self.predict(input[0], input[1]))


    def predict(self,
                text1: str,
                text2: str) -> Tuple[int, float]:
        text1_embed = self.model.encode([text1])
        text2_embed = self.model.encode([text2])

        score = 1 - paired_distances(text1_embed, text2_embed,
                                     metric="cosine")[0]
        return int(score > self.threshold), score

