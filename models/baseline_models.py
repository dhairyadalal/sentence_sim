import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
from string import punctuation
from .test_interface import ModelInterface

nlp = spacy.load("en")


class BaselineModel(ModelInterface):

    def __init__(self, threshold: float = .85):
        super().__init__("baseline_model")
        self.threshold = threshold

    def _strip_lemmatize(self, text: str) -> str:
        text = text if text[-1] not in punctuation else text[:-1]
        lemmas = " ".join(
            [tok.lemma_ for tok in nlp(text) if tok.text != "-PRON-"])
        return lemmas

    def predict(self,
                text1: str,
                text2: str) -> Tuple[int, float]:

        text1 = self._strip_lemmatize(text1)
        text2 = self._strip_lemmatize(text2)

        tfidf = TfidfVectorizer(stop_words="english")
        tfidf = tfidf.fit_transform([text1, text2])
        sim_score = (tfidf * tfidf.T).toarray()[0, 1]
        return int(sim_score > self.threshold), sim_score

    def predict_batch(self,
                      inputs: List[Tuple[str, str]]) -> List[Tuple[int, float]]:

        predictions = list()
        for input in inputs:
            predictions.append(self.predict(input[0], input[1]))

        return predictions


class CannonicalQA(ModelInterface):

    def __init__(self):
        super().__init__("cannonical-qa")

    def predict(self,
                text1: str,
                text2: str) -> Tuple[int, float]:
        pass

    def predict_batch(self,
                      inputs: List[Tuple[str, str]]) -> List[Tuple[int, float]]:
        pass

