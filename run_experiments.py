import argparse
from pydoc import locate
import os
import yaml

from sklearn.metrics import f1_score, accuracy_score
from typing import List, Tuple
from tqdm import tqdm

from data.datasets import DataReader, Example
from models.baseline_models import BaselineModel
from models.test_interface import ModelInterface

parser = argparse.ArgumentParser()
parser.add_argument("-config", help="Location of experiments config file.")
parser.add_argument("-output_loc", help="Folder where results will be dumped")
parser.add_argument("--include_baseline", help="Flag to run cosine baseline.",
                    action='store_true')

MSR_LOC = "data/msr-paraphrase/msr-para-test.tsv"
KGAP_LOC = "data/kgap/kgap_semantic_sim.csv"


class ExperimentRun(object):
    def __init__(self,
                 name: str,
                 dataset_name: str,
                 model: ModelInterface,
                 params: dict = None):
        self.name = name
        self.dataset_name = dataset_name
        self.model = model
        self.params = params
        self.predictions = list()
        self.accuracy = 0.0
        self.f1_score = 0.0

    def generate_predictions(self, data: List[Example]):
        for example in tqdm(data):
            self.predictions.append(self.model.predict(example.text1,
                                                       example.text2)[0])

    def evaluate_predictions(self, gold_labels: List[int]):
        self.accuracy = accuracy_score(gold_labels, self.predictions)
        self.f1_score = f1_score(gold_labels, self.predictions)


def main():
    args = parser.parse_args()

    print("Reading in datasets ....")
    # 1. Load Datasets
    msr_data = DataReader("msr-paraphrase")
    msr_data.load_data(MSR_LOC)

    kgap_data = DataReader("kgap")
    kgap_data.load_data(KGAP_LOC)

    print("Successfully read in datasets")

    print("Running baseline experiments ...")
    experiment_results = list()
    # [Optional] Run baseline against both datasets
    if args.include_baseline:
        print("runnning experiments against msr-paraphrase")
        model = BaselineModel()
        msr_base = ExperimentRun(name="cosine baseline",
                                 dataset_name="msr-paraphrase",
                                 model=model)
        msr_base.generate_predictions(msr_data.data)
        msr_base.evaluate_predictions(msr_data.gold_labels)
        experiment_results.append(msr_base)
        print(f"msr: accuracy-{msr_base.accuracy}, f1-score-{msr_base.f1_score}")


        print("running experiments against kgap")
        kgap_base = ExperimentRun(name="cosine baseline",
                                  dataset_name="kgap",
                                  model=model)
        kgap_base.generate_predictions(kgap_data.data)
        kgap_base.evaluate_predictions(kgap_data.gold_labels)
        experiment_results.append(kgap_base)
        print(f"kgap: accuracy-{kgap_base.accuracy}, f1-score-{kgap_base.f1_score}")

    print("Running experiments")
    # 2. Load experiments from config
    experiments = yaml.safe_load(open(args.config, 'r'))

    for exp in experiments["experiments"]:

        try:
            params = exp["params"]

            model = locate(exp["model"])(**params)
            experiment = ExperimentRun(name=exp["name"],
                                       dataset_name=exp["dataset"],
                                       model=model)
            data = msr_data.data if exp["dataset"] == "msr-paraphrase" \
                else kgap_data.data
            gold_labels = msr_data.gold_labels if exp["dataset"] == "msr-paraphrase" \
                else kgap_data.gold_labels

            experiment.generate_predictions(data)
            experiment.evaluate_predictions(gold_labels)

            experiment_results.append(experiment)
        except:
            continue

    # 3. Write results to file
    output_loc = args.output_loc if args.output_loc[-1] == "/" \
        else args.output_loc + "/"

    if not os.path.exists(output_loc):
        os.mkdir(output_loc)

    with open(output_loc+"experiment_results.csv", "w") as f:
        f.write("Experiment Name,dataset,params,f1_score,accuracy\n")
        for experiment in experiment_results:
            f.write(f"{experiment.name},{experiment.dataset_name},"
                    f"{experiment.params},{round(experiment.f1_score,3)},"
                    f"{round(experiment.accuracy, 3)}" + "\n")
    print(f"Wrote results to: {output_loc}")


if __name__ == "__main__":
    main()
