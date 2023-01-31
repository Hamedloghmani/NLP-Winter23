import re
import nltk
import pickle
import pytrec_eval
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import wordnet
from joblib import Parallel, delayed
from nltk.metrics.distance import edit_distance

nltk.download('wordnet')


def create_pairs(spelling_error_file: str) -> list:
    """
    Args:
        spelling_error_file: address of spelling error file

    Returns:
        list of pair as follows: (wrong_spelling, correct_spelling)
    """
    with open(spelling_error_file) as file:
        spellings = file.readlines()
        spellings = list(map(str.lower, map(str.strip, spellings)))
    correct_spelling = ''
    dataset = list()
    for word in spellings:
        if word.startswith('$'):
            correct_spelling = word.replace('$', '')
        else:
            dataset.append((word, correct_spelling))
    return dataset


def process_wordnet() -> list:
    """
    Returns:
        processed version of wordnet
    """
    processed_wordnet = list()
    for word in tqdm(wordnet.all_synsets()):
        processed_wordnet.append(re.sub("\.[a-zA-Z]*\d*", '', word.name()))
    return list(set(processed_wordnet))


def get_similarities(misspelled_pairs: list, processed_wordnet: list, limit: int = 10) -> dict:
    """
    Args:
        misspelled_pairs: list of tuples made by create_pairs function
        processed_wordnet: process_wordnet() output
        limit: cut-off for number of suggestions for each word

    Returns:
        dictionary made of misspelled words as keys, tuple of label and list of suggestions as value
    """
    output_dict = dict()
    for pair in tqdm(misspelled_pairs):
        distances = list()
        for word in processed_wordnet:
            distances.append((word, edit_distance(pair[0], word)))
        distances.sort(key=lambda x: x[1])
        output_dict[pair[0]] = (pair[1], distances[:limit])

    with open('../data/similarity_dict.pkl', 'ab') as file:
        pickle.dump(output_dict, file)

    return output_dict


def parallel(misspelled_pairs: list, word_net: list, limit_: int = 10, cluster_number: int = 100) -> list:
    """
    Args:
        misspelled_pairs: list of tuples made by create_pairs function
        word_net: process_wordnet() output
        limit_: cut-off for number of suggestions for each word
        cluster_number: number of different clusters to parallelize

    Returns:
        a list of results from different clusters
    """
    chunks = np.array_split(np.array(misspelled_pairs), len(misspelled_pairs) / cluster_number)
    results = Parallel(n_jobs=-1, prefer="processes")(delayed(get_similarities)(i, word_net, limit_) for i in chunks)
    return results


def run(spelling_error_file: str, limit: int, cluster_number: int = 100):
    print('Creating pairs from misspellings...')
    misspelled_pairs = create_pairs(spelling_error_file)
    print('Processing wordnet...')
    wordnet_ = process_wordnet()
    parallel(misspelled_pairs, wordnet_, limit, cluster_number)


def building_pytrec_input(similarity_dict: dict) -> tuple:
    """
    Args:
        similarity_dict: output of get_similarities() function
    Returns:
        qrel and run for misspelled words
    """
    qrel, run = dict(), dict()
    for word in similarity_dict:

        label, predictions = similarity_dict[word]
        run[word] = dict()
        """
        next loop's idea for reverse priority, is inspired by 1)https://github.com/bandpooja/Assignment_1
        2)https://github.com/cvangysel/pytrec_eval/issues/16
        """
        for pred in predictions[0:1]:
            run[word][pred[0]] = 3

        for pred in predictions[1:5]:
            if pred[0] not in run[word]:
                run[word][pred[0]] = 2

        for pred in predictions[5:10]:
            if pred[0] not in run[word]:
                run[word][pred[0]] = 1

        qrel[word] = {label: 1}
    return qrel, run

def evaluate(similarity_dict: dict, metric={'success_1,5,10'}) -> tuple:
    """
    Args:
        similarity_dict: output of get_similarities() function
        metric: evaluation metrics
    Returns:
        tuple of two dataframes with calculated evaluation metrics and mean of evaluation metrics
    """
    qrel, run_ = building_pytrec_input(similarity_dict)
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metric).evaluate(run_))
    df_mean = df.mean(axis=1)
    df_mean.to_csv('../output/evaluation_mean.csv', header=False, index=True)
    df.to_csv('../output/evaluation.csv', header=False, index=True)
    return df, df_mean


if __name__ == "__main__":

    run('../birkbeck_tes.dat', 10, 100)
    objects = list()

    with (open('../data/similarity_dict.pkl', 'rb')) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    similarities = dict()
    for dictionary in objects:
        similarities.update(dictionary)

    evaluate(similarities)
