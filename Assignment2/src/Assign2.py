import nltk.lm
from tqdm import tqdm
from nltk.corpus import brown
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

from Assignment1.src.Assign1 import evaluate


# nltk.download('brown')

def load_brown() -> list:
    """
    Returns:
        a list of lower cased sentences in brown dataset.
    """
    sentences = brown.sents()
    output = list()
    for i in range(len(sentences)):
        output.append(list(map(str.lower, sentences[i])))

    return output


def preprocess(misspelling_file: str = '../data/APPLING1DAT.643') -> list:
    """
    Args:
        misspelling_file: address of the misspelling corpus
    Returns:
        list of tuples ( each tuple is (misspelled word, correct spelling, sentence that misspelling happened))
    """
    data = list()
    with open(misspelling_file) as file:
        spellings = file.readlines()
        spellings = list(map(str.lower, map(str.strip, spellings)))
    for record in tqdm(spellings):
        if record.startswith('$'):
            pass
        else:
            record = record.split()
            data.append((record[0], record[1], record[2: record.index('*')]))
    return data


def build_ngram(n: int, sentences: list) -> nltk.lm.MLE:
    """
    Args:
        n: ngram will be build based on this value
        sentences: list of sentences to train language model on them
    Returns:
        language model's object
    """
    train_data, padded_sentences = padded_everygram_pipeline(n, sentences)
    model = MLE(n)
    model.fit(train_data, padded_sentences)
    return model


def get_suggestions(limit: int, language_model: nltk.lm.MLE, misspelled_dataset: list) -> dict:
    """
    Args:
        limit: suggestion list size
        language_model: language model object
        misspelled_dataset: dictionary of misspelled dataset
    Returns:
        dictionary with misspelled word as keys, tuple of correct spelling and suggestions as value
    """
    output_dict = dict()

    for record in tqdm(misspelled_dataset):
        probs = list()
        for i in language_model.vocab:
            probs.append((i, language_model.score(i, record[2][:2])))

        probs.sort(key=lambda x: x[1], reverse=True)
        output_dict[record[0]] = (record[1], probs[:limit])

    return output_dict


def run(n: int, output: str):
    """
    Args:
        n: value for n-gram
        output: output directory for evaluation results
    """
    sentences = load_brown()
    data = preprocess()
    model = build_ngram(n, sentences)
    suggestions = get_suggestions(10, model, data)
    evaluate(suggestions, output=output)


if __name__ == "__main__":
    n_list = [1, 2, 3, 5, 10]
    for n in n_list:
        run(n, f'../output/{n}')