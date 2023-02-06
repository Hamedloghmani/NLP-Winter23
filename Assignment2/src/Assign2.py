from tqdm import tqdm
from nltk.corpus import brown
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

from Assignment1.src.Assign1 import evaluate


# nltk.download('brown')

def load_brown() -> list:
    sentences = brown.sents()
    output = list()
    for i in range(len(sentences)):
        output.append(list(map(str.lower, sentences[i])))

    return output


def preprocess(misspelling_file: str = '../data/APPLING1DAT.643'):
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


def build_ngram(n: int, sentences: list):

    train_data, padded_sentences = padded_everygram_pipeline(n, sentences)
    model = MLE(n)
    model.fit(train_data, padded_sentences)
    return model


def get_suggestions(limit: int, language_model, misspelled_dataset):

    output_dict = dict()

    for record in tqdm(misspelled_dataset):
        probs = list()
        for i in language_model.vocab:
            probs.append((i, language_model.score(i, record[2][:2])))

        probs.sort(key=lambda x: x[1], reverse=True)
        output_dict[record[0]] = (record[1], probs[:limit])

    return output_dict


def run(n, output):
    sentences = load_brown()
    data = preprocess()
    model = build_ngram(n, sentences)
    suggestions = get_suggestions(10, model, data)
    evaluate(suggestions, output=output)


if __name__ == "__main__":
    n_list = [1, 2, 3, 5, 10]
    for n in n_list:
        run(n, f'../output/{n}')