import pytrec_eval
import pandas as pd


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

def evaluate(similarity_dict: dict, metric={'success_1,5,10'}, output = '../output') -> tuple:
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
    df_mean.to_csv(f'{output}/evaluation_mean.csv', header=False, index=True)
    df.to_csv(f'{output}/evaluation.csv', header=False, index=True)
    return df, df_mean
