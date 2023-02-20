import pandas as pd
from pathlib import Path


def csv_Reader():
    """
    created w.r.t the USDEB-Data
    reads the csv-file and returns a List of Tuples containing the String-Sentence and its Classification for Claims
    Sentences annotated as Claims or Mixed have value True for Claim and Premiss don't.
    """
    path = Path("./data/usdeb/sentence_db_candidate.csv")
    tuples_list = []
    file = pd.read_csv(path)
    for i in range(len(file)):
        tuples_list.append((file.loc[i, 'Speech'], file.loc[i, 'MainTag'] in ('Claim', 'Mixed')))

    return tuples_list
