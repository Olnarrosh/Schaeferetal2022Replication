import pandas as pd


def csv_Reader(filename: str):
    """
    created w.r.t the USDEB-Data
    reads a csv-file and returns a List of Tuples containing the String and the Calssification for Claims/Premise
    if the Classification is 'Mixed' it also shows
    """
    tuples_list = []
    file = pd.read_csv(filename)
    for i in range(len(file)):
        if file.loc[i, 'Component'] == 'Claim':
            tuple = (file.loc[i, 'Speech'], bool(1),)
        else:
            tuple = (file.loc[i, 'Speech'], bool(0))

        tuples_list.append(tuple)

    return tuples_list


if __name__ == '__main__':
    #path = "H:\\Uni\\Winter Semester 2022 2023\\Projekt _Seminar\\sentence_db_candidate.csv"
    path = "/home/users0/murativa/Downloads/sentence_db_candidate.csv"
    csv_Reader(path)