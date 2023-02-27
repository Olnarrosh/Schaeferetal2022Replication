import pandas as pd

def parse_mardy_corpus():
    mardy = pd.read_csv("data/mardy/entire_debatenet.csv")
    tuple_list = []

    for sent, claim in zip(mardy["Sentence"], mardy["Overlaped_claim_ids"]):
        tuple_list.append((sent, isinstance(claim, str)))

    return tuple_list


"""
def parse_mardy_corpus():
    corpus = []
    for f in Path("data/mardy_old").iterdir():
        corpus += parse_mardy_file(str(f))
    return corpus

def parse_mardy_file(filename):
    with open(filename, encoding="utf-8") as file:
        data = load(file)
    claims = [(c["begin"], c["end"]) for c in data["claims"]]
    sentences = []
    for sent in data["sentences"]:
        begin = sent["begin"]
        end = sent["end"]
        # the sentence contains a claim if its range and that of a claim overlap
        is_claim = any(c[0] <= end and c[1] >= begin for c in claims)
        sentences.append((data["text"][begin:end], is_claim))
    return sentences
"""
