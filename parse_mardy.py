from pathlib import Path
from json import load

def parse_mardy_corpus():
    return [parse_mardy_file(str(f)) for f in Path("./data/mardy").iterdir()]

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
