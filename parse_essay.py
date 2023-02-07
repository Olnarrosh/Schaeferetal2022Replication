from pathlib import Path
from sentence_splitter import SentenceSplitter

def parse_essay_corpus():
    docs = []
    for f in Path("./data/essay/").iterdir():
        if f.suffix == ".txt":
            docs += parse_essay_file(str(f), str(f.with_suffix(".ann")))
    return docs

def parse_essay_file(filename_txt, filename_ann):
    with open(filename_txt, encoding="utf-8") as txt:
        splitter = SentenceSplitter(language="en")
        sentences = filter(None, splitter.split(txt.read()))
    with open(filename_ann, encoding="utf-8") as ann:
        claims = []
        for line in ann:
            if len(line) < 2:
                continue
            col = line[:-1].split("\t")
            if "Claim" in col[1]:
                claims.append(col[2])
    return [(s, any(x for x in claims if x in s)) for s in sentences]
