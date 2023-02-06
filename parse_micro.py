from pathlib import Path
from xml.etree.ElementTree import ElementTree

def parse_micro_corpus():
    docs = []
    for f in Path("./data/micro/").iterdir():
        if f.suffix == ".xml":
            docs += parse_micro_file(str(f))
    return docs

def parse_micro_file(filename):
    reader = ElementTree(file=filename)
    # mark only claims with the "implicit" tag as claims (?)
    return [(x.text, "implicit" in x.attrib) for x in reader.findall("./edu")]
