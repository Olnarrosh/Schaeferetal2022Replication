from pathlib import Path

def parse_cmv_corpus():
    docs = []
    for cls in ["positive", "negative"]:
        path = Path(f"data/cmv/{cls}/")
        for f in path.iterdir():
            docs += parse_cmv_file(str(f))
    return docs

def parse_cmv_file(filename):
    sentences = []
    with open(filename, encoding="utf-8") as file:
        for line in file:
            # stdlib xml parsers would be overkill -- we don't care about
            # document structure, we only need to detect <claim> tags
            tags = ""
            text = ""
            position = 0
            while position < len(line) - 1:
                next_tag_start = line.find("<", position)
                if next_tag_start == -1:
                    text += line[position:-1]
                    break
                next_tag_end = line.index(">", position)+1
                text += line[position:next_tag_start]
                tags += line[next_tag_start:next_tag_end]
                position = next_tag_end
            tags = tags.strip()
            text = text.strip()
            if text.startswith("%gt;"):
                # reddit quote -- ignore to avoid duplicate sentences
                continue
            if "<title>" in tags or "<source>" in tags:
                # titles usually do contain claims, but are not marked as such
                continue
            if len(text):
                sentences.append((text, "<claim" in tags or "</claim" in tags))
    return sentences



