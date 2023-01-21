from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def convert_corpus(corpus: list[(list[str], bool)]) -> list[(dict, bool)]:
    sentences = []
    for sentence, is_claim in corpus:
        tokenized = tokenizer(sentence)
        tokens = tokenized["input_ids"]
        pooled = round(sum(tokens)/len(tokens))
        sentences.append(({"input_ids": [pooled], "attention_mask": [1]}, is_claim))
    return sentences
