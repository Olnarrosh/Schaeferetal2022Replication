from transformers import BertTokenizer, BertModel
from torch import tensor, cuda

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def convert_corpus(corpus: list[(list[str], bool)]) -> list[(list[float], bool)]:
    sentences = []
    for sentence, is_claim in corpus:
        tokenized = tokenizer(sentence)
        sentences.append((tokenized["input_ids"], is_claim))
    sentence_embeddings = []
    for sentence, is_claim in sentences:
        word_embeddings = model(tensor([sentence]))["last_hidden_state"][0]
        embeddings_sum = [sum(word_embeddings[i][j].item() for i in range(len(sentence))) for j in range(word_embeddings.size()[1])]
        sentence_embeddings.append(([e / len(sentence) for e in embeddings_sum], is_claim))
    return sentence_embeddings
