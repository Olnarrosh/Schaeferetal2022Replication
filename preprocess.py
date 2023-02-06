import torch
from transformers import BertTokenizer, BertModel
from scipy.stats import spearmanr

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# tokenize all sentences in the corpus, then extract and pool embeddings
def convert_corpus(corpus: list[(str, bool)]) -> list[(str, list[float], bool)]:
    sentence_embeddings = []
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for sentence, is_claim in corpus:
        tokens = tokenizer(sentence)["input_ids"]
        word_embeddings = model(torch.tensor([tokens], device=dev))["last_hidden_state"][0]
        embeddings_sum = [sum(word_embeddings[i][j].item() for i in range(len(tokens))) for j in range(word_embeddings.size()[1])]
        sentence_embeddings.append((sentence, [e / len(tokens) for e in embeddings_sum], is_claim))
    return sentence_embeddings

# compute spearman similarity based on 500 most frequent words of each corpus
def compute_similarity(corpora):
    top_types = []
    for corpus in corpora:
        tokens = [tokenizer(sentence)["input_ids"] for sentence, _ in corpus]
        tokens = [token for sentence in tokens for token in sentence]
        types = set(tokens)
        freq = {t: tokens.count(t) for t in types}
        top_types.append(sorted(types, key=freq.get, reverse=True)[:500])
    return spearmanr(*top_types).statistic
