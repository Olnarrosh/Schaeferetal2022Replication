
import numpy as np
from itertools import product, permutations
from preprocess import compute_similarity, tokenize
from sklearn.linear_model import LinearRegression
import parse_cmv, parse_essay, parse_mardy, parse_micro, parse_usdeb




# takes two dicts as input, the first with corpus names (should be "cmv",
# "usdeb", "essay", "mardy", "micro") as keys and the tokenized sentences
# (use the tokenize function from preprocess) as values, the second with
# tuples of corpora as keys and f-scores as values
def regression(corpora: dict[str: list[float]], scores: dict[str: float]):
    # compute all combinations of corpora (order doesn't matter)
    corp_combos = product(corpora.keys(), corpora.keys())
    corp_combos = set(tuple(sorted(x)) for x in corp_combos)
    
    print("now computes similaritys")
    # compute similarity, corpus size, and claim ratio
    sim = {cc: compute_similarity([corpora[c] for c in cc]) for cc in corp_combos}
    print("done with similaritys")
    size = {c: len(corpora[c]) for c in corpora}
    claim_ratio = {c: len([s for s in corpora[c] if s[1]])/size[c] for c in corpora}
    corpus = list(corpora.keys()) # use list to ensure fixed order
    # compute independent variables for all corpus combinations
    indep_vars = []
    dep_vars = []
    for cc in corp_combos:
        order = _get_contained_ordering_(cc, scores.keys())
        if not order:
            if cc[0] == cc[1] and cc[0] in scores:
                order = cc[0]
            else:
                continue
        iv_vals = [sim[cc], size[cc[0]], claim_ratio[cc[0]], claim_ratio[cc[1]]]
        iv_vals += [1 if i == corpus.index(cc[0]) else 0 for i in range(len(corpus))]
        indep_vars.append(iv_vals)
        dep_vars.append(scores[order])
    # compute leave one out values
    print("now at leave one out")
    for c in corpus:
        other = [x for x in corpus if c != x]
        order = _get_contained_ordering_(other, scores.keys())
        if not order:
            continue
        iv_vals = [compute_similarity(other),
                   sum(size[o] for o in other),
                   sum(len([s for s in corpora[o] if s[1] for o in other])) / sum(size(o) for o in other),
                   ratio[c]]
        iv_vals += [0 if i == corpus.index(c) else 1 for i in range(len(corpus))]
        indep_vars.append(iv_vals)
        dep_vars.append(scores[order])
    return LinearRegression().fit(np.array(indep_vars), np.array(dep_vars)).coef_
            

# helper function so tuples in scores keys can be in any order
def _get_contained_ordering_(permutable, container):
    for p in permutations(permutable):
        if p in container:
            return p
    return None


#unabh. Variablen: spearman ähnlichkeit (3 kommastellen), Korpsugrößetraining(#Sätze), verhältnisclaimnichtclaimquell (anzahlclaims/anzahlsätzegesamt) auf 3 kommastellen, verhältnisclaimnichtclaimziel,
# usdebtrain, microtrain, essaytrain, cmvtrain, mardytrain  (--> alle binary)

# für leave one out: korpusgröße = summe aller größen, verhältnisclaims = summe aller claims / gesamtkorpusgröße

# also Länge pro liste ist 9
# spearman: cmv essay 0.209, cmv micro 0.249, micro essay 0.302, usdeb cmv 0.211, usdeb essay 0.281, usdeb micro 0.206
# Anzahl Listen = Anzahl Experimente: 5 + 20 + 5 = 30
if __name__ == "__main__":
    mardy = parse_mardy.parse_mardy_corpus()
    sub_mardy = []
    for el in mardy:
        sub_mardy.extend(el)

    corpora_dict = {
    "cmv": [tokenize(sent) for sent in parse_cmv.parse_cmv_corpus()],
    "usdeb":[tokenize(sent) for sent in parse_usdeb.parse_usdeb_corpus()],
    "micro":[tokenize(sent) for sent in parse_micro.parse_micro_corpus()],
    "essay":[tokenize(sent) for sent in parse_essay.parse_essay_corpus()],
    "mardy":[tokenize(sent) for sent in sub_mardy]
    }

    print("now at scores dict")

    # f-score results from Linear Regression Model
    scores_dict = {
        ("cmv", "cmv"): 0.627,
        ("cmv", "usdeb"): 0.427,
        ("cmv", "micro"): 0.091,
        ("cmv", "essay"): 0.288,
        ("cmv", "mardy"): 0.015,
        ("usdeb", "cmv"): 0.532,
        ("usdeb", "usdeb"): 0.694,
        ("usdeb", "micro"): 0.157,
        ("usdeb", "essay"): 0.551,
        ("usdeb", "mardy"): 0.057,
        ("micro", "cmv"): 0.0,
        ("micro", "usdeb"): 0.001,
        ("micro", "micro"): 0.0,
        ("micro", "essay"): 0.0,
        ("micro", "mardy"): 0.0,
        ("essay", "cmv"): 0.345,
        ("essay", "usdeb"): 0.254,
        ("essay", "micro"): 0.190,
        ("essay", "essay"): 0.467,
        ("essay", "mardy"): 0.0,
        ("mardy", "cmv"): 0.0,
        ("mardy", "usdeb"): 0.0, 
        ("mardy", "micro"): 0.0,
        ("mardy", "essay"): 0.0,
        ("mardy", "mardy"): 0.0,
        ("cmv", "usdeb", "micro", "essay"): 0.0,
        ("cmv", "usdeb", "micro", "mardy"): 0.447,
        ("cmv", "usdeb", "essay", "mardy"): 0.182,
        ("cmv", "micro", "essay", "mardy"): 0.37,
        ("usdeb", "micro", "essay", "mardy"): 0.4
        }
    
    print("now at regression")

    print(regression(corpora_dict, scores_dict))
    
    """
                    # 5 in domain
                    # TODO: alle leave one out, alle spearman mit mardy, alle korpusgröße mardy, alle claimverhältnisse mardy
    X = np.array([[1, 29621, 0.481, 0.481, 1, 0, 0, 0, 0],
                  [1, 965, 0.035, 0.035, 0, 1, 0, 0, 0],
                  [1, 1665, 0.305, 0.305, 0, 0, 1, 0, 0],
                  [1, 2798, 0.366, 0.366, 0, 0, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                  # 5 leave one out
                  [0, 0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 0, 0, 0, 1, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 0, 1],
                  [0, 35049, 0.451, 0, 1, 1, 1, 1, 0],
                  # 4 train usdeb
                  [0.206, 29621, 0.481, 0.035, 1, 0, 0, 0, 0],
                  [0.281, 29621, 0.481, 0.305, 1, 0, 0, 0, 0],
                  [0.211, 29621, 0.481, 0.366, 1, 0, 0, 0, 0],
                  [0, 29621, 0.481, 0, 1, 0, 0, 0, 0],
                  #4 train micro
                  [0.206, 965, 0.035, 0.481, 0, 1, 0, 0, 0],
                  [0.302, 965, 0.035, 0.305, 0, 1, 0, 0, 0],
                  [0.249, 965, 0.035, 0.366, 0, 1, 0, 0, 0],
                  [0, 965, 0.035, 0, 0, 1, 0, 0, 0],
                  # 4 train essay
                  [0.281, 1665, 0.305, 0.481, 0, 0, 1, 0, 0],
                  [0.302, 1665, 0.305, 0.035, 0, 0, 1, 0, 0],
                  [0.209, 1665, 0.305, 0.366, 0, 0, 1, 0, 0],
                  [0, 1665, 0.305, 0, 0, 0, 1, 0, 0],
                  #4 train cmv
                  [0.211, 2798, 0.366, 0.481, 0, 0, 0, 1, 0],
                  [0.249, 2798, 0.366, 0.035, 0, 0, 0, 1, 0],
                  [0.209, 2798, 0.366, 0.305, 0, 0, 0, 1, 0],
                  [0, 2798, 0.366, 0, 0, 0, 0, 1, 0],
                  # 4 train mardy
                  [0, 0, 0, 0.481, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0.035, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0.305, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0.366, 0, 0, 0, 0, 1]])
    y = [] #TODO insert all F-Scores, auf 3 kommastellen runden
    reg = LinearRegression().fit(X, y)
    # prints all weights -> see which independent variables are most important!!
    print(reg.coef_)
    """

