
import numpy as np
from itertools import product, permutations

import preprocess
from preprocess import compute_similarity, tokenize
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import parse_cmv, parse_essay, parse_mardy, parse_micro, parse_usdeb




# takes two dicts as input, the first with corpus names (should be "cmv",
# "usdeb", "essay", "mardy", "micro") as keys and the tokenized sentences
# (use the tokenize function from preprocess) as values, the second with
# tuples of corpora as keys and f-scores as values
def regression(corpora: dict[str: list[float]], scores: dict[str: float]):
    # compute all combinations of corpora (order doesn't matter)
    corp_combos = product(corpora.keys(), corpora.keys())
    corp_combos = set(tuple(sorted(x)) for x in corp_combos)
    
    # compute similarity, corpus size, and claim ratio
    sim = {cc: compute_similarity([corpora[c] for c in cc]) for cc in corp_combos}
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

    """
    corpora_dict = {
    "cmv": [sent.tokenize() for sent in parse_cmv],
    "usdeb":[sent.tokenize() for sent in parse_usdeb],
    "micro":[sent.tokenize() for sent in parse_micro],
    "essay":[sent.tokenize() for sent in parse_essay],
    "mardy":[sent.tokenize() for sent in [el for el in parse_mardy]]
    }

    scores_dict = {
        ("cmv", "cmv"): 0.627,
        ("cmv", "usdeb"): 0.427,
        ("cmv", "micro"): 0.091,
        ("cmv", "essay"): 0.288,
        ("cmv", "mardy"): # TODO
        ("usdeb", "cmv"): 0.532,
        ("usdeb", "usdeb"): 0.694,
        ("usdeb", "micro"): 0.157,
        ("usdeb", "essay"): 0.551,
        ("usdeb", "mardy"): # TODO
        ("micro", "cmv"): 0.661,
        ("micro", "usdeb"): 0.001,
        ("micro", "micro"): 0.0,
        ("micro", "essay"): 0.0,
        ("micro", "mardy"): # TODO
        ("essay", "cmv"): 0.345,
        ("essay", "usdeb"): 0.254,
        ("essay", "micro"): 0.190,
        ("essay", "essay"): 0.467,
        ("essay", "mardy"): # TODO
        ("mardy", "cmv"): #TODO
        ("mardy", "usdeb"): # TODO
        ("mardy", "micro"): # TODO
        ("mardy", "essay"): # TODO
        ("mardy", "mardy"): # TODO
        ("cmv", "usdeb", "micro", "essay"): # TODO
        ("cmv", "usdeb", "micro", "mardy"): # TODO
        ("cmv", "usdeb", "essay", "mardy"): # TODO
        ("cmv", "micro", "essay", "mardy"): # TODO
        ("usdeb", "micro", "essay", "mardy"): # TODO
        }

    regression(corpora_dict, scores_dict)
    
    
    # unabh. Variablen: spearman ähnlichkeit (3 kommastellen), Korpsugrößetraining(#Sätze), verhältnisclaimnichtclaimquell (anzahlclaims/anzahlsätzegesamt) auf 3 kommastellen, verhältnisclaimnichtclaimziel,
    # usdebtrain, microtrain, essaytrain, cmvtrain, mardytrain  (--> alle binary)

    # für leave one out: korpusgröße = summe aller größen, verhältnisclaims = summe aller claims / gesamtkorpusgröße

    # also Länge pro liste ist 9
    # spearman: cmv essay 0.209, cmv micro 0.249, micro essay 0.302, usdeb cmv 0.211, usdeb essay 0.281, usdeb micro 0.206
    # Anzahl Listen = Anzahl Experimente: 5 + 20 + 5 = 30
                    # 5 in domain
                    # TODO: alle leave one out, alle spearman mit mardy, alle korpusgröße mardy, alle claimverhältnisse mardy
    X = np.array([[29621, 0.481, 0.481, 1, 0, 0, 0, 0],
                  [965, 0.035, 0.035, 0, 1, 0, 0, 0],
                  [1665, 0.305, 0.305, 0, 0, 1, 0, 0],
                  [2798, 0.366, 0.366, 0, 0, 0, 1, 0],
                  [18477, 0.048, 0.048, 0, 0, 0, 0, 1],
                  # 5 leave one out
                  [23905, 0.103, 0.481, 0, 1, 1, 1, 1],
                  [52561, 0.317, 0.035, 1, 0, 1, 1, 1],
                  [51861, 0.312, 0.305, 1, 1, 0, 1, 1],
                  [50728, 0.309, 0.366, 1, 1, 1, 0, 1],
                  [35049, 0.451, 0.048, 1, 1, 1, 1, 0],
                  # 4 train usdeb
                  [29621, 0.481, 0.035, 1, 0, 0, 0, 0],
                  [29621, 0.481, 0.305, 1, 0, 0, 0, 0],
                  [29621, 0.481, 0.366, 1, 0, 0, 0, 0],
                  [29621, 0.481, 0.048, 1, 0, 0, 0, 0],
                  #4 train micro
                  [965, 0.035, 0.481, 0, 1, 0, 0, 0],
                  [965, 0.035, 0.305, 0, 1, 0, 0, 0],
                  [965, 0.035, 0.366, 0, 1, 0, 0, 0],
                  [965, 0.035, 0.048, 0, 1, 0, 0, 0],
                  # 4 train essay
                  [1665, 0.305, 0.481, 0, 0, 1, 0, 0],
                  [1665, 0.305, 0.035, 0, 0, 1, 0, 0],
                  [1665, 0.305, 0.366, 0, 0, 1, 0, 0],
                  [1665, 0.305, 0.048, 0, 0, 1, 0, 0],
                  #4 train cmv
                  [2798, 0.366, 0.481, 0, 0, 0, 1, 0],
                  [2798, 0.366, 0.035, 0, 0, 0, 1, 0],
                  [2798, 0.366, 0.305, 0, 0, 0, 1, 0],
                  [2798, 0.366, 0.048, 0, 0, 0, 1, 0],
                  # 4 train mardy
                  [18477, 0.048, 0.481, 0, 0, 0, 0, 1],
                  [18477, 0.048, 0.035, 0, 0, 0, 0, 1],
                  [18477, 0.048, 0.305, 0, 0, 0, 0, 1],
                  [18477, 0.048, 0.366, 0, 0, 0, 0, 1]])

    y = [0.69, 0.83, 0.47, 0.63, 0,
         0.37, 0.18, 0.45, 0.4, 0,
         0.16, 0.55, 0.53, 0.06,
         0.0014, 0, 0, 0,
         0.25, 0.19, 0.35, 0,
         0.43, 0.09, 0.29, 0,
         0, 0, 0, 0] #TODO insert all F-Scores, auf 3 kommastellen runden
    reg = LinearRegression().fit(X, y)
    # prints all weights -> see which independent variables are most important!!
    print(reg.coef_)
    """

