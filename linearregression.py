
import pandas as pd
from itertools import product, permutations
from preprocess import compute_similarity, tokenize
import statsmodels.api as sm
from math import log
from collections.abc import Iterable
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
    sim = {cc: 1 if cc[0] == cc[1] else compute_similarity([corpora[c] for c in cc]) for cc in corp_combos}
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
        iv_vals = [sim[cc], log(size[cc[0]]), claim_ratio[cc[0]], claim_ratio[cc[1]]]
        iv_vals += [1 if i == corpus.index(cc[0]) else 0 for i in range(len(corpus))]
        iv_vals.append(1)
        for score in (scores[order] if isinstance(scores[order], Iterable) else [scores[order]]):
            indep_vars.append(iv_vals)
            dep_vars.append(score)
    # compute leave one out values
    for c in corpus:
        other = [x for x in corpus if c != x]
        order = _get_contained_ordering_(other, scores.keys())
        if not order:
            continue
        other_combos = product(other, other)
        other_combos = set(tuple(sorted(x)) for x in other_combos if x[0] != x[1])
        iv_vals = [sum(sim.get(_get_contained_ordering_((c, o), sim), 0) for o in other_combos) / len(other),
                   sum(log(size[o]) for o in other),
                   sum(len([s for s in corpora[o] if s[1]]) for o in other) / sum(size[o] for o in other),
                   claim_ratio[c]]
        iv_vals += [0 if i == corpus.index(c) else 1 for i in range(len(corpus))]
        iv_vals.append(1)
        for score in (scores[order] if isinstance(scores[order], Iterable) else [scores[order]]):
            indep_vars.append(iv_vals)
            dep_vars.append(score)
    indep_vars = pd.DataFrame(indep_vars, columns=["similarity", "log(size)", "claim ratio (source)", "claim ratio (target)"] + corpus + ["(intercept)"])
    dep_vars = pd.DataFrame(dep_vars)
    return sm.OLS(dep_vars, indep_vars).fit()
            

# helper function so tuples in scores keys can be in any order
def _get_contained_ordering_(permutable, container):
    for p in permutations(permutable):
        if p in container:
            return p
    return None



if __name__ == "__main__":

    corpora_dict = {
    "cmv": [tokenize(sent) for sent in parse_cmv.parse_cmv_corpus()],
    "usdeb":[tokenize(sent) for sent in parse_usdeb.parse_usdeb_corpus()],
    "micro":[tokenize(sent) for sent in parse_micro.parse_micro_corpus()],
    "essay":[tokenize(sent) for sent in parse_essay.parse_essay_corpus()],
    "mardy":[tokenize(sent) for sent in parse_mardy.parse_mardy_corpus()]
    }

    # f-score results from Logistic Regression Model
    scores_dict = {
        ("cmv", "cmv"): [0.337, 0.667, 0.566, 0.648],
        ("cmv", "usdeb"): 0.427,
        ("cmv", "micro"): 0.091,
        ("cmv", "essay"): 0.288,
        ("cmv", "mardy"): 0.009,
        ("usdeb", "cmv"): 0.532,
        ("usdeb", "usdeb"): [0.585, 0.705, 0.687, 0.729],
        ("usdeb", "micro"): 0.157,
        ("usdeb", "essay"): 0.551,
        ("usdeb", "mardy"): 0.182,
        ("micro", "cmv"): 0.0,
        ("micro", "usdeb"): 0.001,
        ("micro", "micro"): [0.0, 0.833, 0.444, 0.0],
        ("micro", "essay"): 0.0,
        ("micro", "mardy"): 0.0,
        ("essay", "cmv"): 0.345,
        ("essay", "usdeb"): 0.254,
        ("essay", "micro"): 0.190,
        ("essay", "essay"): 0.467,
        ("essay", "mardy"): 0.279,
        ("mardy", "cmv"): 0.128,
        ("mardy", "usdeb"): 0.237,
        ("mardy", "micro"): 0.118,
        ("mardy", "essay"): 0.149,
        ("mardy", "mardy"): 0.414,
        ("cmv", "usdeb", "micro", "essay"): 0.026,
        ("cmv", "usdeb", "micro", "mardy"): 0.424,
        ("cmv", "usdeb", "essay", "mardy"): 0.188,
        ("cmv", "micro", "essay", "mardy"): 0.457,
        ("usdeb", "micro", "essay", "mardy"): 0.455
        }

    print(regression(corpora_dict, scores_dict).summary())


