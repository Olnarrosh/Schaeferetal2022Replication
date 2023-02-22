"""
@Author Tana Deeg and Sandro Weick
"""
import parse_essay
import preprocess, parse_cmv, parse_micro, parse_essay , parse_usdeb, parse_mardy
import random
import pickle
# get all tuple lists from the 5 corpora
# split lists for train-test (shuffle first!), in Schaefer paper: 90% train, 10% validation&test
# merge lists for leave one out approach
# save all lists to have easy access

def get_corpus_ready(corpus: str):
    match corpus:
        case "cmv":
            loadedcorpus = parse_cmv.parse_cmv_corpus()
        case "essay":
            loadedcorpus = parse_essay.parse_essay_corpus()
        case "usdeb":
            loadedcorpus = parse_usdeb.parse_usdeb_corpus()
        case "mardy":
            loadedcorpus = parse_mardy.parse_mardy_corpus()
        case "micro":
            loadedcorpus = parse_micro.parse_micro_corpus()

    finallist = preprocess.convert_corpus(loadedcorpus)
    random.shuffle(finallist)
    train = finallist[0:int(len(finallist)*0.9)]
    print(len(train))
    validate = finallist[int(len(finallist)*0.9):]
    print(len(validate))

    with open(f"./processed_data_results/train_{corpus}_file.pkl", "wb") as file_train:
        pickle.dump(train, file_train)
    with open(f"./processed_data_results/val_{corpus}_file.pkl", "wb") as file_val:
        pickle.dump(validate, file_val)

# functionally equivalent to get_usdeb_ready(), but should work for all corpora
def get_leave_one_out(leave_out:str, corpora=["cmv", "essay", "micro"]):
    # corpora == list with names of all corpora ( == ["cmv", "essay", "mardy", "micro", "usdeb"])
    merged = []
    for corpus in corpora:
        if corpus != leave_out:
            with open(f"./processed_data_results/train_{corpus}_file.pkl", "rb") as g:
                corp = pickle.load(g)
                merged.extend(corp)
        
    with open(f"./processed_data_results/train_without_{leave_out}_file.pkl", "wb") as file:
        pickle.dump(merged, file)
            
# input should be eval split of corpus
# output: list with gold values (0 or 1)
def create_gold_list(corpus:str):
    with open(f"./processed_data_results/val_{corpus}_file.pkl", "rb") as f:
        file = pickle.load(f)
    goldlist  = []
    for i in file:
        goldlist.append(i[2])
    with open(f"./processed_data_results/gold_{corpus}_list.pkl", "wb") as goldfile:
        pickle.dump(goldlist, goldfile)
        

def process_all_corpora(corpora=["cmv", "essay", "mardy", "micro", "usdeb"]):
    for corpus in corpora:
        get_corpus_ready(corpus)
        get_leave_one_out(corpus, corpora)
        create_gold_list(corpus)

def claimcounter():
    microcount = 0
    micro  = preprocess.convert_corpus(parse_usdeb.parse_usdeb_corpus())
    for i in micro:
        if i[2] == 1:
            microcount += 1
    print(microcount)

if __name__ == "__main__":

    #process_all_corpora()
    claimcounter()

