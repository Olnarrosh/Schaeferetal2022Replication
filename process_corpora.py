import preprocess, parse_cmv, parse_micro
import random
import pickle
# get all tuple lists from the 5 corpora
# split lists for train-test (shuffle first!), in Schaefer paper: 90% train, 10% validation&test
# merge lists for leave one out approach
# save all lists to have easy access

def get_cmv_ready():
    loadedcorpus = parse_cmv.parse_cmv_corpus()
    finallist = preprocess.convert_corpus(loadedcorpus)
    print(len(finallist))
    # shuffle to avoid order bias
    random.shuffle(finallist)
    print(len(finallist))
    # 90% Training, 10% validation
    train_cmv = finallist[0:int(len(finallist)*0.9)]
    print(len(train_cmv))
    validate_cmv = finallist[int(len(finallist)*0.9):]
    print(len(validate_cmv))
    with open("train_cmv_file.pkl", "wb") as train_cmv_file:
        pickle.dump(train_cmv, train_cmv_file)
    with open("val_cmv_file.pkl", "wb") as val_cmv_file:
        pickle.dump(validate_cmv, val_cmv_file)

def get_corpus_ready(corpus: str):
    match corpus:
        case "cmv":
            loadedcorpus = parse_cmv.parse_cmv_corpus()
        case "essay":
            loadedcorpus = [] # TODO parse_essay()
        case "usdeb":
            loadedcorpus = [] # TODO parse_usdeb()
        case "mardy":
            loadedcorpus = [] # TODO parse_mardy()
        case "micro":
            loadedcorpus = parse_micro.parse_micro_corpus()

    finallist = preprocess.convert_corpus(loadedcorpus)
    random.shuffle(finallist)
    train = finallist[0:int(len(finallist)*0.9)]
    print(len(train))
    validate = finallist[int(len(finallist)*0.9):]
    print(len(validate))

    with open(f"train_{corpus}_file.pkl", "wb") as file_train:
        pickle.dump(train, file_train)
    with open(f"val_{corpus}_file.pkl", "wb") as file_val:
        pickle.dump(validate, file_val)

# functionally equivalent to get_usdeb_ready(), but should work for all corpora
def get_leave_one_out(leave_out:str, corpora=["cmv", "essay", "mardy", "micro", "usdeb"]):
    # corpora == list with names of all corpora ( == ["cmv", "essay", "mardy", "micro", "usdeb"])
    merged = []
    for corpus in corpora:
        if corpus != leave_out:
            with open(f"train_{corpus}_file.pkl", "rb") as g:
                corp = pickle.load(g)
                merged.extend(corp)
        
    with open("leave_{leave_out}_out.pkl", "wb") as file:
        pickle.dump(merged, file)
            

"""
# ATTENTION: following four methods can't load corpus yet! (still copy pasted from cmv)
def get_usdeb_ready():
    loadedcorpus = parse_cmv.parse_cmv_corpus()
    tuplelist = preprocess.convert_corpus(loadedcorpus)
    # shuffle to avoid order bias
    finallist = random.shuffle(tuplelist)
    # 90% Training, 10% validation
    train_usdeb = finallist[0:int(len(finallist)*0.9)]
    validate_usdeb = finallist[int(len(finallist)*0.9):]
    with open("train_usdeb_file.pkl", "wb") as train_usdeb_file:
        pickle.dump(train_usdeb, train_usdeb_file)
    with open("val_usdeb_file.pkl", "wb") as val_usdeb_file:
        pickle.dump(validate_usdeb, val_usdeb_file)

def get_micro_ready():
    loadedcorpus = parse_cmv.parse_cmv_corpus()
    tuplelist = preprocess.convert_corpus(loadedcorpus)
    # shuffle to avoid order bias
    finallist = random.shuffle(tuplelist)
    # 90% Training, 10% validation
    train_micro = finallist[0:int(len(finallist)*0.9)]
    validate_micro = finallist[int(len(finallist)*0.9):]
    with open("train_micro_file.pkl", "wb") as train_micro_file:
        pickle.dump(train_micro, train_micro_file)
    with open("val_micro_file.pkl", "wb") as val_micro_file:
        pickle.dump(validate_micro, val_micro_file)

def get_mardy_ready():
    loadedcorpus = parse_cmv.parse_cmv_corpus()
    tuplelist = preprocess.convert_corpus(loadedcorpus)
    # shuffle to avoid order bias
    finallist = random.shuffle(tuplelist)
    # 90% Training, 10% validation
    train_mardy = finallist[0:int(len(finallist)*0.9)]
    validate_mardy = finallist[int(len(finallist)*0.9):]
    with open("train_mardy_file.pkl", "wb") as train_mardy_file:
        pickle.dump(train_mardy, train_mardy_file)
    with open("val_mardy_file.pkl", "wb") as val_mardy_file:
        pickle.dump(validate_mardy, val_mardy_file)


def get_essay_ready():
    loadedcorpus = parse_cmv.parse_cmv_corpus()
    tuplelist = preprocess.convert_corpus(loadedcorpus)
    # shuffle to avoid order bias
    finallist = random.shuffle(tuplelist)
    # 90% Training, 10% validation
    train_essay = finallist[0:int(len(finallist)*0.9)]
    validate_essay = finallist[int(len(finallist)*0.9):]
    with open("train_essay_file.pkl", "wb") as train_essay_file:
        pickle.dump(train_essay, train_essay_file)
    with open("val_essay_file.pkl", "wb") as val_essay_file:
        pickle.dump(validate_essay, val_essay_file)


# Now the merged lists for leave-one-out-approach
# IMPORTANT: run 5 methods for preprocessing first! (get_"corpus"_ready()
def get_without_cmv_ready():
    with open("train_usdeb_file.pkl", "rb") as g:
        usdeb = pickle.load(g)
    with open("train_micro_file.pkl", "rb") as h:
        micro = pickle.load(h)
    with open("train_mardy_file.pkl", "rb") as i:
        mardy = pickle.load(i)
    with open("train_essay_file.pkl", "rb") as i:
        essay = pickle.load(f)
    merged_without_cmv = usdeb + micro + mardy + essay
    with open("leave_cmv_out.pkl", "wb") as leave_cmv_out:
        pickle.dump(merged_without_cmv, leave_cmv_out)

def get_without_usdeb_ready():
    with open("train_cmv_file.pkl", "rb") as f:
        cmv = pickle.load(f)
    with open("train_micro_file.pkl", "rb") as h:
        micro = pickle.load(h)
    with open("train_mardy_file.pkl", "rb") as i:
        mardy = pickle.load(i)
    with open("train_essay_file.pkl", "rb") as i:
        essay = pickle.load(f)
    merged_without_usdeb = cmv + micro + mardy + essay
    with open("leave_usdeb_out.pkl", "wb") as leave_usdeb_out:
        pickle.dump(merged_without_usdeb, leave_usdeb_out)

def get_without_micro_ready():
    with open("train_cmv_file.pkl", "rb") as f:
        cmv = pickle.load(f)
    with open("train_usdeb_file.pkl", "rb") as g:
        usdeb = pickle.load(g)
    with open("train_mardy_file.pkl", "rb") as i:
        mardy = pickle.load(i)
    with open("train_essay_file.pkl", "rb") as i:
        essay = pickle.load(f)
    merged_without_micro = cmv + usdeb + mardy + essay
    with open("leave_micro_out.pkl", "wb") as leave_micro_out:
        pickle.dump(merged_without_micro, leave_micro_out)

def get_without_mardy_ready():
    with open("train_cmv_file.pkl", "rb") as f:
        cmv = pickle.load(f)
    with open("train_usdeb_file.pkl", "rb") as g:
        usdeb = pickle.load(g)
    with open("train_micro_file.pkl", "rb") as h:
        micro = pickle.load(h)
    with open("train_essay_file.pkl", "rb") as i:
        essay = pickle.load(f)
    merged_without_mardy = cmv + usdeb + micro + essay
    with open("leave_mardy_out.pkl", "wb") as leave_mardy_out:
        pickle.dump(merged_without_mardy, leave_mardy_out)


def get_without_essay_ready():
    with open("train_cmv_file.pkl", "rb") as f:
        cmv = pickle.load(f)
    with open("train_usdeb_file.pkl", "rb") as g:
        usdeb = pickle.load(g)
    with open("train_micro_file.pkl", "rb") as h:
        micro = pickle.load(h)
    with open("train_mardy_file.pkl", "rb") as i:
        mardy = pickle.load(i)
    merged_without_essay = cmv + usdeb + micro + mardy
    with open("leave_essay_out.pkl", "wb") as leave_essay_out:
        pickle.dump(merged_without_essay, leave_essay_out)

"""
# input should be eval split of corpus
# output: list with gold values (0 or 1)
def create_gold_list(corpus:str):
    with open(f"val_{corpus}_file.pkl", "rb") as f:
        file = pickle.load(f)
    goldlist  = []
    for i in file:
        goldlist.append(i[2])
    with open(f"gold_{corpus}_list.pkl", "wb") as goldfile:
        pickle.dump(goldlist, goldfile)






if __name__ == "__main__":

    get_corpus_ready("micro")

    create_gold_list("micro")
    with open("gold_cmv_list.pkl", "rb") as a:
        gold_list = pickle.load(a)
        print(gold_list)



