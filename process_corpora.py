"""
@Author Tana Deeg and Sandro Weick
"""
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
    validate = finallist[int(len(finallist)*0.9):]

    with open(f"./processed_data_results/train_{corpus}_file.pkl", "wb") as file_train:
        pickle.dump(train, file_train)
    with open(f"./processed_data_results/val_{corpus}_file.pkl", "wb") as file_val:
        pickle.dump(validate, file_val)

# prepare all 5 leave one out approaches by merging four 90% splits
def get_leave_one_out(leave_out:str, corpora=["cmv", "essay", "mardy", "micro", "usdeb"]):
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
        create_gold_list(corpus)
    for corpus in corpora:
        get_leave_one_out(corpus, corpora)




if __name__ == "__main__":

    process_all_corpora(corpora=["cmv", "essay", "micro", "usdeb", "mardy"])
    sub_all = []
    subset_1 = []
    subset_2 = []
    subset_3 = []
    subset_4 = []
    sub_5 = []
    sub_6 = []
    sub_7 = []
    for corpus in ["cmv", "essay", "micro", "usdeb", "mardy"]:
        match corpus:
            case "cmv":
                cmv_corpus = parse_cmv.parse_cmv_corpus()
            case "essay":
                essay_corpus = parse_essay.parse_essay_corpus()
            case "usdeb":
                usdeb_corpus = parse_usdeb.parse_usdeb_corpus()
            case "mardy":
                mardy_corpus = parse_mardy.parse_mardy_corpus()
            case "micro":
                micro_corpus = parse_micro.parse_micro_corpus()

    sub_all.append(cmv_corpus)
    sub_all.append(essay_corpus)
    sub_all.append(micro_corpus)
    sub_all.append(usdeb_corpus)
    print(f"sim ohne mardy")
    print(preprocess.compute_similarity(sub_all))

    print(f"sim mit mardy")
    sub_all.append(mardy_corpus)
    print(preprocess.compute_similarity(sub_all))

    subset_1.append(cmv_corpus)
    subset_1.append(essay_corpus)
    print(f"cmv and essay")
    print(preprocess.compute_similarity(subset_1))

    subset_2.append(cmv_corpus)
    subset_2.append(micro_corpus)
    print(f"cmv and micro")
    print(preprocess.compute_similarity(subset_2))

    subset_3.append(micro_corpus)
    subset_3.append(essay_corpus)
    print(f"micro and essay")
    print(preprocess.compute_similarity(subset_3))

    subset_4.append(micro_corpus)
    subset_4.append(micro_corpus)
    print(f"micro and micro")
    print(preprocess.compute_similarity(subset_4))


    sub_5.append(usdeb_corpus)
    sub_5.append(cmv_corpus)
    print(f"usdeb and cmv")
    print(preprocess.compute_similarity(sub_5))

    sub_6.append(usdeb_corpus)
    sub_6.append(essay_corpus)
    print(f"usdeb and essay")
    print(preprocess.compute_similarity(sub_6))

    sub_7.append(usdeb_corpus)
    sub_7.append(micro_corpus)
    print(f"usdeb and micro")
    print(preprocess.compute_similarity(sub_7))

    print("trying with mardy")
    sub_8 =[]
    sub_8.append(mardy_corpus)
    sub_8.append(micro_corpus)
    print(f"mardy and micro")
    print(preprocess.compute_similarity(sub_8))

    sub_9 = []
    sub_9.append(mardy_corpus)
    sub_9.append(usdeb_corpus)
    print(f"mardy and usdeb")
    print(preprocess.compute_similarity(sub_9))

    sub_10 = []
    sub_10.append(mardy_corpus)
    sub_10.append(cmv_corpus)
    print(f"mardy and cmv")
    print(preprocess.compute_similarity(sub_10))

    sub_11 = []
    sub_11.append(mardy_corpus)
    sub_11.append(essay_corpus)
    print(f"mardy and essay")
    print(preprocess.compute_similarity(sub_11))







