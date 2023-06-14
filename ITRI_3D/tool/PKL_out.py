import pickle


def pkl(file_root,data):
    with open(file_root,'wb') as F:
        pickle.dump(data,F)
    ##ck
    with open(file_root,'rb') as G:
        ls = pickle.load(G)
    # print('Success!\n',ls)