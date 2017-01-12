import csv
import numpy as np
import glob

# all file names, we have about 228
file_names = glob.glob("data/*/*.feat42")
num_files = len(file_names)


# we need to reorder the amino acids so it fits with the CB6133 style
aa_dict = {'A':0, 'C':1, 'E':2, 'D':3, 'G':4, 'F':5, 'I':6, 'H':7, 'K':8, 'M':9, 'L':10, 'N':11, 'Q':12, 'P':13, 'S':14, 'R':15, 'T':16, 'W':17, 'V':18, 'Y':19, 'X':20,'NoSeq':21}
aa_order = [aa_dict[i] for i in list("ARNDCQEGHILKMFPSTWYVX")]

pssm_dict = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K': 8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'X':19, 'Y':20}
pssm_order = [pssm_dict[i] for i in list("ARNDCQEGHILKMFPSTWYVX")]

ss_dict = {'L':0, 'B':1, 'E':2, 'G':3, 'I':4, 'H':5, 'S':6, 'T':7}

# reorder acording to a list of intergers (e.g. pssm_order)
def reorder_list(mylist, myorder):
    return [mylist[i] for i in myorder]

# setup data holders
X = np.zeros((num_files, 700, 42))
t = np.zeros((num_files, 700))
mask = np.zeros((num_files, 700))

# go through each file and fill information into data holders
for idx, file_name in enumerate(file_names):
    reader = csv.reader(open(file_name), delimiter="\t") # load data
    my_dict = {}
    for row in reader:
        fix = row[0].strip().split(' ')
        aa_elem = [float(i) for i in reorder_list(fix[3:24], aa_order)]
        aa_elem = np.array(aa_elem) # AA
        pssm_elem = [float(i) for i in reorder_list(fix[24:], pssm_order)]
        pssm_elem = np.asarray(pssm_elem) # PSSM
        my_dict[int(fix[0])-1] = (fix[1], # amino acid residue
                                  ss_dict[fix[2]],
                                  aa_elem,
                                  pssm_elem)
    X_part = np.zeros((700, 42))
    t_part = np.zeros((700,))
    mask_part = np.zeros((700,))
    for key, values in my_dict.iteritems():
        _, ss, aa, pssm = values
        john = np.concatenate([aa, pssm])
        X_part[key] = john
        t_part[key] = ss
        mask_part[key] = 1
    X[idx] = X_part
    t[idx] = t_part
    mask[idx] = mask_part

# make data accessible with a function
def get_data():
    return X, t, mask

if __name__ == '__main__':
    print(X.shape)
    print(t.shape)
    print(mask.shape)
