import csv
import numpy as np
import glob

# this is a script to format the CASP10 + CASP11 in similar manner to CB6133

# all file names, we have about 228
file_names = glob.glob("data/*/*.feat42")
num_files = len(file_names)


# we need to reorder the amino acids so it fits with the CB6133 style
aa_dict = {'A':0,
           'R':1,
           'N':2,
           'D':3,
           'C':4,
           'Q':5,
           'E':6,
           'G':7,
           'H':8,
           'I':9,
           'L':10,
           'K':11,
           'M':12,
           'F':13,
           'P':14,
           'S':15,
           'T':16,
           'W':17,
           'Y':18,
           'V':19,
           'X':20}
aa_order = [aa_dict[i] for i in list('ACEDGFIHKMLNQPSRTWVYX')]

pssm_dict = {'A':0,
             'R':1,
             'N':2,
             'D':3,
             'C':4,
             'Q':5,
             'E':6,
             'G':7,
             'H':8,
             'I':9,
             'L':10,
             'K':11,
             'M':12,
             'F':13,
             'P':14,
             'S':15,
             'T':16,
             'W':17,
             'Y':18,
             'V':19,
             'X':20}
pssm_order = [pssm_dict[i] for i in list("ACDEFGHIKLMNPQRSTVWXY")]

ss_dict = {'L':0,
           'B':1,
           'E':2,
           'G':3,
           'I':4,
           'H':5,
           'S':6,
           'T':7}

ss_dict = {'L':0,
           'B':1,
           'E':2,
           'G':3,
           'I':4,
           'H':5,
           'S':6,
           'T':7}

# reorder acording to a list of intergers (e.g. pssm_order)
def reorder_list(mylist, myorder):
    return [mylist[i] for i in myorder]

# setup data holders
X = np.zeros((num_files, 700, 42))
t = np.zeros((num_files, 700))
mask = np.zeros((num_files, 700))

# go through each file and fill information into data holders
for idx, file_name in enumerate(file_names):
    # open file
    reader = csv.reader(open(file_name), delimiter="\t")
    my_dict = {}
    # for each row in the file (sample along sequence)
    for row in reader:
        row_info = row[0].strip().split(' ')
        aa_info = row_info[3:24] # extract aa info [0, 0, .., 1, 0, ... ]
        pssm_info = row_info[24:] # extract pssm info [0.12, 071, .., 0.42 ]
        # order accordingly to CB6133
        aa_elem = [float(i) for i in reorder_list(aa_info, aa_order)] 
        aa_elem = np.array(aa_elem) # AA
        pssm_elem = [float(i) for i in reorder_list(pssm_info, pssm_order)]
        pssm_elem = np.asarray(pssm_elem) # PSSM
        # stack into a dictionary
        my_dict[int(row_info[0])-1] = (row_info[1], # amino acid residue
                                       ss_dict[row_info[2]],
                                       aa_elem,
                                       pssm_elem)
    # setup one sequence
    X_part = np.zeros((700, 42))
    t_part = np.zeros((700,))
    mask_part = np.zeros((700,))
    for key, values in my_dict.items():
        _, ss, aa, pssm = values
        X_sample = np.concatenate([aa, pssm])
        X_part[key] = X_sample
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
