import os
import random
import torch
import pandas as pd

data = []
for i in range(5):
    data_path_i = os.path.join(os.path.dirname(__file__), 'NetMHCpan_train',f'c00{i}_ba')
    data_i = pd.read_csv(data_path_i,names = ["R", "Y", "Genotype" ], sep = ' ')
    data.append(data_i)

data = pd.concat(data)
geno = data.Genotype.value_counts().index.to_list()
value = data.Genotype.value_counts().values

elements = "ACDEFGHIKLMNPQRSTVWY"
element_to_index = {char: idx for idx, char in enumerate(elements)}
def one_hot_encoding(seq):
    one_hot = torch.zeros((len(seq), len(elements)))
    for i, char in enumerate(seq):
        one_hot[i, element_to_index[char]] = 1
    return one_hot

def get_train_test_data(j):
    genotype = geno[j]
    train_test_ratio = random.uniform(0.7, 0.9)
    data_HLA_21 = data[data['Genotype'] == genotype]
    data_HLA_21 = data_HLA_21.drop("Genotype", axis = 1)
    data_HLA_21 = data_HLA_21[data_HLA_21['R'].str.len() >= 9]
    data_HLA_21 = data_HLA_21.reset_index(drop=True)
    train_size = int(len(data_HLA_21) * train_test_ratio)
    train_data, test_data = data_HLA_21[:train_size], data_HLA_21[train_size:]
    train_R, test_R = train_data.R.values, test_data.R.values
    train_Y, test_Y = train_data.Y.values, test_data.Y.values
    train_Y, test_Y = torch.tensor(train_Y, dtype=torch.float), torch.tensor(test_Y, dtype=torch.float)
    for i in range(len(train_R)):
        train_R[i] = one_hot_encoding(train_R[i])
    for i in range(len(test_R)):
        test_R[i] = one_hot_encoding(test_R[i])
    return train_R,train_Y,test_R,test_Y