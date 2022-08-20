import numpy as  np
import os
import pandas as pd
import networkx as nx
from pandas import DataFrame

def Deduplicate(ls):
    news_list =[]
    for id in ls:
        if id not in news_list:
            news_list.append(id)
    return np.array(news_list)

def put_csv(matrix):
    # 输出到CSV文件
    ss = DataFrame(matrix)
    ss.to_csv("human_Related.csv", index=False, header=None, sep=',')

Related = pd.read_csv("dataset/human/human_lr_pair.csv").to_numpy()
ligand_ensembl_protein_id = Related[:, [5]]
ligand_ensembl_protein_id_only = Deduplicate(ligand_ensembl_protein_id)

receptor_ensembl_protein_id = Related[:, [6]]
receptor_ensembl_protein_id_only = Deduplicate(receptor_ensembl_protein_id)

Related_temp = Related[:, [5,6]]

Related_temp =Related_temp.tolist()

Related = np.zeros((len(ligand_ensembl_protein_id_only),len(receptor_ensembl_protein_id_only)))

for (x,y) in Related_temp:
     X_index = np.where(ligand_ensembl_protein_id_only == x)
     Y_index = np.where(receptor_ensembl_protein_id_only == y)
     X=int(X_index[0][0])
     Y=int(Y_index[0][0])
     Related[X,Y] = 1

print(Related.shape)
print(receptor_ensembl_protein_id_only.shape)
Related = np.vstack([receptor_ensembl_protein_id_only.T,Related])
print(Related.shape)
ligand_name = np.vstack([np.zeros((1, 1)),ligand_ensembl_protein_id_only])
Related = np.hstack([ligand_name,Related])
print(Related)
put_csv(Related)