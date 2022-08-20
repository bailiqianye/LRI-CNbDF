import urllib.request

import numpy as np
import pandas as pd
from pandas import DataFrame


def Deduplicate(ls):
  news_list = []
  for id in ls:
    if id not in news_list:
      news_list.append(id)
  return np.array(news_list)


def GetMiddleStr(content,startStr,endStr):
  startIndex = content.index(startStr)
  if startIndex>=0:
    startIndex += len(startStr)
  endIndex = content.index(endStr)
  return content[startIndex:endIndex]

# Related = pd.read_csv("../dataset/mouse/mouse_lr_pair.csv").to_numpy()
# receptor_ensembl_protein_id = Related[:, [5]]
# receptor_ensembl_protein_id = Deduplicate(receptor_ensembl_protein_id)
# DataFrame(receptor_ensembl_protein_id).to_csv("../dataset/mouse/ligand_ensembl_protein_id_only_name.csv", index=False, header=None, sep=',')

receptor_ensembl_protein_id_only = pd.read_csv("../dataset/mouse/receptor_ensembl_protein_id_only_name.csv",header=None).to_numpy()



for i in range(len(receptor_ensembl_protein_id_only)):
  url = "https://www.uniprot.org/uniprot/?query="+receptor_ensembl_protein_id_only[i][0]+"&sort=score"
  req=urllib.request.Request(url)
  resp=urllib.request.urlopen(req)
  data=resp.read().decode('utf-8')
  id = GetMiddleStr(data,'<input class="basket-item namespace-uniprot" id="checkbox_','" type="checkbox"/></td><td class="entryID">')
  print(id)
  File1 = open("../dataset/mouse/receptor_uniport_uid.txt", 'a')
  File1.write(id+'\n')
  File1.close()
  url_fasta = "https://www.uniprot.org/uniprot/"+id+".fasta"
  req2 = urllib.request.Request(url_fasta)
  resp2 = urllib.request.urlopen(req2)
  fasta = resp2.read().decode('utf-8')
  File = open("../dataset/mouse/receptor_uniport.fasta", 'a')
  File.write(fasta)
  File.close()



