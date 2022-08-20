fr=open('../dataset/human/ligand_uniport.fasta', 'r')
fw=open('../dataset/human/ligand_uniport_cleaning.fasta', 'w')
seq={}
for line in fr:
    if line.startswith('>'):    #判断字符串是否以‘>开始’
        name=line.split()[0]    #以空格为分隔符，并取序列为0的项。
        name = name.split('|')
        name = '>'+name[1]
        seq[name] = ''
    else:
        seq[name]+=line.replace('\n', '')
fr.close()



for i in seq.keys():
    fw.write(i)
    fw.write('\n')
    seq[i] = seq[i].replace('"','')
    fw.write(seq[i])
    fw.write('\n')
fr.close()