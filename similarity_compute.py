#This code is to calculate the similarity
from collections import Counter

DIR = ''
fname = 'brownclusters.txt'
fpath = DIR+fname
#read file
data = {}
f=open(fpath, 'rb')
lines = f.readlines()
rows = [eval(i.strip()) for i in lines]
for i in rows:
    data[i[0]] = i[1]
f.close()
#print data


def Similarity(word1,word2,data):
    #wor1: eg'the'
    #Here w1 and w2 are both code, eg:'11110'
    w1 = data[word1]
    w2 = data[word2]
    len_min = min([len(w1),len(w2)])
    if len_min == 0:
        print "The length of ", word1, word2, "is 0"
    score = 0
    for i in range(len_min):
        if w1[i] == w2[i]:
            score += 1
        else:
            break
    return score

#print Similarity('111110','1')

def Ten_most_similar(w1, data):
    #here w1 is a word
    #     data is a dict
    word_score = Counter()
    for elem in data.items():
        if elem[0] == w1:
            continue
        score = Similarity(w1,elem[0],data)
        word_score[elem[0]] = score
    answer = [i[0] for i in word_score.most_common(10)]
    return answer

print Ten_most_similar('the',data)
print Ten_most_similar('army',data)
print Ten_most_similar('received',data)
print Ten_most_similar('famous',data)