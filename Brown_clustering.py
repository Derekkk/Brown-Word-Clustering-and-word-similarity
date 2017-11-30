import os
import collections
from nltk.corpus import brown
from collections import defaultdict, Counter
from math import log
from itertools import chain, combinations


'''
Part 1: Read and process the dataprint vocab_UNK
'''
def toLowerCase(s):
    #Convert a sting to lowercase. E.g., 'BaNaNa' becomes 'banana'
    return s.lower()

def stripNonAlpha(s):
    # Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana'
    return ''.join([c for c in s if (c.isalpha()) ] )
######################################
# ReadData: read all the texts from the specific file, and return a list of all the text
# Input: file path, eg:DIR = 'brown/'
# output: list of all the text(tokenized)
#####################################
#DIR = 'brown/'
def ReadData(DIR):
    files = os.listdir(DIR)
    data = []
    for file in files:
        if file not in ['README', 'cats.txt', 'CONTENTS']:
            f = open(DIR + file, 'rb')
            line = f.readlines()
            temp_1 = [i.strip().split() for i in line if i != '\n' and i.strip() != '']
            for element in temp_1:
                temp_2 = ['<s>']+[toLowerCase(token.rsplit('/',1)[0]) for token in element if stripNonAlpha(token.rsplit('/',1)[0]) != '']+['</s>']
                data.append(temp_2)
            """
            temp_1 = [file+''+i for i in line if i != '\n' and i.strip()!='']
            data += temp_1
            """
            f.close()
    return data

######################################
# Vocabulary: Get vocabulary of the corpus.
# Input: corpus data, list: [[s1],[s2],...]
# output: dictionary, vorcabulary and corresponding number
#####################################
def Vocabulary(data):
    Vocab = []
    for sample in data:
        for word in sample:
            if word not in ['','<s>','</s>']:
                Vocab.append(word)
    Vocab = collections.Counter(Vocab)
    return Vocab

######################################
# UNK_handling: UNK handling for the corpus.
# Input: Vocab (Dict), data (list)
# output: Vocab after UNK (dict), data after UNK (list)
#####################################
def UNK_handing(Vocab, data):
    UNK_word = []
    vocabulary = Counter() #new Vocab with UNK
    UNK_count = 0
    for k in Vocab:
        if Vocab[k] <= 10:
            UNK_word.append(k)
            UNK_count += Vocab[k]
        else:
            vocabulary[k] = Vocab[k]
    vocabulary['UNK'] = UNK_count
    UNK_word = set(UNK_word)
    data_UNK = []
    for i in data:
        temp = []
        for j in i:
            if j in UNK_word:
                temp.append('UNK')
            else:
                temp.append(j)
        data_UNK.append(temp)
    return vocabulary,data_UNK



data = ReadData('brown/') #brown corpus, [[tokenized s1],[tokenized s2],[tokenized s3],...]
#print data[0]
#print brown.sents()[0]

vocab = Vocabulary(data) # original vocab, type: dictionary
vocab_UNK, data_UNK = UNK_handing(vocab,data) # vocab and data after UNK handling

#print data_UNK[0]
#print len(vocab.keys()),len(vocab_UNK.keys())
#print vocab_UNK.keys()

'''
Part 2: build a bi-gram tokens table, eg: table[token2][token1] == count(token1,token2)
        for computational convenience
'''
def Bigram_table(vocab, data):

    table = defaultdict(int)
    for tokens in data:
        for i in range(1, len(tokens) - 1):
            prev = tokens[i]
            curr = tokens[i + 1]
            if prev in table:
                temp = table[prev]
                if curr in temp:
                    temp[curr] += 1
                else:
                    temp[curr] = 1
            else:
                table[prev] = {}
                table[prev][curr] = 1
            '''
            if curr in table:
                d = table[curr]
                d[prev] += 1
            else:
                d = defaultdict(int)
                d[prev] += 1
                table[curr] = d
            '''
    bi_table = defaultdict(int)
    for sentence in data_UNK:
        for i in range(0,len(sentence)-1):
            token_1 = sentence[i]
            token_2 = sentence[i+1]
            bi_table[(token_1,token_2)] += 1

    return bi_table,table
# Bi_table is use method 1, temp is use method 2, the answer is the same
Bi_table, temp = Bigram_table(vocab_UNK,data_UNK)  #Bi_table: {(token1,token2):num, ...} ; temp: { token1:{token2: num, ...}, ...}

vocabulary = sorted(vocab_UNK.items(),key = lambda x:(x[1],x[0]),reverse = True) # sorted vocab, eg: [('UNK', 92350), ('the', 69971), ('of', 36412), ('and', 28853),...]
#print Bi_table.items()[:10] #[(('helium', 'temperature'), 1), (('little', 'note'), 1), (('youth', 'adopt'), 1),...]


'''define some global variants'''
def Merge_table(w1,w2):
#w1,w2: 'note','little'
    vocab_UNK[w1] += vocab_UNK[w2]
    del vocab_UNK[w2]
    for (x1, x2), count in Bi_table.items():
        if w2 not in (x1, x2):
            continue
        if x1 == w2 and x2 == w2:
            Bi_table[(w1, w1)] += count
        elif x1 == w2:
            Bi_table[(w1, x2)] += count
        elif x2 == w2:
            Bi_table[(x1, w1)] += count
        del Bi_table[(x1, x2)]

#N = len(vocab_UNK)
N = sum(Bi_table.values())
'''
# write words and corpus

f = open('brown_corpus_afterUNK.txt', 'w')
for element in data_UNK:
    sentence = ''
    for token in element:
        sentence = sentence +token + ' '
    f.write(sentence)
    f.write('\n')
f.close()
'''

'''
Part 3: Brown Clustering
'''
def count(c):

    return sum(vocab_UNK[w] for w in c)

def BiCount(c1, c2):
    # c1,c2:  ('of',) ('be', 'had')
    # cluster: {('the',): [('the',)], ('at',): [('at',)], ('a',): [('a',)], ('UNK',): [('UNK',)], ('be',): [('be',)], ('on',): [('on',)],...}
    return  sum(Bi_table[(i, j)] for i in c1 for j in c2) * 1.0


# c1:('the','dog')
def Mutual_Information(c1, c2):
    #c1, c2:  ('of', 'this')('of', 'this')
    biCount = BiCount(c1, c2)
    if not biCount:
        return 0
    individualCounts = count(c1) * count(c2) * 1.0
    return (biCount/N) * log((biCount * N) / individualCounts, 2)


# W_cache is cache to store the computation cost; here W_cache is a {}: W_cache:  {(('at',), ('with',)): -88.05399628932321, ...}
# c1 c2 :  ('the', 'a') ('it',)
def W(c1, c2, W_cache):
    if (c1, c2) in W_cache:
        return W_cache[(c1, c2)]
    result = Mutual_Information(c1, c2)
    if c1 != c2:
        result += Mutual_Information(c2, c1)
    W_cache[(c1, c2)] = result
    return result

#here cluster_i is a list
#cluster1 = ('apple',), here cluster is cluster name actually
def Compute_L(c1,c2,C,W_cache):
    otherNodes = tuple(x for x in C if x != c1 and x != c2)
    #print "answer",c1,c2
    return -W(c1, c2,W_cache) - W(c1, c1,W_cache) - W(c2, c2,W_cache) + W(c1 + c2, c1 + c2,W_cache) - \
           sum(W(c1, w,W_cache) for w in otherNodes) - \
           sum(W(c2, w,W_cache) for w in otherNodes) + \
           sum(W(c1 + c2, w, W_cache) for w in otherNodes)


# L(c1,c2) stores the change in total graph weight if merging c1 and c2 (http://cs.stanford.edu/~pliang/papers/meng-thesis.pdf)
# here c1,c2 is list
def Initial_L(C,W_cache):
    L = Counter()
    all_claster_pair = combinations(C,2) #[(('the',), ('at',)), (('the',), ('where',)), (('the',), ('what',)), (('the',), ('war',)), (('the',), ('may',)),...]

    for c1, c2 in all_claster_pair:
        L[(tuple(c1),tuple(c2))] = Compute_L(c1,c2,C,W_cache)

    return L

def Merged_L(c1,c2,m1,m2,C,W_cache):
    #print "c1,c2,m1,m2: ",c1,c2,m1,m2
    #print "cluster: ",cluster
    if (c1,c2) in L:
        temp = L[(c1,c2)]
    elif (c2,c1) in L:
        temp = L[(c1,c2)]
    else:
        return Compute_L(c1,c2,C,W_cache)
    return temp - sum(W(c,m,W_cache) for c in (c1,c2) for m in (m1,m2)) + sum(W(c,m1+m2,W_cache) for c in (c1,c2))


# C: ['cluster1', 'cluster2', ...]
#word_string: { 'word1':000, 'word2':001,...}
#cluster_i: cluster name, a single token, eg: ('cluster1')
# cluster: { 'cluster1':[w1,w2,w3], 'cluster2':[w1,w2,w3],...}
# merge_History: []
# L: store the change
#remaining_word: word list
def Merge(C, word_string,cluster1,cluster2, cluster , L,merge_History, remain_word,W_cache):
    '''part 1: encoder'''
    for x in cluster[cluster1]:
        #print "x: ",x,word_string[x[0]]
        word_string[x[0]] = '0'+word_string[x[0]]
    for x in cluster[cluster2]:
        word_string[x[0]] = '1'+word_string[x[0]]

    '''part2: upadate the cluster'''
    merge_History.append((cluster1,cluster2))
    cluster[cluster1] += cluster[cluster2] # lsit merge
    del cluster[cluster2] # we keep cluster and delete cluster2

    '''part 3: intial L'''
    del L[(cluster1, cluster2)]
    del L[(cluster2, cluster1)]
    '''part 4: new node come in'''

    new_word = remain_word[0]
    remain_word = remain_word[1:]
    cluster[new_word] = [new_word]
    other_clusters = tuple([x for x in C if x != cluster1 and x != cluster2])
    C = other_clusters + (cluster1,new_word)  # return C


    for i, j in combinations(other_clusters, 2):
        #print "combinations: ",other_clusters
        L[(i, j)] = Merged_L(i, j, cluster1, cluster2, C, W_cache )
    merged_node = cluster1

    Merge_table(cluster1[0], cluster2[0])

    for elem in other_clusters:
        del L[(elem, cluster1)]
        del L[(elem, cluster2)]
        del L[(cluster1, elem)]
        del L[(cluster2, elem)]
        L[(elem, merged_node)] = Compute_L(elem, merged_node, C, W_cache)
        L[(elem, new_word)] = Compute_L(elem, new_word, C, W_cache)
    L[(merged_node, new_word)] = Compute_L(merged_node, new_word, C, W_cache)

    return C, L,word_string,cluster ,merge_History, remain_word,W_cache

# Keep merging to construct a full hierarchy
def full_Merge(C, word_string,cluster1,cluster2, cluster , L,merge_History,W_cache):
    '''part 1: encoder'''
    for x in cluster[cluster1]:
        #print "x: ",x,word_string[x[0]]
        word_string[x[0]] = '0'+word_string[x[0]]
    for x in cluster[cluster2]:
        word_string[x[0]] = '1'+word_string[x[0]]

    '''part2: upadate the cluster'''
    merge_History.append((cluster1,cluster2))
    cluster[cluster1] += cluster[cluster2] # lsit merge
    del cluster[cluster2] # we keep cluster and delete cluster2

    '''part 3: intial L'''
    del L[(cluster1, cluster2)]
    del L[(cluster2, cluster1)]
    '''part 4: new node come in'''

    other_clusters = tuple([x for x in C if x != cluster1 and x != cluster2])
    C = other_clusters + (cluster1,)  # return C


    for i, j in combinations(other_clusters, 2):
        #print "combinations: ",other_clusters
        L[(i, j)] = Merged_L(i, j, cluster1, cluster2, C, W_cache )
    merged_node = cluster1

    Merge_table(cluster1[0], cluster2[0])

    for elem in other_clusters:
        del L[(elem, cluster1)]
        del L[(elem, cluster2)]
        del L[(cluster1, elem)]
        del L[(cluster2, elem)]
        L[(elem, merged_node)] = Compute_L(elem, merged_node, C, W_cache)

    return C, L,word_string,cluster ,merge_History, W_cache



'''
def saveProgress():
    with open('Clusters.pyon', 'w') as outfile:
        outfile.write(repr(cluster))
    with open('BinaryStrings.pyon', 'w') as outfile:
        outfile.write(repr(word_string))
'''
def saveProgress_final():
    with open('savedClusters_final.pyon', 'w') as outfile:
        outfile.write(repr(cluster))
    with open('savedBinaryStrings_final.pyon', 'w') as outfile:
        outfile.write(repr(word_string))

'''Initiate the parameters and start'''
print "Initialize parameters"
C = tuple([(i[0],) for i in vocabulary[:200]])  #[('UNK',), ('the',), ('of',), ('and',), ('to',), ('a',), ('in',), ('that',), ('is',), ('was',), ...]
remain_word = [(i[0],) for i in vocabulary[200:]]

word_string = Counter()
for i in vocabulary:
    word_string[i[0]] = ''
merge_history = []
W_cache = {}

# {('the',): [('the',)], ('at',): [('at',)], ('where',): [('where',)], ('what',): [('what',)], ....}
cluster = {}
for c in C:
        cluster[c] = [c]

#print vocab_UNK
print "Initialize L"
L = Initial_L(C,W_cache)
print L


mergeNumber = 0


#print  vocab_UNK['does'],vocab_UNK['is'],Bi_table[('there','is')]
while remain_word != []:
#for _ in range(10):
    mergeNumber += 1
    #choose highest to merge
    (winner1, winner2), quality = L.most_common(1)[0]
    print "___________________"
    print('Merging {} and {}'.format(winner1, winner2))
    print "number of remaining words",len(remain_word)
    C, L,word_string, cluster, merge_History, remain_word, W_cache = Merge(C,word_string,winner1, winner2,cluster,L,merge_history,remain_word,W_cache)

    W_cache = {}  # evict Weight cache for the next round of merge
    #print "quality: ",Quality(temp,vocab_UNK,cluster)
    #print 'L: ',L

    #print "this is ", _
    #print "cluster: ",cluster
    #print remain_word
    #print "L:",L
    #print "encode",word_string
    #print "Cluster: ",C

f = open('cluster_before_keep_merging_2.txt', 'w')
for element in cluster.items():
    f.write(str(element))
    f.write('\n')
f.close()

f_2 = open('encode_before_keepmerging_2.txt', 'w')
for element in word_string.items():
    f_2.write(str(element))
    f_2.write('\n')
f_2.close()


#Keep merging
while len(C) != 1:
    (winner1, winner2), quality = L.most_common(1)[0]
    print('Merging {} and {}'.format(winner1, winner2))
    C, L, word_string, cluster, merge_History, W_cache = full_Merge(C, word_string,winner1,winner2, cluster , L,merge_History,W_cache)
    #print "C: ",C


saveProgress_final()
print 'cluster: ',cluster
#print 'word_string: ',word_string
print 'C: ',C

f_3 = open('cluster_keepmerging_2.txt', 'w')
for element in cluster.items():
    f_3.write(str(element))
    f_3.write('\n')
f_3.close()

f_4 = open('encode_keepmerging_2.txt', 'w')
for element in word_string.items():
    f_4.write(str(element))
    f_4.write('\n')
f_4.close()
