# Brown-Word-Clustering-and-word-similarity

## Brown Clustering is a hierarchical clustering algorithm. This is a python based implementation of Brown Clustering, and prefix-based word similarity measurement. 

### Reference
The reference of the implementation includes:
[1] Semi-Supervised Learning for Natural Language, Percy Liang. http://cs.stanford.edu/~pliang/papers/meng-thesis.pdf
[2]	Class-based n-gram models of natural language, Peter F. Brown Vincent J. Della Pietra Peter V. deSouza Jenifer C. Lai Robert L. Mercer. 
https://pdfs.semanticscholar.org/7834/80acff435bfbc15ffcdb4f15eccddaa0c810.pdf
[3]	Public course: https://www.youtube.com/playlist?list=PLO9y7hOkmmSEAqCc0wrNB

Also, I use some of the following implementations as my reference:
[4]	https://github.com/percyliang/brown-cluster
[5]	https://github.com/xueguangl23/brownClustering

### File dictionary
1.	brown_clustering.py: implementation of brown clustering. 
2.	similarity_compute.py: implementation of measuring similarity. 
3.	binary_representation_padding.py: This code is to pad the binary representation of words to make them be in the same length. Here I add ‘0’s for padding.

4.	wordlist_decreasing.txt: word list sorted by decreasing frequency.
5.	brown_corpus_afterUNK.txt: corpus after process, specifically, with START and END, remove symbols, non-alphas and replace UNK.
6.	cluster_before_keep_merging.txt: After iterations(no remaining words and there are 200 clusters), the 200 clusters and corresponding words.
7.	cluster_keepmerging.txt: After constructing the full hierarchy, the final results. Now there is only one cluster left.
8.	encode_before_keepmerging.txt: Word binary representations after first iterations(there are 200 clusters).
9.	encode_keepmerging.txt: Word binary representations after creating full hierarchy.
10.	brownclusters.txt: word binary representation after brown clustering and padding.
11.	Results-brown.txt: The top 10 similar words of ‘the’, ‘army’, ’received’ and ‘famous’.

### Run the code
1.	For brown_clustering.py, you can run the code directly without setting any parameters. It may take quite a long time to run the code. Also, the number of clusters K is 200.
2.	For similarity_compute.py, you can run the code directly without setting any parameters. Also, you can change the other words to see the top 10 similar words.
3.	For binary_representation_padding.py, you don’t need to change any parameters. Here the input is encode_keepmerging.txt, and output is binary_representation.txt. This code is to pad binary expressions to be in the same length.

### High-level interpretation
1.	brown_clustering.py

Basically it has 3 steps:
	Read and process the data
	Build a Bigram tokens table, for computation convenience. Here Bigram table is a dictionary. eg: table[token2][token1] == count(token1,token2)
	Brown clustering.

Specifically, in the first step, the corpus is processed including tokenization, removing POS tags, lowering case all words, etc. After this process we have the processed text data and word list.

In the second step, a bigram token table is built. The main reason is that Brown Clustering is essentially is a bigram model, because when we compute transition probability we only need to consider TransitionP(token(i-1), token(i)). So here Bigram table can make it much convenient in the following computation so that we do not need to search all corpus during clustering. Also, Bigram table is dynamic.

The third step is core part. Here we calculate L instead of Quality in each loop, according to [1]. L(i,j) is the loss when merge cluster_i and cluster_j. In specific, in each loop I update L and decide which two clusters I want to merge. Then when merging these two clusters, the word encoding, cluster, remaining word, etc are updated. Also, L is update for the next loop. Last but important, when updating L, I don’t compute it start from scratch, instead I just update it with function Merged_L(), inspired by [5]. This makes computation much faster.

Also, see some more detailed explanations in the code.

Note: feel free to report any bugs or incorrections. Thank you!
