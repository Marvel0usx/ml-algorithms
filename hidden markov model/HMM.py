# Here is an POS tagger implemented based off the Hidden Markov Model.
# 
# There are two main aspects that I considered to improve the performance of my HMM.The first measure is to 
# add two extra POS tags, i.e. and , representing the beginning of a sentence and the end of a sentence. 
# Since all of the sentences starts with ,the prior for the HMM will be easily determined and it is stable 
# across all training data. Therefore, the model will know what tags tend to appear at the beginning of the
# sentence, and have more accurate predictions near the start of the sentence. Similarly, the tag at the end 
# gives the HMM more knowledge about what tags tend to appear at the end, and hence, increases the accuracy 
# towards the end of the sentence. The second concern is to give unseen token a equal emission probability. 
# When the model encounters an unseen token, we assume the conditional probability of seeing this token given 
# any POS tag is the same. In this way, the unseen token does not negatively affect the probability of the 
# most likely path. Besides, Numpy is also used to vectorize the code to boost the performance/speed of the 
# algorithm.
#
# Final accuracy: 85.27%


import sys  

import numpy as np  
from typing import Tuple, List  
from itertools import chain  
from collections import deque  
  
  
class StationaryDiscreteHMM:  
    """ 
    Implementation of a scalable stationary, discrete hidden markov model: 
    S_0 --> S_1 --> S_2 --> ... --> S_t --> S_t+1 
     |       |       |               |       | 
    E_0     E_1     E_2             E_t     E_t+1 
    States and observations are both discrete R.V., and are discrete in time. 
    Each state is associated with a transition probability and 
    an emission probability. 
    """  
  
    def __init__(self):  
        """ 
        Initializes an emission model of N stages. 
        emission: |E|x|S| 
        transition: |S|x|s| 
        """  
        self.transition = None  
        self.emission = None  
        self.S = None  
        self.E = None  
  
    def train(self, data: List[List[Tuple["state", "evidence"]]]):  
        """ 
        Simple training by counting. Should use Forward-Backward generally. 
        Calculates the transition probability and the emission probability 
        of each stage, i.e. the probability of stage S_t given S_t-1, and 
        the probability of observing E_t given S_t. 
        """  
        # Create index maps.  
        self.S = {k: v for v, k in enumerate(sorted(list(set([d[1] for d in chain(*data)]))))}  
        self.E = {k: v for v, k in enumerate(sorted(list(set([d[0] for d in chain(*data)]))))}  
        self.transition = np.zeros((len(self.S), len(self.S)), dtype=np.float64)  
        self.emission = np.zeros((len(self.E), len(self.S)), dtype=np.float64)  
  
        # Create counting table.  
        for sample in data:  
            for idx in range(len(sample)):  
                e, s = sample[idx]  
                self.emission[self.E[e], self.S[s]] += 1  
                if idx + 1 < len(sample):  
                    s_next = sample[idx + 1][1]  
                    self.transition[self.S[s], self.S[s_next]] += 1  
  
        # Calculate conditional probabilities, P(E|S) and P(S_t|S_t-1).  
        self.emission /= self.emission.sum(axis=0)  
        self.transition = (self.transition.T / self.transition.sum(axis=1)).T  
  
        self.emission[np.isnan(self.emission)] = 0.  
        self.transition[np.isnan(self.transition)] = 0.  
  
    def emission_prob(self, evidence):  
        if evidence in self.E:  
            return self.emission[self.E[evidence]]  
        else:  
            return np.ones((1, self.transition.shape[0]))[0] * 1 / self.transition.shape[0]  
  
    def viterbi(self, evidences: List["evidence"], prior):  
        """ 
        Finds the best assignment of S_0 ... S_t such that they maximize 
        the probability of observing e_0 ... e_t. 
        """  
        # Stores the highest probability of path.  
        prob = np.zeros((len(evidences), len(self.S)))  
        # Stores the highest possible previous state.  
        prev = np.zeros((len(evidences), len(self.S)))  
  
        # P(S_0|E_0) = c * Prior(S_0) * P(E_0|S_0) by bayes rule.  
        prob[0] = np.multiply(prior, self.emission_prob(evidences[0]))  
        prev[0] = -1  
  
        for i in range(1, len(evidences)):  
            for j in range(len(self.S)):  
                k = np.argmax(prob[i - 1] * self.transition[:, j] * self.emission_prob(evidences[i])[j])  
                prob[i, j] = prob[i - 1, k] * self.transition[k, j] * self.emission_prob(evidences[i])[j]  
                prev[i, j] = k  
  
        # Back trace the best states.  
        res = [np.argmax(prob[-1])]  
        for i in range(len(evidences) - 1, 0, -1):  
            res.append(int(prev[i, res[-1]]))  
  
        backward_map = {k: v for v, k in self.S.items()}  
        res = [backward_map[k] for k in reversed(res)]  
  
        return res  
  
  
def clean(fin):  
    data = deque([tuple(s.rstrip().split(" : ")) for s in fin.readlines()])  
    corpus = []  
    sentence = [("<CLS>", "CLS")]  
    while len(data):  
        token = data.popleft()  
        sentence.append(token)  
        if token[0] in (".", "?", "!"):  
            sentence.append(("<SEP>", "SEP"))  
            corpus.append(sentence)  
            sentence = [("<CLS>", "CLS")]  
    return corpus  
  
  
def tag_impl(training_list, testing_data, output_file):  
    train_corpus = []  
    for training_file in training_list:  
        with open(training_file, encoding="utf-8") as train_fin:  
            train_corpus += clean(train_fin)  
  
    with open(testing_data, encoding="utf-8") as test_fin:  
        test_corpus = clean(test_fin)  
  
    tagger = StationaryDiscreteHMM()  
    tagger.train(train_corpus)  
  
    prior = np.zeros(len(tagger.S))  
    prior[tagger.S["CLS"]] = 1.  
  
    res = []  
    for sentence in test_corpus:  
        sentence = [w[0] for w in sentence]  
        ret = tagger.viterbi(sentence, prior)  
        res.append(ret[1:-1])  
  
    # Output.  
    with open(output_file, "w", encoding="utf-8") as fout:  
        buf = ""  
        for i in range(len(test_corpus)):  
            for token, tag in zip(test_corpus[i][1:-1], res[i]):  
                buf += f"{token[0]} : {tag}\n"  
        fout.write(buf)  
  
  
def tag(training_list, test_file, output_file):  
    tag_impl(training_list, test_file, output_file)  
  
  
if __name__ == '__main__':  
    # Run the tagger function.  
    print("Starting the tagging process.")  
  
    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"  
    parameters = sys.argv  
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]  
    print(training_list)  
    test_file = parameters[parameters.index("-t")+1]  
    output_file = parameters[parameters.index("-o")+1]  
    # print("Training files: " + str(training_list))  
    # print("Test file: " + test_file)  
    # print("Output file: " + output_file)  
  
    # Start the training and tagging operation.  
    tag (training_list, test_file, output_file)
