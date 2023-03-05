### Hidden Markov Model Part-of-Speech Tagger

This POS tagger is based off of a stationary Markov emission model. The inference is done by [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm). Transition and emission matrices are naively found by using counting.

Run a demo:

```bash
python ./HMM.py -d ../data/pos/training1.txt ../data/pos/training2.txt ../data/pos/training3.txt -t ../data/pos/test4.txt -o ./test.txt
```

Average accuracy on the test files: 85%