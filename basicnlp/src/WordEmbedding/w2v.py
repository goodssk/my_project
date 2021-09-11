import gensim.models.word2vec
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec, KeyedVectors
import numpy as np


class w2v:
    def __init__(self, path):

        self.model = KeyedVectors.load_word2vec_format(path, binary=True)
        self.keys = list(self.model.vocab.keys())
        self.unk_vec = np.mean([self.model[self.keys[i]] for i in range(0, len(self.keys), 200000)], axis=0)

    def similariy(self, word):
        if word not in self.model:
            word_vec = self.get_ngram_vec(word)
            self.model.add(word, word_vec)
        return self.model.similar_by_word(word, 100)

    def word_vec(self, word):
        if isinstance(word, str):
            if word in self.model:
                return self.model[word]
            else:
                return self.get_ngram_vec(word)

        if isinstance(word, list):
            vector = []
            for i in word:
                if i in self.model:
                    vector.append(self.model[i])
                if i not in self.model:
                    vector.append(self.get_ngram_vec(i))
            return vector

    def compute_ngrams(self, word: str, min_n: int, max_n: int):
        extended_word = word
        ngrams = []
        for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
            for i in range(0, len(extended_word) - ngram_length + 1):
                ngrams.append(extended_word[i:i + ngram_length])
        return list(set(ngrams))

    def get_ngram_vec(self, word: str):
        word_vec = []
        ngrams_word = self.compute_ngrams(word, 1, 3)
        ngrams_found = 0
        ngrams_single = [ng for ng in ngrams_word if len(ng) == 1]
        ngrams_more = [ng for ng in ngrams_word if len(ng) > 1]
        for ngram in ngrams_more:
            if ngram in self.model:
                word_vec.append(self.model[ngram])
                ngrams_found += 1
        if ngrams_found == 0:
            for ngram in ngrams_single:
                if ngram in self.model:
                    word_vec.append(self.model[ngram])
                    ngrams_found += 1
        if ngrams_found > 0:
            return np.mean(word_vec, axis=0)
        else:
            return self.unk_vec

path = "D:\project\model\word2vec/200w.bin"
w = w2v(path)
print(w.similariy("我爱龙祖贤"))
