from src.read_data import read_data
from src.find_new_word import word_recognition
from src.find_new_word import find_word
from src.TriplesExtraction.triple_extraction import TripleExtractor

def find_new_word():
    path = '../data/zeng.txt'
    data = read_data(path)
    tag_dict = ['nr']
    word1 = (word_recognition(data, tag_dict, 'jieba'))
    word2 = (word_recognition(data, tag_dict, 'hanlp'))
    print(word1)
    print(word2)


def find_new_word2():
    path = '../data/zeng.txt'
    data = read_data(path)
    print(data)
    t = TripleExtractor()
    for s in data:
        l = t.triples_main(s)
        print(l)

print('dada')
find_new_word2()