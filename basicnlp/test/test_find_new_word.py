from src.read_data import read_data
from src.find_new_word import word_recognition


def find_new_word():
    path = '../data/zeng.txt'
    data = read_data(path)
    tag_dict = ['nr']
    word1 = (word_recognition(data, tag_dict, 'jieba'))
    word2 = (word_recognition(data, tag_dict, 'hanlp'))
    print(word1)
    print(word2)


find_new_word()