#coding: utf-8
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
import collections
import jieba.posseg as psg
from pyhanlp import *
import re
from numpy import log


def word_recognition(sentences, tag_dict, tool='jieba'):
    words = []
    for sentence in tqdm(sentences, desc='Extracting specific words '):
        if tool=='jieba':
            sentence = psg.cut(sentence)
        elif tool=='hanlp':
            sentence = HanLP.segment(sentence)
        sentence = [str(word).split('/') for word in sentence]
        sentence = list(filter(lambda x: x[1] in tag_dict, sentence))
        if not sentence:continue
        words.extend(np.array(sentence)[:, 0])
    words = collections.Counter(words)
    return words

def extrac_word():
    #hanlp = HanLPClient('https://www.hanlp.com/api', auth=None, language='zh')
    #content = '三十年前，我在整理出版近代湖湘历史文献的工作中，被一批来自湖南省图书馆的百年旧藏文稿所震惊：它不是一般的私家档案，也不是一般的历史资料，它是极不寻常、极其罕见、极为难得的文字中的珍品。'
    #content = '好嘞！”酒保高声答应，兴冲冲地走下楼去。很快便端上四大盘：一盘油焖香葱白豆腐，一盘红椒炒玉兰片，一盘茭瓜丝加捆鸡条，一盘新上市的娃娃菜，外加金针木耳蘑菇汤。红白青翠、飘香喷辣地摆在桌上。曾国藩喝着水酒，就着素菜，吃得很是香甜。喝完酒，酒保又端来两碗晶莹的大米饭，曾国藩吃得味道十足。不仅是这些日子，他仿佛觉得自从离开湖南以来，就再也没有吃过这么好的饭菜了。“还是家乡好哇！”曾国藩放下筷子，感慨地说。刚放下碗，酒保又殷勤地端来两碗热气腾腾的茶，说：“客官看来是远道而来，不瞒二位，这茶是用地道的君山毛尖泡的。”见曾国藩微笑地望着自己，酒保心中得意，“客官有所不知，君山上有五棵三百年的老茶树。当中一棵，是给皇上的贡茶，左右两边两棵是抚台大人和知府老爷送给亲戚朋友的礼品。左边第二棵是茶场老板的私用，右边第二棵则是小楼世代包下的。不是小的吹牛，这碗茶在京城，怕是出一百文也买不到。小楼规矩，每位客官用完饭后，奉送一碗地道的君山茶。”酒保边说边利索地收拾碗筷，擦干净桌面，下楼去了。曾国藩呷了一口茶，虽比不上京师买的上等毛尖，但也确实使人心脾清爽。他没有想到，破败的岳阳楼上却有这样好的饭菜和能说会道的酒保，心情舒畅多了。他端起茶碗，向窗外的湖面眺望。阳光照在湖水上，泛起点点金光。远处，一片片白帆在游弋，极目处，有一团淡淡的黑影。曾国藩知道，那就是君山。近处，沿湖岸停泊着一个接一个木排。这些木材大半出自湘南山区，扎成排后顺着湘江漂流，越过洞庭湖，进入长江，再远漂武昌、江宁、上海等地。放排的人叫作排客，排客们终年在水面漂浮，把家也安在排上。排上用杉树皮盖成小棚子，家眷就住在里面。曾国藩正颇有兴趣地看着楼下几个排上人家的生活，不料湖面陡然起风了，满天乌云翻滚，像要下雨的样子。刚才还是明镜般平静的湖面，顿时波浪翻卷。风越刮越大，波浪也越卷越高，湖面上的木排随着波浪在上下起伏，几个离岸边不远的木排在迅速向湖边靠拢。大雨哗哗而下，雨急风猛，温顺的洞庭湖霎时变成了一条狂暴的恶龙。曾国藩坐在楼上，浑身感到凉飕飕的。他有点担心，这座千年古楼，会不会被这场暴风雨击垮？正在胡思乱想之际，他看到离岸边约百来丈远的湖面上，一个小排被风浪打得左右摇晃，却一步也不能前进。一个汉子死死地扶着排后舵把，另一个汉子急得这边跑到那边。猛地一个大浪打来，木排上低矮的杉树皮屋垮了，一个木箱被水冲到湖里。两边跑的汉子纵身跳到水中去抓木箱。木排上一个十来岁的小女孩吓得蹲在排上，紧紧地抓着一根缆绳。一个四十余岁的妇人急得在排上前后乱窜。又一个大浪打过来，小女孩被卷进了湖中。“不得了！”曾国藩喊了一声，放下茶碗，猛地站起。荆七也赶紧站起，紧张地倚着窗口观望。正在这危急时刻，湖边木排上跳下一个年轻人，冒雨迎浪向湖中游去。只见那青年一个猛子扎入水底，刚好到排边又露出头来。他轻捷地游到手脚乱抓的小女孩身边，把她高高托出水面，游到排边。曾国藩到这时才舒了一口气。那青年上了木排，用手指指点点，排上的汉子拿来一大捆粗绳。青年接过绳子，走到排头，将绳子一头系在排上，另一头系在自己腰上，复跳入湖中，用自己一人之力在前面水中拉排。那木排居然跟着年轻人前进起来，湖边观看的人一齐喝彩。曾国藩被眼前这一幕惊呆了。木排缓缓地向岸边移动，平安地来到岳阳楼脚下。排上那两个汉子上得岸来，扶住年轻人，纳头便拜。曾国藩对那个年轻人见义勇为的品德和罕见的神力感慨不已，对荆七说：“你去请那位壮士来，我要见见他。'
    content = '曾国藩又亲自通过考核比较，从八十名亲兵中挑出彭毓橘、鲍超霆萧庆衍彭玉麟杨岳等六人来，由康福负责训练，黄老之术涌上心头充当自己的贴身保镖。这六个人都是曾国藩的亲'
    #key_word = HanLP.extractKeyword(content, 5)
    #extract = HanLP.extractSummary(content, 3)
    #phrase = HanLP.extractPhrase(content, 3)
    #parse = HanLP.parseDependency(content)
    word = HanLP.extractWords(content, 3, True)
    segment = HanLP.newSegment().enableNameRecognize(True)
    ner = segment.seg(content)
    print(ner)
    print(HanLP.segment(content))
    print(word)
    #print(key_word)
    #print(extract)
    #print(phrase)
    #print(parse)


class find_word:
    def __init__(self, path):
        self.path = path
        self.data = self.read_data()
        self.word_dict = collections.Counter(''.join(self.data))
        self.word_count = len(''.join(self.data))
        self.myre = {2: '(..)', 3: '(...)', 4: '(....)', 5: '(.....)', 6: '(......)', 7: '(.......)'}
        self.word_frequency = []
        self.nearby_word = []
        self.drop_dict = [u'，', u'\n', u'。', u'、', u'：', u'(', u')', u'[', u']', u'.', u',', u' ', u'\u3000', u'”', u'“', u'？', u'?', u'！', u'‘', u'’', u'…']
        self.max_length = 2
        self.min_count = 10
        self.min_value = 0
        self.data = ''.join(self.data)
        for i in self.drop_dict:
            self.data = self.data.replace(i, '')

    def sta_fre(self):
        self.word_frequency.append(pd.Series(list(self.data)).value_counts())
        print(self.word_frequency)

        for length in range(2, self.max_length+1):
            self.word_frequency.append([])
            for i in range(length):
                self.word_frequency[length-1] += re.findall(self.myre[length], self.data[i:])
            self.word_frequency[length-1] = pd.Series(self.word_frequency[length-1]).value_counts()
            self.word_frequency[length-1] = self.word_frequency[length-1][self.word_frequency[length-1]>self.min_count]

        print(self.word_frequency)

    ### 最后得到每个词的凝聚度，左熵和又熵
    ### 计算互信息
    def cal_MI(self):
        for length in range(2, self.max_length+1):
            for word in self.word_frequency[length-1]:
                PIN = []
                for i in range(len(word)-1):
                    unite_count = self.word_frequency[length-1][word]
                    left_count = self.word_frequency[i][word[:i]]
                    right_count = self.word_frequency[len(word)-1-i][word[i:]]
                    current_pin = unite_count/self.word_count*math.log(unite_count*self.word_count/(left_count*right_count))
                    PIN.append(current_pin)
                min(PIN)

    def cal_s(self, sl):
        return -((sl / sl.sum()).apply(log) * sl / sl.sum()).sum()

    def cal_left_entropy(self):
        for i in range(2, self.max_length+1):
            pp = []
            for j in range(i+2):
                pp += re.findall('(.)%s(.)' % self.myre[i], self.data[j:])
            pp = pd.DataFrame(pp).set_index(1).sort_index()
            index = pp.index
            pim_left = list(map(lambda s: self.cal_s(pd.Series(pp[0][s]).value_counts()), index))
            pp = pp[np.array(pim_left)>self.min_value]
            pim_left = np.array(pim_left)
            pim_left = pim_left[np.where(pim_left > self.min_value)]
            pp[3] = pim_left

            pim_right = list(map(lambda s: self.cal_s(pd.Series(pp[2][s]).value_counts()), pp.index))
            pp = pp[np.array(pim_right) > self.min_value]
            pim_right = np.array(pim_right)
            pim_right = pim_left[np.where(pim_right > self.min_value)]
            pp[4] = pim_right
            pp = pp[~pp.index.duplicated(keep="first")]
            pp.to_csv('data.csv')


    def read_data(self):
        with open(self.path, encoding='utf-8') as f:
            data = [line.strip() for line in f.readlines()]
        return data


if __name__ == '__main__':
    path = '../data/zeng.txt'
    new_word = find_word(path)
    new_word.cal_left_entropy()
# sentence = '怎么又生气了你是怎么了'
# for i
# a = re.findall('(...)', )
# print(a)