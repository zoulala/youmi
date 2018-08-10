
"""
"""

from numpy import array,zeros, float32
from gensim import matutils
from sklearn import preprocessing
import re

def str2split(sentence):
    '''将句子按照中文、英文分开，如：‘我 is your朋友123’--> [‘我 ','is', 'your', '朋', '友',['1','2','3']]'''
    sens = []
    eword = ''
    nword = []

    sentence += ' '
    for uchar in sentence:
        if (uchar >= u'\u4e00' and uchar <= u'\u9fa5'):
            if eword:
                sens.append(eword)
                eword = ''
            if nword:
                sens.append(nword)
                nword = []
            sens.append(uchar)
        elif uchar == ' ':
            if eword:
                sens.append(eword)
                eword = ''
            if nword:
                sens.append(nword)
                nword = []
        elif uchar >= u'\u0030' and uchar<=u'\u0039':
            if eword:
                sens.append(eword)
                eword = ''
            nword.append(uchar)
        else:
            if nword:
                sens.append(nword)
                nword = []
            eword += uchar.lower()
    return sens

def trange(word):
    a = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    b = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    # b = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    if word in a:
        dex = a.index(word)
        return b[dex]
    else:
        return word

def get_vec_sen(sentence, models, size):
    '''获得：语义向量, 字向量求和'''
    sentence = sentence.strip()  # 去除前后空格及换行符
    sens = str2split(sentence)
    vector = zeros((size), dtype=float32)

    for w in sens:
        if isinstance(w, list):  # 对数字单独处理
            ln = len(w)
            vector_n = zeros((size), dtype=float32)
            for i in range(ln):
                wi = trange(w[i])
                for model in models:
                    if wi in model:
                        # vector_n += preprocessing.scale(model[wi])
                        vector_n += preprocessing.scale(model[wi])* (2 ** (ln - i - 1)) #*
                        # vector_n += model[wi] + model['零']*(ln-i-1)
                        break
            vector += vector_n  # preprocessing.scale(vector_n)  # 标准化
        else:
            if w in models[0]:
                vector += preprocessing.scale(models[0][w])  # 标准化
            elif w in models[1]:
                vector += preprocessing.scale(models[1][w])  # 标准化
            else:
                for wi in w:
                    if wi in models[-1]:
                        vector += preprocessing.scale(models[-1][wi])  # 标准化
    # vector = preprocessing.scale(vector)
    vector = matutils.unitvec(array(vector))  # 单位圆化：模为1
    return vector

def get_vec_sen_list(sen_list, models ,size):
    print('正在构建库向量...')
    vec_list = [get_vec_sen(sen, models, size) for sen in sen_list]
    print('构建完成。')
    return vec_list

r = re.compile(r'[\d]+')
def sen_num_list(sen):
    num_list = r.findall(sen)
    return num_list



if __name__=="__main__":
    v = get_vec_sen('no what AAA\n',123,123)