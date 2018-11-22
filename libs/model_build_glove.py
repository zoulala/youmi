
"""
"""
from numpy import array,zeros, float32
from gensim import matutils
from sklearn import preprocessing
import re
import jieba
# jieba.load_userdict('data/userdicts_all.txt')
jieba.load_userdict('data/vocab_label.txt')



def seg_sentence(sentence):
    ws = jieba.cut(sentence)
    ws = list(ws)
    return ws

def get_vec_sen(sentence, models, size):
    '''获得：语义向量, 字向量求和'''
    text = re.sub(r'[^\w]+', ' ', sentence)  # 句子清洗
    text = seg_sentence(text)
    sens = [w for w in text if w != ' ']  # 去除空格

    vector = zeros((size), dtype=float32)
    for w in sens:
        if w in models[0]:
            vector += preprocessing.scale(models[0][w])  # 标准化
        else:
            for d in w:
                if d in models[1]:
                    vector += preprocessing.scale(models[1][d])*0.3  # 标准化
    vector = matutils.unitvec(array(vector))  # 单位圆化：模为1
    return vector

def get_vec_sen_list(sen_list, models ,size):
    print('正在构建库向量...')
    vec_list = [get_vec_sen(sen, models, size) for sen in sen_list]
    print('构建完成。')
    return vec_list

if __name__=="__main__":
    print(seg_sentence('天堑飞桥是什么？'))