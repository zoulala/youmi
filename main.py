'''
输入一个问句，从候选库中找出与之相似的问题。
'''
from gensim import matutils
from numpy import dot

from libs.model_build import get_vec_sen, get_vec_sen_list
from libs.read_utils import get_libs, load_word2vec_model


def get_similar_index(vec1, vec_list, topn=10):  # 默认输出10个最相似的标题 的（索引号,相似度）列表
    try:
        dists = dot(vec_list, vec1)
        topn_idex = matutils.argsort(dists, topn=topn, reverse=True)
        topn_tuple = [(idex, dists[idex]) for idex in topn_idex]
        return topn_tuple
    except:
        print(' calculate dot error ! ')

if __name__=="__main__":
    # load word2vec modle...
    model_zh = load_word2vec_model('models/wiki.zh.word_200v.model')
    model_en = load_word2vec_model('models/wiki.en.word_200v.model')
    # model_cha = load_word2vec_model('models/wiki.en.char_200v.model')
    # models = [model_zh, model_en, model_cha]
    models = [model_zh, model_en]
    model_size = 200

    responses = get_libs('data/tianlong_libs.xlsx')
    responses_vec_list = get_vec_sen_list(responses, models, model_size)
    while True:
        query = input('you:')
        query_vec = get_vec_sen(query,models, model_size)

        topn_tuple = get_similar_index(query_vec, responses_vec_list, 10)
        topn_responses = [(responses[index],score) for index, score in topn_tuple]

        print(topn_responses)
