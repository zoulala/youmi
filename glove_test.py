'''
用项目方法进行word2vec 和 glove 对比测试。
结果：word2vec 直观感觉更优
'''

from numpy import dot
from gensim import matutils
from read_utils import get_excel_libs, load_word2vec_model
from  model_build import get_vec_sen, get_vec_sen_list


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
    models1 = [model_zh, model_en]
    model_size = 200

    # load glove modle...

    from gensim.models.keyedvectors import KeyedVectors
    from gensim.test.utils import get_tmpfile
    from gensim.scripts.glove2word2vec import glove2word2vec

    glove_file = 'data/wiki.zh.jian.word_200v.txt'
    word2vec_file = get_tmpfile("word2vec.format.vec")
    glove2word2vec(glove_file, word2vec_file)
    model_glove = KeyedVectors.load_word2vec_format(word2vec_file)
    models2 = [model_glove,]
    # model_size = 200

    responses = get_excel_libs('data/tianlong_libs.xlsx')
    responses_vec_list1 = get_vec_sen_list(responses, models1, model_size)
    responses_vec_list2 = get_vec_sen_list(responses, models2, model_size)
    while True:
        query = input('you:')
        query_vec1 = get_vec_sen(query,models1, model_size)
        query_vec2 = get_vec_sen(query, models2, model_size)

        topn_tuple = get_similar_index(query_vec1, responses_vec_list1, 10)
        topn_responses = [(responses[index],score) for index, score in topn_tuple]
        print(topn_responses)

        topn_tuple = get_similar_index(query_vec2, responses_vec_list2, 10)
        topn_responses = [(responses[index],score) for index, score in topn_tuple]
        print(topn_responses)

