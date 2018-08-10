import numpy as np
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
    model_cha = load_word2vec_model('models/wiki.en.char_200v.model')
    models = [model_zh, model_en, model_cha]
    model_size = 200

    # 获取某一类问题集合
    sen_dict = {'我的账号被盗了怎么办':1,'号被别人盗了如何解决':1,'我的号被别人盗了':1,'账号怎么被盗了':3}
    sen_list = ['我的账号被盗了怎么办','号被别人盗了如何解决','我的号被别人盗了','账号怎么被盗了','账号怎么被盗了','账号怎么被盗了']  # 读取某一类的问题
    # 转换为语义向量
    vec_list = get_vec_sen_list(sen_list, models, model_size)
    # 计算向量中心
    vec_center = np.sum(vec_list,axis=0)/len(vec_list)

    # 新问题
    query = '账号怎么被盗了'  # input('you:')
    print('新问题：',query)
    query_vec = get_vec_sen(query, models, model_size)

    # 计算新问题与所有问题的相似度
    for vec in vec_list:
        print('-相似度- ',np.dot(query_vec, vec), '-距离-',np.sqrt(np.sum(np.square(query_vec - vec))))

    # 计算新问题与类中心的相似度
    score = np.dot(query_vec, vec_center)
    print('\n与类中心的相似度:',score)

    # 计算新问题与类中心的距离
    dist = np.sqrt(np.sum(np.square(query_vec - vec_center)))
    print('\n与类中心的距离:',dist)

    # 更新类
    if score>0.65 and dist<0.6:
        sen_list.append(query)
        print('更新类：',sen_list)

    # 输出最能代表类的问题
    vec_list = get_vec_sen_list(sen_list, models, model_size)
    new_center = np.sum(vec_list,axis=0)/len(vec_list)

    topn_tuple = get_similar_index(new_center, vec_list, 10)
    topn_responses = [(sen_list[index], score) for index, score in topn_tuple]
    print('与类中心score排序: ',topn_responses)


