'''
用项目方法进行word2vec 和 glove 对比测试。
结果：word2vec 直观感觉更优
'''
import xlwt

from gensim import matutils
from numpy import dot

from libs.model_build_glove import get_vec_sen, get_vec_sen_list
from libs.read_utils import get_libs,get_txt_libs,get_QAs,get_excel_QAs,load_word2vec_model


def get_similar_index(vec1, vec_list, topn=10):  # 默认输出10个最相似的标题 的（索引号,相似度）列表
    try:
        dists = dot(vec_list, vec1)
        topn_idex = matutils.argsort(dists, topn=topn, reverse=True)
        topn_tuple = [(idex, dists[idex]) for idex in topn_idex]
        return topn_tuple
    except:
        print(' calculate dot error ! ')

if __name__=="__main__":
    # load glove modle...
    from gensim.models.keyedvectors import KeyedVectors
    from gensim.test.utils import get_tmpfile
    from gensim.scripts.glove2word2vec import glove2word2vec

    # glove_file = 'data/wiki.zh.jian.word_200v_glove.txt'
    glove_file = 'data/embs.txt'   # 将深度学习训练产生的embedding语义向量进行匹配运算
    word2vec_file = get_tmpfile("word2vec.format.vec")
    glove2word2vec(glove_file, word2vec_file)
    model_glove = KeyedVectors.load_word2vec_format(word2vec_file)

    model_zh = load_word2vec_model('models/wiki.zh.word_200v.model')

    models2 = [model_glove,model_zh]
    model_size = 200

    responses = get_libs('data/tianlong_libs.xlsx')
    QAs = get_excel_QAs('data/去除2和null.xlsx', 0)
    # responses = get_txt_libs('data/titles_text.txt')
    # QAs = get_QAs('data/unique_text.txt')

    responses_vec_list2 = get_vec_sen_list(responses, models2, model_size)

    outputbook = xlwt.Workbook()
    oh = outputbook.add_sheet('sheet1', cell_overwrite_ok=True)
    i,k,p = 0,0,0

    # while True:
        # query = input('you:')
    for query,r,y in QAs:
        query_vec2 = get_vec_sen(query, models2, model_size)

        topn_tuple = get_similar_index(query_vec2, responses_vec_list2, 10)
        topn_responses = [(responses[index],score) for index, score in topn_tuple]
        # print(topn_responses)
        topn_dict = dict(topn_responses)

        oh.write(i, 0, query)
        oh.write(i, 1, r)
        j=0
        for rs,sc in topn_responses:
            oh.write(i, 2+j, rs)
            oh.write(i, 2+j+1, '%.3f'%sc)
            j+=2

        if r == topn_responses[0][0]:
            k+=1
            print(k,'/',i)
        if r in topn_dict:
            p+=1
        i+=1
        if i >20000:break
    print('acc:', k/float(i), p/float(i))
    outputbook.save('embedding_results_old.xls')



