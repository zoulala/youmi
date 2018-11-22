'''
未知问题聚类
'''
import datetime

from gensim import matutils
from numpy import dot

from  libs.model_build import get_vec_sen, get_vec_sen_list
from libs.read_utils import load_word2vec_model, MysqlThoth2


def get_similar_index(vec1, vec_list, topn=10):  # 默认输出10个最相似的标题 的（索引号,相似度）列表
    try:
        dists = dot(vec_list, vec1)
    except:
        dists=[]
        print(' calculate dot error ! ')
    topn_idex = matutils.argsort(dists, topn=topn, reverse=True)
    topn_tuple = [(idex, dists[idex]) for idex in topn_idex]
    return topn_tuple

def insert_new_question(mysql_obj, question, robotId, tenantId):
    '''插入新问题'''
    # # 插入问题表
    sql_insert_q = "INSERT INTO tbl_unknown_questions(id,tenantId,robotId, question) VALUES (NULL, '%d', '%d', '%s')" % (
        tenantId, robotId, question)
    new_id = mysql_obj.insert_into_mysql(sql_insert_q)

    # 插入日期表
    time_a = datetime.datetime.now().strftime("%Y-%m-%d")
    sql_insert_d = "INSERT INTO tbl_unknown_dates(id,tenantId,robotId,questionId, date, count) VALUES (NULL,'%d','%d', '%d', '%s', '%d')" % (
        tenantId, robotId, new_id, time_a, 1)
    mysql_obj.insert_into_mysql(sql_insert_d)

def update_quesiton_count(mysql_obj, questionId, robotId,tenantId):

    sql_select = "SELECT * FROM tbl_unknown_dates WHERE questionId=%d and to_days(date) = to_days(now());" % questionId
    results = mysql_obj.select_from_mysql(sql_select)
    if results:
        sql_update = "UPDATE tbl_unknown_dates SET count=count+1 WHERE questionId=%d and to_days(date) = to_days(now());" % questionId
        mysql_obj.update_to_mysql(sql_update)
    else:
        # 插入日期表
        time_a = datetime.datetime.now().strftime("%Y-%m-%d")
        sql_insert_d = "INSERT INTO tbl_unknown_dates(id,tenantId,robotId,questionId, date, count) VALUES (NULL,'%d','%d', '%d', '%s', '%d')" % (
            tenantId, robotId, questionId, time_a, 1)
        mysql_obj.insert_into_mysql(sql_insert_d)


def load_old_qustions(mysql_obj, robotId):
        '''读取已有未知问题列表'''
        sql_select = "SELECT * FROM tbl_unknown_questions \
               WHERE robotId = %d " % (robotId)
        results = mysql_obj.select_from_mysql(sql_select)
        questionList = [(row[0], row[3]) for row in results]
        print(questionList)
        return questionList



if __name__=="__main__":
    # load word2vec modle...
    model_zh = load_word2vec_model('models/wiki.zh.word_200v.model')
    model_en = load_word2vec_model('models/wiki.en.word_200v.model')
    model_cha = load_word2vec_model('models/wiki.en.char_200v.model')
    models = [model_zh, model_en, model_cha]
    model_size = 200

    mysql_obj = MysqlThoth2()


    ## 删除表
    # sql_drop1 = "DROP TABLE tbl_unknown_questions ;"
    # sql_drop2 = "DROP TABLE tbl_unknown_dates ;"
    # mysql_obj.create_tbl(sql_drop1)
    # mysql_obj.create_tbl(sql_drop2)
    # #创建表
    # sql_create1 = "create table if not exists tbl_unknown_questions(id bigint unsigned not null auto_increment primary key,tenantId bigint,robotId bigint,question text(100));"
    # sql_create2 = "create table if not exists tbl_unknown_dates(id bigint unsigned not null auto_increment primary key,tenantId bigint, robotId bigint,questionId bigint,date date,count bigint, ignores int(11) default 0);"
    # mysql_obj.create_tbl(sql_create1)
    # mysql_obj.create_tbl(sql_create2)
    ## 增加字段
    # sql_add = "alter table tbl_unknown_dates add tenantId bigint DEFAULT NULL AFTER id"
    # mysql_obj.create_tbl(sql_add)
    ##删除字段
    # sql_drop = "alter table tbl_unknown_dates drop column ignore"
    # mysql_obj.create_tbl(sql_drop)
    ## 创建索引
    # sql_index1 = "ALTER TABLE tbl_unknown_questions ADD INDEX  (robotId)"
    # sql_index2 = ""
    # mysql_obj.create_tbl(sql_index1)

    while True:
        tenantId = 1008
        robotId = 36
        question = input('unknow:')
        # question = '畅游地址是多少'

        questionList = load_old_qustions(mysql_obj, robotId)
        print('questionList:',questionList)

        # 计算question归属类
        sen_list = [q for _,q in questionList]
        ids = [i_d for i_d, _ in questionList]
        # 转换为语义向量
        query_vec = get_vec_sen(question, models, model_size)
        vec_list = get_vec_sen_list(sen_list, models, model_size)

        topn_tuple = get_similar_index(query_vec, vec_list, 10)  # 默认输出10个最相似的标题 的（索引号,相似度）列表
        print(topn_tuple)
        if topn_tuple and topn_tuple[0][1] > 0.8:
            # 更新问题次数
            index = topn_tuple[0][0]
            q_id = ids[index]
            update_quesiton_count(mysql_obj, q_id, robotId,tenantId)
        else:
            # 插入新问题
            insert_new_question(mysql_obj, question, robotId, tenantId)




