import os
import re
import jieba
import pickle
import xlrd,xlwt
import pymysql
import random
import numpy as np
from gensim.models import word2vec
from gensim import matutils

MYSQL_SETTINGS = {
    'db': 'test_thoth_data',
    'host': '10.127.138.2',
    'port': 3306,
    'username': 'app_cloudcs_im',
    'password': 'app_cloudcs_1',
    'table_1': 'tbl_unknown_questions',
    'talbe_2': 'tbl_unknown_dates',
}

# jieba.load_userdict('data/userdicts_all.txt')
jieba.load_userdict('data/vocab_label.txt')

def get_txt_libs(file):
    libs = []
    with open(file,'r') as f:
        for line in f:
            line= line.strip()
            libs.append(line)
        return libs

def get_libs(excel_file):
    '''获取excel文件响应库'''
    libs = []
    inputbook = xlrd.open_workbook(excel_file)
    excel_sheet = inputbook.sheet_by_index(0)
    n = excel_sheet.nrows
    print("库问题数量:%s" % n)
    for i in range(1, n):
        row_data = excel_sheet.row_values(i)
        response = str(row_data[0])
        libs.append(response)
    return libs

def get_QAs(text_file):
    QAs = []
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            ls = line.strip().split('\t:\t')  # 旧：\t:\t   新：\t+\t
            if ls[-1]=='1':  #满意
                ls[-1]=1
                QAs.append(ls)
    print('样本数量：',len(QAs))
    return QAs

def get_excel_QAs(excel_file, sheet_i):
    '''获取excel文件问答对'''
    QAs = []
    inputbook = xlrd.open_workbook(excel_file)
    excel_sheet = inputbook.sheet_by_index(sheet_i)
    n = excel_sheet.nrows
    print("问答对数量:%s" % n)
    for i in range(1, n):
        row_data = excel_sheet.row_values(i)
        # id = str(row_data[0])
        query = str(row_data[1])
        response = str(row_data[2])
        label = 0 if row_data[4]==2.0 else 1
        QAs.append((query,response,label))
    return QAs

def seg_sentence(sentence):
    ws = jieba.cut(sentence)
    ws = list(ws)
    return ws

def load_word2vec_model(model_file):
    model = word2vec.Word2Vec.load(model_file)
    return model



class MysqlThoth2():
    """mysql 操作类，和MysqlThoth区别是不经过ORM框架"""
    def __init__(self):
        self.db = pymysql.connect(host=MYSQL_SETTINGS['host'],
                           port=MYSQL_SETTINGS['port'],
                           user=MYSQL_SETTINGS['username'],
                           password=MYSQL_SETTINGS['password'],
                           database=MYSQL_SETTINGS['db'],
                           charset='utf8')
    def __del__(self):
        self.db.close()

    def create_tbl(self,sql):
        cursor = self.db.cursor()
        cursor.execute(sql)

    def select_from_mysql(self, sql):
        cursor = self.db.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return results

    def insert_into_mysql(self,sql):
        cursor = self.db.cursor()
        try:
            # 执行sql语句
            cursor.execute(sql)
            new_id = cursor.lastrowid  # 获取新增数据的自增ID
            self.db.commit()
        except:
            # 发生错误时回滚
            self.db.rollback()
            new_id = None
        return new_id

    def update_to_mysql(self,sql):
        cursor = self.db.cursor()
        try:
            # 执行sql语句
            cursor.execute(sql)
            self.db.commit()
        except:
            # 发生错误时回滚
            self.db.rollback()

class TextConverter(object):
    def __init__(self, QAs=None, save_dir=None, max_vocab=5000 , seq_length = 20):
        if os.path.exists(save_dir):
            with open(save_dir, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab_dict={}
            for qas in QAs:
                text = re.sub(r'[^\w]+', ' ', qas[0])  # 句子清洗
                text = seg_sentence(text)
                ws = [w for w in text if w != ' ']  # 去除空格
                for w in ws:
                    if w in vocab_dict:
                        vocab_dict[w] += 1
                    else:vocab_dict[w] = 1
            vocab_tuple = sorted(vocab_dict.items(), key=lambda x:x[1], reverse=True )
            vocab = [w for w,n in vocab_tuple[:max_vocab]]

            self.vocab = vocab
            with open(save_dir, 'wb') as f:
                pickle.dump(self.vocab, f)
            with open(save_dir[:-4]+'.txt', 'w', encoding='utf8') as f:
                for w in self.vocab:
                    f.write(w + ' %d\n' % 100)


        self.seq_length = seq_length  # 样本序列最大长度
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def save_to_file(self, filename, var):
        with open(filename, 'wb') as f:
            pickle.dump(var, f)

    def text_to_arr(self, text):
        arr = []
        last_num = len(self.vocab)

        text = re.sub(r'[^\w]+',' ',text)  #句子清洗
        text = seg_sentence(text)
        text = [w for w in text if w != ' ']  #去除空格
        query_len = len(text)
        for word in text:
            arr.append(self.word_to_int(word))

        # padding
        if query_len < self.seq_length:
            arr += [last_num] * (self.seq_length - query_len)
        else:
            arr = arr[:self.seq_length]
            query_len = self.seq_length

        return np.array(arr), np.array(query_len)

    def QAs_to_arr(self, QAs):
        QA_arrs = []
        for query, response, label in QAs:
            # text to arr
            query_arr,query_len = self.text_to_arr(query)
            response_arr,response_len = self.text_to_arr(response)
            QA_arrs.append([query_arr,query_len,response_arr,response_len, float(label)])
        return QA_arrs

    def libs_to_arrs(self,libs):
        libs_arrs = []
        for response in libs:
            response_arr,response_len = self.text_to_arr(response)
            libs_arrs.append([response_arr,response_len])
        return libs_arrs

    def batch_generator(self,QA_arrs, batchsize, neg_flag=0):
        '''产生训练batch样本'''
        n_samples = len(QA_arrs)
        n_batches = int(n_samples / batchsize)
        n = n_batches * batchsize
        while True:
            random.shuffle(QA_arrs)  # 打乱顺序
            for i in range(0, n, batchsize):
                batch_samples = QA_arrs[i:i + batchsize]
                batch_q = []
                batch_q_len = []
                batch_r = []
                batch_r_len = []
                batch_y = []
                ln = len(batch_samples)
                for j in range(ln):
                    sample = batch_samples[j]
                # for sample in batch_samples:
                    batch_q.append(sample[0])
                    batch_q_len.append(sample[1])
                    batch_r.append(sample[2])
                    batch_r_len.append(sample[3])
                    batch_y.append(sample[4])
                    if neg_flag==1: # 针对全正样本情况，进行负样本分配
                        k = ln-1-j
                        neg_sample = batch_samples[k]
                        batch_q.append(neg_sample[0])
                        batch_q_len.append(neg_sample[1])
                        batch_r.append(sample[2])
                        batch_r_len.append(sample[3])
                        batch_y.append(float(0))


                yield np.array(batch_q), np.array(batch_q_len), np.array(batch_r), np.array(batch_r_len), np.array(batch_y)


    def val_samples_generator(self,QA_arrs, batchsize=500, neg_flag=0):
        '''产生验证样本，batchsize分批验证，减少运行内存'''

        val_g = []
        n = len(QA_arrs)
        for i in range(0, n, batchsize):
            batch_samples = QA_arrs[i:i + batchsize]
            batch_q = []
            batch_q_len = []
            batch_r = []
            batch_r_len = []
            batch_y = []
            ln = len(batch_samples)
            for j in range(ln):
                sample = batch_samples[j]
                # for sample in batch_samples:
                batch_q.append(sample[0])
                batch_q_len.append(sample[1])
                batch_r.append(sample[2])
                batch_r_len.append(sample[3])
                batch_y.append(sample[4])
                if neg_flag == 1:  # 针对全正样本情况，进行负样本分配
                    k = ln - 1 - j
                    neg_sample = batch_samples[k]
                    batch_q.append(neg_sample[0])
                    batch_q_len.append(neg_sample[1])
                    batch_r.append(sample[2])
                    batch_r_len.append(sample[3])
                    batch_y.append(float(0))
            val_g.append((np.array(batch_q), np.array(batch_q_len), np.array(batch_r), np.array(batch_r_len), np.array(batch_y)))
        return val_g


    def index_to_QA_and_save(self,indexs,QAs, path):
        print("start writing to eccel... ")
        outputbook = xlwt.Workbook()
        oh = outputbook.add_sheet('sheet1',cell_overwrite_ok=True)
        for index_q, index_r in indexs:
            que = QAs[index_q][0]
            oh.write(index_q, 0, que)
            k = 0
            for r_i in list(index_r):
                res = QAs[r_i][1]
                oh.write(index_q, 2+k, res)
                k += 1
                if k > 5:
                    break
        outputbook.save(path+'_Q_for_QA.xls')
        print('finished!')

    def index_to_response(self, index_list, libs):
        responses = []
        for index in index_list:
            responses.append(libs[index])
        return responses
    def index_to_response2(self, index_list, QAs):
        responses = []
        for index in index_list:
            responses.append(QAs[index][-1])
        return responses

    def save_to_excel(self, QAY, path):
        '''result to save...'''
        outputbook = xlwt.Workbook()
        oh = outputbook.add_sheet('sheet1', cell_overwrite_ok=True)
        k = 0
        for query, y_response, responses ,topn_cos in QAY:
            oh.write(k, 0, query)
            oh.write(k, 1, y_response)
            i = 0
            for response in responses:
                oh.write(k, 2 + i, str(response))
                i += 1
            cos = ' /'.join([str(a) for a in topn_cos])
            oh.write(k, 2+i, cos)
            k += 1
        outputbook.save(path )
        print('finished!')

    def save_embedding(self,word_model_txt, np_embs):
        w_f = open(word_model_txt, 'w',encoding='utf8')
        for i in range(np_embs.shape[0]):
            word = self.int_to_word(i)
            if word == ' ':continue
            w_f.write(word)
            unitvec = matutils.unitvec(np_embs[i])
            for num in unitvec:
                w_f.write(' %.6f'% num)
            w_f.write('\n')
        w_f.close()




# import jieba.analyse
# speech = '5秒注册  送2888武圣卡,.在"哪领'
# speech = re.sub(r'[^\w]+',' ',speech)
# print(' '.join(jieba.cut(speech)))
# print(jieba.analyse.textrank(speech, withWeight=True, topK=20))
# print(jieba.analyse.extract_tags(speech, topK=20, withWeight=True))

