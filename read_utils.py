import xlrd
import pymysql
from config import Config
from gensim.models import word2vec

def get_excel_libs(excel_file):
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

def load_word2vec_model(model_file):
    model = word2vec.Word2Vec.load(model_file)
    return model




class MysqlThoth2():
    """mysql 操作类，和MysqlThoth区别是不经过ORM框架"""
    def __init__(self):
        config = Config()
        self.db = pymysql.connect(host=config.MYSQL_SETTINGS['host'],
                           port=config.MYSQL_SETTINGS['port'],
                           user=config.MYSQL_SETTINGS['username'],
                           password=config.MYSQL_SETTINGS['password'],
                           database=config.MYSQL_SETTINGS['db'],
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

