import xlrd
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