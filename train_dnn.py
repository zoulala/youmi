import os
import tensorflow as tf
from libs.read_utils import TextConverter, get_QAs
# from deeplearning.Dual_LSTM_old import Model,Config
# from deeplearning.Dot_LSTM import Model,Config
# from deeplearning.Dense_LSTM import Model,Config
from deeplearning.CNN import Model,Config
neg_flag = 1  # 深度方法是否需要负样本

def main(_):

    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    vocab_file = os.path.join(model_path, 'vocab_label.pkl')

    # 获取原始数据
    input_file = 'data/unique_text.txt'
    QAs = get_QAs(input_file)

    # 分配训练和验证数据集
    thres = int(0.9*len(QAs))
    train_QAs = QAs[:thres]
    val_QAs = QAs[thres:]

    # 数据处理
    converter = TextConverter(train_QAs, vocab_file, max_vocab=Config.vocab_max_size, seq_length=Config.seq_length)
    print('vocab size:',converter.vocab_size)

    # 产生训练样本
    train_QA_arrs = converter.QAs_to_arr(train_QAs)
    train_g = converter.batch_generator(train_QA_arrs, Config.batch_size, neg_flag)

    # 产生验证样本
    val_QA_arrs = converter.QAs_to_arr(val_QAs)
    val_g = converter.val_samples_generator(val_QA_arrs, Config.batch_size, neg_flag)

    # 加载上一次保存的模型
    model = Model(Config,converter.vocab_size)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    print('start to training...')
    model.train(train_g, model_path, val_g)
    # converter.save_embedding(model_path+'/embs.txt', model.save_embeds())


if __name__ == '__main__':
    tf.app.run()
