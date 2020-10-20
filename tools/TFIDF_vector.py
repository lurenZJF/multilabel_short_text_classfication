from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import pickle


def feature_vector(sentences: list, dimension=-1):
    """
    用TF——IDF来表征数据
    :param sentences: 文本分词后的形式
    :param dimension: PCA降维后的维度;-1时表示不进行降维处理
    """
    # 词频矩阵 Frequency Matrix Of Words
    # sublinear_tf,是否应用子线性tf缩放，即用1 + log（tf）替换tf；
    # max_df：float in range [ 0.0，1.0 ]或int，default = 1.0；当构建词汇时，忽略文档频率严格高于给定阈值（语料库特定停止词）的术语。
    vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
    transformer = TfidfTransformer()
    # Fit Raw Documents
    freq_words_matrix = vertorizer.fit_transform(sentences)
    # Get Words Of Bag
    # words = vertorizer.get_feature_names()
    # tfidf = transformer.fit_transform(freq_words_matrix)
    # w[i][j] represents word j's weight in text class i
    weight = freq_words_matrix.toarray()
    # 将词频矩阵降维
    if dimension > 0:
        pca = PCA(n_components=dimension)
        training_data = pca.fit_transform(weight)
        return training_data


class TF_IDF_VECTOR:
    def __int__(self):
        self.num = 1

    @staticmethod
    def save_model(model, path):
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def train_count(sentences: list):
        """
        根据训练集训练TF-IDF模型,并保存模型
        :param sentences: 训练集分词数据
        :param path: TF_IDF模型存储路径
        :return: 训练集的TF_IDF向量
        """
        count_vect = CountVectorizer()
        train_counts = count_vect.fit_transform(sentences)
        tfidf_transformer = TfidfTransformer()
        train_tfidf = tfidf_transformer.fit_transform(train_counts)
        return train_tfidf, count_vect

    def generate_vector(self, sentences: list, path, dimension=-1):
        """
        加载TF_IDF模型，并生成传入的sentences的TF_IDF表示
        :param sentences:分词列表
        :param path:模型路径
        :param dension:默认-1，表示不进行降维处理，贝叶斯方法不能进行降维；
        :return:TF_IDF向量
        """
        model = self.load_model(path)
        vector = model.transform(sentences)
        if dimension == -1:
            return vector
        else:
            pca = PCA(n_components=dimension)
            vector = vector.toarray()
            vector = pca.fit_transform(vector)
            return vector





