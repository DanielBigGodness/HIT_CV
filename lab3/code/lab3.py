import cv2
import glob
import re
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import *
from sklearn import preprocessing
from sklearn.svm import SVC
import joblib

label_name = ["cat", "lynx", "wolf", "coyote", "cheetah", "jaguer", "chimpanzee", "orangutan", "hamster", "guinea pig"]


#要求一：SIFT特征提取
def calcSiftFeature(img):
    sift = cv2.SIFT_create()
    _, features = sift.detectAndCompute(img, None)
    return features


def path_to_label(path, mode):
    # 如果模式是训练
    if mode == "train":
        # 定义一个正则表达式，匹配训练图片的路径前缀
        re_match_1 = r"C:\\Users\\86139\\Desktop\\computer_vision_class\\lab\\lab3\\code\\raw_image_ver\\raw_image\\training\\"
        # 定义一个正则表达式，匹配图片的文件名后缀
        re_match_2 = r"_img_\d*.jpg"
        # 用空字符串替换路径中的前缀和后缀，得到图片的标签
        string = re.sub(re_match_2, '', re.sub(re_match_1, '', path))
        # 返回图片的标签
        return string
    # 如果模式是测试
    if mode == "test":
        # 定义一个正则表达式，匹配测试图片的路径前缀
        re_match_1 = r"C:\\Users\\86139\\Desktop\\computer_vision_class\\lab\\lab3\\code\\raw_image_ver\\raw_image\\testing\\"
        # 定义一个正则表达式，匹配图片的文件名后缀
        re_match_2 = r"_img_\d*.jpg"
        # 用空字符串替换路径中的前缀和后缀，得到图片的标签
        string = re.sub(re_match_2, '', re.sub(re_match_1, '', path))
        # 返回图片的标签
        return string
    # 如果模式不是训练或测试，返回None
    else:
        return None


# 定义一个函数，根据图片的标签编号返回图片的标签名称
def label_no2name(no):
    # 从全局变量label_name中获取图片的标签名称
    return label_name[no]


# 定义一个函数，根据图片的路径和模式（训练或测试）创建数据集
def dataset_create(path, mode):
    # 获取路径下的所有图片的路径
    all_path = glob.glob(path)
    # 定义一个空列表，存储所有图片的标签
    all_label = []
    # 定义一个空列表，存储所有图片的路径
    all_img = []
    # 定义一个空列表，存储所有图片的特征描述
    all_des = []
    # 遍历所有图片的路径
    for i in tqdm(range(len(all_path))):
        # 获取当前图片的路径
        tmp = all_path[i]
        # 读取当前图片
        img = cv2.imread(tmp)
        # 调用path_to_label函数，获取当前图片的标签
        label = path_to_label(tmp, mode)
        # 调用calcSiftFeature函数，计算当前图片的SIFT特征
        fes = calcSiftFeature(img)
        # 如果当前图片的特征不为空
        if fes is not None:
            # 将当前图片的路径添加到all_img列表中
            all_img.append(tmp)
            # 将当前图片的特征描述添加到all_des列表中
            all_des.append(fes)
            # 将当前图片的标签添加到all_label列表中
            all_label.append(int(label))
    # 返回所有图片的路径，标签和特征描述
    return all_img, all_label, all_des


# 定义一个函数，根据图片的特征描述列表，获取所有图片的特征
def get_all_features(des_list):
    # 定义一个空列表，存储所有图片的特征
    outputs = []
    # 遍历图片的特征描述列表
    for i in range(len(des_list)):
        # 遍历每张图片的特征描述
        for one_des in des_list[i]:
            # 将每个特征描述添加到outputs列表中
            outputs.append(one_des)
    # 返回所有图片的特征
    return outputs


# 定义一个函数，根据图片的特征描述列表和聚类中心，将图片转换为向量表示
def images2vec(des_list, centers):
    # 创建一个零矩阵，大小为图片数量乘以聚类中心数量，数据类型为float32，用于存储所有图片的向量表示
    vecs = np.zeros((len(des_list), len(centers)), "float32")
    # 遍历图片的特征描述列表
    for i in range(len(des_list)):
        # 调用vq函数，根据聚类中心，将每张图片的特征描述分配到最近的聚类中心，并返回对应的索引和距离
        words, _ = vq(des_list[i], centers)
        # 遍历每个索引
        for w in words:
            # 在对应的位置上，将向量表示的值加一
            vecs[i][w] += 1
    # 返回所有图片的向量表示
    return vecs


# 定义一个函数，根据图片的向量表示，进行TF-IDF向量化
def tf_idf_vectorization(vecs):
    # 计算每个聚类中心在所有图片中出现的次数
    nbr_occurences = np.sum((vecs > 0) * 1, axis=0)
    # 计算每个聚类中心的逆文档频率（IDF）
    idf = np.array(np.log((1.0 * len(vecs) + 1) / (1.0 * nbr_occurences + 1)), 'float32')
    # 将每个图片的向量表示乘以对应的IDF
    vecs = vecs * idf
    # 对每个图片的向量表示进行L2归一化
    vecs = preprocessing.normalize(vecs, norm='l2')
    # 返回TF-IDF向量化后的图片向量表示
    return vecs


#要求四：SVM支持向量机
def train_svm(train_x, train_y, c):
    svm_model = SVC(C=c)
    svm_model.fit(train_x, train_y)
    return svm_model


if __name__ == "__main__":

    # train_path = r'C:\Users\86139\Desktop\computer_vision_class\lab\lab3\code\raw_image_ver\raw_image\training\*.jpg'
    # print("开始读取训练集数据：")
    # train_img_path, train_labels, train_des = dataset_create(train_path, mode="train")
    #
    # print("\n整合特征到同一列表下：")
    # all_features = get_all_features(train_des)
    # print("训练集共有特征数目：", len(all_features))
    # print("每条特征的长度为：", len(all_features[0]))
    # print("摘取第一条特征结果：", all_features[0])
    #
    # print("\n开始特征聚类：")
    # num_word = 200
    # voc, variance = kmeans(all_features, num_word, iter=1)
    # print(f"获得的聚类中心的变量：({len(voc)}, {len(voc[0])})")
    # print("打印聚类中心内容：")
    # print(voc)
    # print("存储词袋模型：")
    # bow_model = np.array(voc)
    # np.save('bow_model.npy', bow_model)
    # print("存储成功！")
    #
    # print("\n开始图片向量化：")
    # train_vec = images2vec(train_des, voc)
    # print(f"图像向量化后的输入矩阵大小为：({len(train_vec)}, {len(train_vec[0])})")
    # print(train_vec[0])
    #
    # print("\n开始向量规范化：")
    # train_vec_norm = tf_idf_vectorization(train_vec)
    # print(f"向量规范化后的矩阵大小为：({len(train_vec_norm)}, {len(train_vec_norm[0])})")
    # print(train_vec_norm[0])
    #
    # print("\n保存训练可用的数据")
    # train_X = np.array(train_vec_norm)
    # train_Y = np.array(train_labels)
    # np.save('svm_train_inputs.npy', train_X)
    # np.save('svm_train_labels.npy', train_Y)
    # print("保存成功")
    """

    r"""
    # train_X = np.load('svm_train_inputs.npy')
    # train_Y = np.load('svm_train_labels.npy')
    # print("训练集输入数据大小：", train_X.shape)
    # print("训练集标签数据大小：", train_Y.shape)
    #
    # print("\n开始训练分类器：")
    # model = train_svm(train_X, train_Y, 1)
    # print("训练结束！")
    # total = len(train_Y)
    # train_predict = model.predict(train_X)
    # correct = 0
    # for i in range(total):
    #     if train_predict[i] == train_Y[i]:
    #         correct += 1
    # acc = correct * 100 / total
    # print("whole acc: {}%".format(acc))
    # print("保存模型！")
    # joblib.dump(model, "svm_for_animals")
    # print("保存成功！")



    # test_path = r'C:\Users\86139\Desktop\computer_vision_class\lab\lab3\code\raw_image_ver\raw_image\testing\*.jpg'
    # print("开始读取测试集数据：")
    # test_img_path, test_labels, test_des = dataset_create(test_path, mode="test")
    # voc = np.load('bow_model.npy')
    #
    # print("\n开始图片向量化：")
    # test_vec = images2vec(test_des, voc)
    # print(f"图像向量化后的输入矩阵大小为：({len(test_vec)}, {len(test_vec[0])})")
    #
    # print("\n开始向量规范化：")
    # test_vec_norm = tf_idf_vectorization(test_vec)
    # print(f"向量规范化后的矩阵大小为：({len(test_vec_norm)}, {len(test_vec_norm[0])})")
    #
    # print("\n保存训练可用的数据")
    # test_X = np.array(test_vec_norm)
    # test_Y = np.array(test_labels)
    # np.save('svm_test_inputs.npy', test_X)
    # np.save('svm_test_labels.npy', test_Y)

    print("\n开始预测：")
    test_X = np.load('svm_test_inputs.npy')
    test_Y = np.load('svm_test_labels.npy')
    model = joblib.load('svm_for_animals')
    pre = model.predict(test_X)
    total = np.zeros((10, 1), dtype=float)
    correct = np.zeros((10, 1), dtype=float)
    for i in range(len(test_Y)):
        if pre[i] == test_Y[i]:
            correct[pre[i]] += 1
        total[test_Y[i]] += 1
    acc = correct * 100 / total
    print("各类预测准确率：")
    for i in range(len(acc)):
        if i % 2 == 0:
            print(f"{label_name[i]}: {acc[i]}%")

