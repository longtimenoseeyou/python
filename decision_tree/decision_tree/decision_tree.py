
from numpy import *
from scipy import *
from pandas import *
from math import log
import operator
from sklearn.model_selection import train_test_split
import treePlotter
"""
加载训练数据和测试数据，样本中没有给出
因此按数据集中的要求切割出3133个样本作为训练集，剩下1044个样本作为测试集
使用sklearn库中的train_test_split来划分，每次产生的训练集都是随机的
得到的训练集和测试集是包含了标签的
"""

def load_data(train_size):
    row_data=read_excel('C:/Users/10102/Desktop/作业二的数据.xls')
    data_size = len(row_data)
    test_data_size = data_size - train_size
    train_data, test_data = train_test_split(row_data, test_size=test_data_size / data_size)  # 测试集所占的比例
    return train_data.values, test_data.values

"""
决策树的生成，data_set为训练集，attribute_label为属性名列表
决策树用字典结构表示，递归的生成
"""

def generate_decision_tree(data_set, attribute_label):
    label_list = [entry[-1] for entry in data_set]#获取到每条数据的标签
    if label_list.count(label_list[0]) == len(label_list):  # 如果所有的数据都属于同一个类别，则返回该类别
        return label_list[0]
    if len(data_set[0]) == 1:  # 如果数据没有属性值数据，则返回该其中出现最多的类别作为分类
        return most_voted_attribute(label_list)
    best_attribute_index, best_split_point = attribute_selection_method(data_set)#找到最佳划分属性索引
    best_attribute = attribute_label[best_attribute_index]
    decision_tree = {best_attribute: {}}
    del (attribute_label[best_attribute_index])  # 找到最佳划分属性后需要将其从属性名列表中删除
    """
    如果best_split_point为空，说明此时最佳划分属性的类型为离散值，否则为连续值
    """
    if best_split_point == None:
        attribute_list = [entry[best_attribute_index] for entry in data_set]
        attribute_set = set(attribute_list)
        for attribute in attribute_set:  # 属性的各个值
            sub_labels = attribute_label[:]
            decision_tree[best_attribute][attribute] = generate_decision_tree(
                split_data_set(data_set, best_attribute_index, attribute, continuous=False), sub_labels)
    else:
        """
        最佳划分属性类型为连续值，此时计算出的最佳划分点将数据集一分为二，划分字段取名为<=和>
        """
        sub_labels = attribute_label[:]
        decision_tree[best_attribute]["<=" + str(best_split_point)] = generate_decision_tree(
            split_data_set(data_set, best_attribute_index, best_split_point, True, 0), sub_labels)
        sub_labels = attribute_label[:]
        decision_tree[best_attribute][">" + str(best_split_point)] = generate_decision_tree(
            split_data_set(data_set, best_attribute_index, best_split_point, True, 1), sub_labels)
    return decision_tree

"""
通过信息增益比来计算最佳划分属性
属性分为离散值和连续值两种情况，分别对两种情况进行相应计算
"""

def attribute_selection_method(data_set):
    num_attributes = len(data_set[0]) - 1  # 属性的个数，减1是因为去掉了标签
    info_D = calc_info_D(data_set)  # 香农熵
    max_grian_rate = 0.0  # 最大信息增益比
    best_attribute_index = -1
    best_split_point = None
    continuous = True #本次实验中全部是连续值属性
    for i in range(num_attributes):
        attribute_list = [entry[i] for entry in data_set]  # 求属性列表，此时为连续值
        info_A_D = 0.0  # 特征A对数据集D的信息增益
        split_info_D = 0.0  # 数据集D关于特征A的值的熵
        # if attribute_list[0] not in set(['M', 'F', 'I']):
        #     continuous = True
        """
        属性为连续值，先对该属性下的所有离散值进行排序
        然后每相邻的两个值之间的中点作为划分点计算信息增益比，对应最大增益比的划分点为最佳划分点
        由于可能多个连续值可能相同，所以通过set只保留其中一个值
        """
        if continuous == True:
            attribute_list = sort(attribute_list)
            temp_set = set(attribute_list)  # 通过set来剔除相同的值
            attribute_list = [attr for attr in temp_set]
            split_points = []
            for index in range(len(attribute_list) - 1):
                # 求出各个划分点
                split_points.append((float(attribute_list[index]) + float(attribute_list[index + 1])) / 2)
            for split_point in split_points:  # 对划分点进行遍历
                info_A_D = 0.0
                split_info_D = 0.0
                for part in range(2):  # 最佳划分点将数据一分为二，因此循环2次即可得到两段数据
                    sub_data_set = split_data_set(data_set, i, split_point, True, part)
                    prob = len(sub_data_set) / float(len(data_set))
                    info_A_D += prob * calc_info_D(sub_data_set)
                    split_info_D -= prob * log(prob, 2)
                if split_info_D == 0:
                    split_info_D += 1
                """
                由于关于属性A的熵split_info_D可能为0，因此需要特殊处理
                常用的做法是把求所有属性熵的平均，为了方便，此处直接加1
                """
                grian_rate = (info_D - info_A_D) / split_info_D  # 计算信息增益比
                if grian_rate > max_grian_rate:
                    max_grian_rate = grian_rate
                    best_split_point = split_point
                    best_attribute_index = i
                    print([best_attribute_index, best_split_point])
        else:  # 划分属性为离散值
            attribute_list = [entry[i] for entry in data_set]  # 求属性列表
            attribute_set = set(attribute_list)
            for attribute in attribute_set:  # 对每个属性进行遍历
                sub_data_set = split_data_set(data_set, i, attribute, False)
                prob = len(sub_data_set) / float(len(data_set))
                info_A_D += prob * calc_info_D(sub_data_set)
                split_info_D -= prob * log(prob, 2)
            if split_info_D == 0:
                split_info_D += 1
            grian_rate = (info_D - info_A_D) / split_info_D  # 计算信息增益比
            if grian_rate > max_grian_rate:
                max_grian_rate = grian_rate
                best_attribute_index = i
                best_split_point = None  # 如果最佳属性是离散值，此处将分割点置为空留作判定
    return best_attribute_index, best_split_point

"""
多数表决：返回标签列表中数量最大的类
"""

def most_voted_attribute(label_list):
    label_nums = {}
    for label in label_list:
        if label in label_nums.keys():
            label_nums[label] += 1
        else:
            label_nums[label] = 1
    sorted_label_nums = sorted(label_nums.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_nums[0][0]

"""
计算数据集D的香农熵
"""

def calc_info_D(data_set):
    num_entries = len(data_set)
    label_nums = {}  # 为每个类别建立字典，value为对应该类别的数目
    for entry in data_set:
        if len(entry) >= 1:
            label = entry[-1]
            if label in label_nums.keys():
                label_nums[label] += 1
            else:
                label_nums[label] = 1
    info_D = 0.0
    for label in label_nums.keys():
        prob = float(label_nums[label]) / num_entries
        info_D -= prob * log(prob, 2)
    return info_D

"""
按属性划分子数据集，分为离散属性的划分与连续属性的划分
index为划分属性的下标，value在离散属性划分的情况下为划分属性的值，continuous决定了是离散还是连续属性划分
part在连续属性划分时使用，为0时表示得到划分点左边的数据集，1时表示得到划分点右边的数据集
"""

def split_data_set(data_set, index, value, continuous, part=0):
    res_data_set = []
    if continuous == True:  # 划分的属性为连续值
        for entry in data_set:
            if part == 0 and float(entry[index]) <= value:  # 求划分点左侧的数据集
                reduced_entry = entry[:index]
                reduced_entry=list(reduced_entry)
                reduced_entry.extend(entry[index + 1:])  # 划分后去除数据中第index列的值
                res_data_set.append(reduced_entry)
            if part == 1 and float(entry[index]) > value:  # 求划分点右侧的数据集
                reduced_entry = entry[:index]
                reduced_entry=list(reduced_entry)
                reduced_entry.extend(entry[index + 1:])
                res_data_set.append(reduced_entry)

    else:  # 划分的属性为离散值
        for entry in data_set:
            if entry[index] == value:  # 按数据集中第index列的值等于value的分数据集
                reduced_entry = entry[:index]
                reduced_entry.extend(entry[index + 1:])  # 划分后去除数据中第index列的值
                res_data_set.append(reduced_entry)
    return res_data_set

"""
对一项测试数据进行预测，通过递归来预测该项数据的标签
decision_tree:字典结构的决策树
attribute_labels:数据的属性名列表
one_test_data：预测的一项测试数据
"""

def decision_tree_predict(decision_tree, attribute_labels, one_test_data):
    first_key = list(decision_tree.keys())[0]
    second_dic = decision_tree[first_key]
    attribute_index = attribute_labels.index(first_key)
    res_label = None
    for key in second_dic.keys():  # 属性分连续值和离散值，连续值对应<=和>两种情况
        if key[0] == '<':
            value = float(key[2:])
            if float(one_test_data[attribute_index]) <= value:
                if type(second_dic[key]).__name__ == 'dict':
                    res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
                else:
                    res_label = second_dic[key]
        elif key[0] == '>':
            # print(key[1:])
            value = float(key[1:])
            if float(one_test_data[attribute_index]) > value:
                if type(second_dic[key]).__name__ == 'dict':
                    res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
                else:
                    res_label = second_dic[key]

        else:
            if one_test_data[attribute_index] == key:
                if type(second_dic[key]).__name__ == 'dict':
                    res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
                else:
                    res_label = second_dic[key]
    return res_label

if __name__ == '__main__':
    train_size = 70 # 训练集大小，数据集中总共有4177项数据
    train_data, test_data = load_data(train_size)
    #print(train_data)
    attribute_label = ['1', '2', '3','4']
    decision_tree = generate_decision_tree(train_data, attribute_label)
    #print(1111)
    # 递归会改变attribute_label的值，此处再传一次
    attribute_label = ['1', '2', '3','4']
    count = 0
    #计算准确率
    for one_test_data in test_data:
        #print(one_test_data)
        if len(one_test_data) >=1:
            if decision_tree_predict(decision_tree, attribute_label, one_test_data) == one_test_data[-1]:
                count += 1
    print(count)
    accuracy = count / len(test_data)
    print('训练集大小%d，测试集大小%d，准确率为:%.1f%%' % (train_size, len(test_data), 100 * accuracy))
    treePlotter.createPlot(decision_tree)
