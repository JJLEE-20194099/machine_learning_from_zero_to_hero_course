import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sqlalchemy import false
import seaborn as sns

def calc_entropy(label_quantity_list):
    """
        input: label_quantity_list - Danh sách số lượng mỗi nhãn
            Ví dụ: [4, 3, 5] - có 4 mẫu thuộc lớp c1, 3 mẫu lớp c2 và 5 mẫu lớp c3
        output: entropy của label_quantity_list
    """ 

    if (len(label_quantity_list) == 0):
        return 0

    arr = np.array(label_quantity_list)
    n_samples = arr.sum()
    prob_arr = arr / n_samples
    entropy = 0

    for label_prob in prob_arr:
        if label_prob != 0:
            entropy -= label_prob * np.log2(label_prob)
    
    return entropy

class Node():
    def __init__(self, row_ids, entropy, depth, children=[]):
        """
            row_ids: Danh sách vị trí của những mẫu dữ liệu trong bộ dataset
                    -> thuận tiện trong việc lấy dữ liệu
            entropy: lưu giá trị entropy của tập dữ liệu (tương ứng với danh sách vị trí row_ids)
            depth: độ sâu của mỗi node, node gốc - root node có độ sâu là 0
            children: danh sách các node con nếu có
            attribute: thuộc tính được lựa chọn theo giải thuật ID3
            values: danh sách những giá trị categorical phân biệt của attribute được lựa chọn trong tập dữ liệu ứng với từng node
            label: None nếu là internal node (node trong) và mang giá trị tên lớp nào đó nếu là leaf node (node lá)
        """
        self.row_ids = row_ids
        self.entropy = entropy
        self.depth = depth
        self.children = children
        self.attribute = None
        self.values = None
        self.label = None
    
    def set_properties(self, attribute, values):
        self.attribute = attribute
        self.values = values

    def set_label(self, label):
        self.label = label

class DecisionTreeClassifier():
    def __init__(self, max_depth, min_samples_split, min_information_gain):
        """
            root: Node gốc của cây
            max_depth: độ sâu lớn nhất trong quá trình phát triển cây
            min_infomation_gain: Nếu gía trị infomation_gain < min_infomation_gain
                                sẽ không phân nhánh và xem node đang xét là node lá
            min_samples_split: Nếu số lượng mẫu dữ liệu tại mỗi node sau khi phân nhánh 
                                của node hiện tại < min_samples_split, thì không thực hiện
                                phân nhánh và xem node đang xét là node lá 
        """

        self.root = None
        self.max_depth = max_depth
        self.min_information_gain = min_information_gain
        self.min_samples_split = min_samples_split
    
    def set_label(self, node):
        label_series = self.target[node.row_ids]

        # Tìm xem lớp chiếm nhiều nhất, lấy lớp đó làm nhãn cho node lá
        label_mode = label_series.mode()[0]
        node.set_label(label_mode)
    
    def entropy(self, row_ids):
        if len(row_ids) == 0:
            return 0
        no_samples_each_label = np.array(self.target[row_ids].value_counts())
        return calc_entropy(no_samples_each_label)
    
    def get_children_node_list(self, node):
        row_ids = node.row_ids
        
        max_information_gain = 0
        best_attribute = None
        best_splits = []
        best_values = None
        data_at_node = self.data.iloc[row_ids, :]

        for idx, attribute in enumerate(self.attributes):
            values = self.data.iloc[row_ids, idx].unique().tolist()

            """
                Nếu chỉ có 1 giá trị categorical tứng với thuộc tính attribute 
                thì không phân nhánh và node đó là node lá (không có children)
            """
            if len(values) == 1:
                continue

            splits = []

            for val in values:
                sub_row_ids = data_at_node.index[data_at_node[attribute] == val].tolist()
                splits.append(sub_row_ids)
            
            check = False
            
            """
                Biến check kiểm tra xem sau khi tách thành các node con thì
                tại các node con đó, số lượng mẫu dữ liệu còn lại nếu mà nhỏ hơn
                ngưỡng min_samples_split thì ta sẽ thực hiện quá trình tách
                và xem node đang xét là node lá
            """

            for split in splits:
                if len(split) < self.min_samples_split:
                    check = True
                    break
            
            if check == True:
                continue

            remaining_average_entropy = 0

            for split in splits:
                remaining_average_entropy += (len(split) / len(row_ids)) * self.entropy(split)
            
            information_gain = node.entropy - remaining_average_entropy

            if information_gain < self.min_information_gain:
                continue
            

            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_splits = splits
                best_attribute = attribute
                best_values = values
        
        node.set_properties(best_attribute, best_values)
        children_node_list = [Node(row_ids = split, entropy = self.entropy(split), depth = node.depth + 1) for split in best_splits]
        
        return children_node_list

    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data
        self.attributes = list(data)
        self.target = target
        self.labels = target.unique()
        row_ids = range(self.Ntrain)
        self.root = Node(row_ids = row_ids, entropy = self.entropy(row_ids), depth=0)
        
        # Thuật toán loang xây dựng decision tree

        queue = [self.root]

        while queue:
            node = queue.pop()
            if node.depth < self.max_depth:
                node.children = self.get_children_node_list(node)
                if not node.children:
                    self.set_label(node)
                queue = queue + node.children     
            else:
                self.set_label(node)
    
    def predict(self, test):
        n_samples = test.shape[0]
        labels = [None] * n_samples

        for idx in range(n_samples):
            sample = test.iloc[idx, :]
            node = self.root

            while node.children:
                node = node.children[node.values.index(sample[node.attribute])]
            
            labels[idx] = node.label

        return labels

if __name__ == "__main__":
    df_train = pd.read_csv('car_evaluation_train.csv', sep=',')
    df_test = pd.read_csv('car_evaluation_test.csv', sep=',')
    tree = DecisionTreeClassifier(max_depth = len(df_train.columns) - 1, min_samples_split = 2, min_information_gain=1e-4)
    tree.fit(df_train.iloc[:, :-1], df_train.iloc[:, -1])
    
    print("###TRAIN###")
    y_pred = tree.predict(df_train.iloc[:, :-1])
    print(f'Accuracy Score For Train Data: {accuracy_score(np.array(y_pred), np.array(df_train.iloc[:, -1].tolist())) * len(y_pred)}')
    print('Confusion matrix: ')
    matrix = confusion_matrix(np.array(y_pred), np.array(df_train.iloc[:, -1].tolist()))
    labels = df_train.iloc[:, -1].value_counts().index
    matrix_df = pd.DataFrame(matrix, index = labels, columns = labels)
    plt.title("Confusition Matrix For Train Data")
    sns.heatmap(matrix_df, annot=True)
    plt.savefig("Confusion Matrix For Car Evaluation Train.png")
    plt.show()

    print("###TEST###")
    y_pred = tree.predict(df_test.iloc[:, :-1])
    print(f'Accuracy Score For Test Data: {accuracy_score(np.array(y_pred), np.array(df_test.iloc[:, -1].tolist())) * len(y_pred)}')
    print('Confusion matrix: ')
    matrix = confusion_matrix(np.array(y_pred), np.array(df_test.iloc[:, -1].tolist()))
    labels = df_test.iloc[:, -1].value_counts().index
    matrix_df = pd.DataFrame(matrix, index = labels, columns = labels)
    plt.title("Confusition Matrix For Test Data")
    sns.heatmap(matrix_df, annot=True)
    plt.savefig("Confusion Matrix For Car Evaluation Test.png")
    plt.show()



            

    
    