# Mô hình rừng ngẫu nhiên - Random forests

Trong một tiết học môn ***nhập môn học máy và khai phá dữ liệu*** mình đã được nghe thấy lại câu chuyên ***thầy bói xem voi***. 

Lúc mà từng ***ông thầy bói*** đoán một cách ngẫu nhiên khi sờ tay vào ***từng khu vực*** của con voi, ông sờ vào chân voi thì đoán đó là ***cái cột đình***, ông sở vào tai voi thì đoán đó là ***cái quạt thóc***, ông sờ vào vòi của voi thì phán đó là ***con đỉa***, ông sờ vào ngà voi thì phán đó là ***cái đòn càn***. 

Ý nghĩa của câu chuyện này có ảnh hưởng rất lớn đến cuộc sống của chúng ta nói riêng và trong học máy nói chung.

Khi mà chúng ta chỉ ***nhìn vào 1 phần của dữ liệu*** thì việc phán đoán cho dữ liệu tương lai sẽ lệch đi rất nhiều. Nhưng khi kết hợp nhiều cách nhìn khác nhau vào dữ liệu thì ta lại có một phán đoán tốt.

Trong học máy cũng có một mô hình rất phổ biến và cũng áp dụng được triết lý trên đó là mô hình rừng ngẫu nhiên - ***Random Forests***

Đây là thuật toán học có giám sát ***(supervised learning)*** được dùng cho cả bài toán hồi quy ***(Regression)*** và vài toán phân loại ***(Classification)***

## 1. Tìm hiểu mô hình

***Mô hình Random Forests*** là tập hợp nhiều ***[cây quyết định - decision tree]()*** và thay vì chỉ dựa vào việc phán đoán của 1 cây duy nhất thì lại sử dụng ***toàn bộ phán đoán*** của toàn bộ cây

### 1.1 Phương pháp lấy mẫu có trùng lặp là gì?

Tập dữ liệu ban đầu có ***n mẫu dữ liệu***. 

Phương pháp lấy mẫu có trùng lặp hay ***Random Sampling with Replacement*** sẽ từng bước chọn ra 1 mẫu trong dữ liệu ban đầu (Khi lấy mẫu dữ liệu đó ra thì trong tập dữ liệu ban đầu sẽ thêm vào lại đúng mẫu dữ liệu đó). Điều kiện dừng là khi ta chọ đủ ***n mẫu dữ liệu***

Vậy phương pháp này mục đích để chọn ra 1 tập dữ liệu từ tập cha có kích thước đúng bằng tập cha mà cho phép các mẫu dữ liệu được trùng lặp.

Mô hình rừng ngẫu nhiên áp dụng phương pháp này để sinh ra tập huấn luyện từ tập dữ liệu ban đầu cho từng cây quyết định

### 1.2 Quá trình huấn luyện

Chúng ta cần có số lượng cây quyết định trong rừng và tạp huấn luyện ứng với mỗi cây

+ Ứng với mỗi cây, tạo ra 1 tập dữ liệu huấn luyện riêng. Các cây sẽ có tập huấn luyện khác nhau.
+ Tại mỗi cây của của rừng ngẫu nhiên, chúng ta chọn ngẫu nhiên một vài thuộc tính tại mỗi node trong quá trình phát triển
+ Mỗi cây sẽ được phát triển một cách hết cỡ (Khác với cây quyết định một số trường hợp chúng ta phải dừng sớm hoặc cắt tỉa cây để tránh trường hợp ***overfitting***)

## 2. Thực hành

### 2.1 Giới thiệu bài toán

Trong bài học này, mình sẽ giới thiệu tới các bạn một ví dụ cơ bản như sau:

Chúng ta có thông số của hơn 30000 mẫu dữ liệu để dự đoán rằng một người có thể thu nhập ***(income)***  hơn ***$50k*** trong 1 năm không. Thông tin của từng mẫu bao gồm: ***age***, ***workclass***, ***education***, ... Và cột cuối cùng là cột để chúng ta dự đoán 

![](https://i.imgur.com/Z1cvE3L.png)

***Bộ dữ liệu đầy đủ ở [đây](https://www.kaggle.com/uciml/adult-census-income)***

### 2.2 Giải quyết bài toán

#### 2.2.1 Khai báo thư viện cần sử dụng

+ scikit-learn(sklearn): thư viện hỗ trợ phong phú các mô hình học máy cùng với các hàm đánh giá và huấn luyện đa dạng
+ pandas: thư viện xử lý dữ liệu dạng bảng
+ numpy: thư viện cho đại số tuyến tính dùng để biến đổi ma trận, làm việc với vector
+ matplotlib: thư viện phục vụ vẽ các đồ thị
+ seaborn: thư viện được xây dựng trên matplotlib, dùng để vẽ hình đẹp hơn

```python=
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.model_selection import learning_curve
```

#### 2.2.2 Đọc dữ liệu

```python=
adult_df = pd.read_csv('adult.csv', sep=',')
print("Kích thước của bộ dữ liệu:", adult_df.shape)
```

#### 2.2.3 Tiền xử lý dữ liệu

+ Chúng ta cần kiểm tra kiểu dữ liệu của tất cả các cột
    ```python=
    print("Kiểu dữ liệu của các trường thuộc tính:")
    print(adult_df.dtypes)
    ```
    ```php=
    Kiểu dữ liệu của các trường thuộc tính:
    age                int64
    workclass         object
    fnlwgt             int64
    education         object
    education.num      int64
    marital.status    object
    occupation        object
    relationship      object
    race              object
    sex               object
    capital.gain       int64
    capital.loss       int64
    hours.per.week     int64
    native.country    object
    income            object
    dtype: object
    ```
+ Nhìn thoáng qua dữ liệu ta thấy một số trường thuộc tính của dữ liệu gặp phải vấn đề ***missing value***. Mình thay dấu chấm ***?*** bằng số ***0***. Và mình đếm số lượng số ***0*** ở những trường thuộc tính ***categorical***(Vì cột có kiểu ***int64*** không thể chứa ***?*** được)
    ```ru=
    adult_df = adult_df.replace("?", 0)

    for col in adult_df.columns:
        if adult_df[col].dtype == 'object':
            vals = adult_df[col].value_counts().index.tolist()
            if (0 in vals):
                print(f'Cột "{col}" có {len(vals)} giá trị khác nhau và bao gồm {count_zero(adult_df[col])} giá trị 0')
            else:
                print(f'Cột "{col}" có {len(vals)} giá trị khác nhau và "không" bao gồm giá trị 0')
    ```
    
    ```python=
    Cột "workclass" có 9 giá trị khác nhau và bao gồm 1836 giá trị 0
    Cột "education" có 16 giá trị khác nhau và "không" bao gồm giá trị 0
    Cột "marital.status" có 7 giá trị khác nhau và "không" bao gồm giá trị 0
    Cột "occupation" có 15 giá trị khác nhau và bao gồm 1843 giá trị 0
    Cột "relationship" có 6 giá trị khác nhau và "không" bao gồm giá trị 0
    Cột "race" có 5 giá trị khác nhau và "không" bao gồm giá trị 0
    Cột "sex" có 2 giá trị khác nhau và "không" bao gồm giá trị 0
    Cột "native.country" có 42 giá trị khác nhau và bao gồm 583 giá trị 0
    Cột "income" có 2 giá trị khác nhau và "không" bao gồm giá trị 0
    ```
+ Do dữ liệu thiếu khá ít nên mình quyết định loại bỏ các mẫu dữ liệu có những trường thuộc tính bị thiếu này
    ```python=
    no_samples = adult_df.shape[0]

    adult_df = adult_df[adult_df["workclass"] != 0]
    adult_df = adult_df[adult_df["native.country"] != 0]
    adult_df = adult_df[adult_df["occupation"] != 0]

    no_samples_after_remove_zero = adult_df.shape[0]
    print(f'Đã loại bỏ {no_samples - no_samples_after_remove_zero} hàng')
    ```
    
+ Đối với trường số, mình chuẩn hóa lại:
    ```jsx=
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    adult_df.iloc[:,[0,2,4,10,11,12]] = sc.fit_transform(adult_df.iloc[:,[0,2,4,10,11,12]])
    adult_df.iloc[:,[0,2,4,10,11,12]] = sc.transform(adult_df.iloc[:,[0,2,4,10,11,12]])
    ```
+ Đối với trường dữ liệu rời rạc có nhiều giá trị gần giống nhau ***(educationl, marital.status)***, mình nhóm lại: 
    ```jsx=
    adult_df['education'] = adult_df['education'].str.replace('HS-grad|9th|Preschool|12th|1st-4th|11th|10th|7th-8th|Some-college|5th-6th','Low-education',regex = True)
    adult_df['education'] = adult_df['education'].str.replace('Assoc-voc|Masters|Prof-school|Doctorate','High-education',regex = True)

    adult_df['marital.status'] = adult_df['marital.status'].str.replace('Widowed|Divorced|Separated|Never-married|Married-spouse-absent','no',regex = True)
    adult_df['marital.status'] = adult_df['marital.status'].str.replace('Married-civ-spouse|Married-AF-spouse','yes',regex = True)
    ```
+ Mã hóa các trường ***categorical*** vể dạng số
    Mình sử dụng lớp ***OrdinalEncoder*** của gói thư viện ***category_encoders***
```jsx=
    import category_encoders as ce

    categorical_cols = []
    for col in adult_df.columns[:-1]:
        if adult_df[col].dtype == 'object':
            categorical_cols.append(col)

    encoder = ce.OrdinalEncoder(cols = categorical_cols)
    modified_adult_df = encoder.fit_transform(adult_df.iloc[:, :-1])
    X_adult_arr = modified_adult_df.to_numpy()
    y_adult_arr = adult_df.iloc[:, -1].to_numpy()
```

#### 2.2.4 Phân tách tập train và tập test

Số lượng mẫu dữ liệ của từng lớp chênh lệch nhau:

```jsx=
adult_df.iloc[:, -1].value_counts().plot(kind='bar')
```
![](https://i.imgur.com/wAWKf7z.png)

Nên khi chia dữ liệu rất dễ gặp phải trường hợp tập ***train*** chứa cực nhiều mẫu thuộc lớp ***<=50K***.

Để giải quyết vấn đề nay, mình sử dụng kĩ thuật lấy mẫu phân tầng ***(mình sẽ có một bài viết về các kĩ thuật lấy mẫu sau.)***


```jsx=
from sklearn.model_selection import StratifiedShuffleSplit
def split_train_test(X_adult_arr, y_adult_arr):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = None, None

    for train_index, test_index in sss.split(X_adult_arr, y_adult_arr):
        train_idx = train_index
        test_idx = test_index
    
    adult_train_arr = {
    'data': X_adult_arr[train_idx, :],
    'target': y_adult_arr[train_idx]
    }

    adult_test_arr = {
        'data': X_adult_arr[test_idx, :],
        'target': y_adult_arr[test_idx]
    }

    return adult_train_arr, adult_test_arr
```

```jsx=
adult_train_arr, adult_test_arr = split_train_test(X_adult_arr, y_adult_arr)
```

#### 2.2.5 Huấn luyện mô hình

Mình sử dụng lớp ***RandomForestClassifier*** của gói ***ensemble*** của ***sklearn***

Do trong lớp RandomForestClassifier có ***1 số tham số*** bạn cần phải điều chỉnh và đánh giá 1 cách củ thể. Hơn thế nữa số bài toán phân loại của chúng ta có ***số lượng 2 lớp khá chênh lệch nhau*** nên mình dùng chến lược ***cross_validation*** (mình sẽ giới thiệu ở các bài sau).

+ Tuning ***n_estimator*** (số lượng cây quyết định trong rừng)

```jsx=
def cross_validation(estimator):
    _, train_scores, test_scores = learning_curve(estimator, adult_train_arr["data"], adult_train_arr["target"], cv=10, n_jobs=-1, train_sizes=[1.0, ], scoring="accuracy")
    test_scores = test_scores[0]
    mean, std = test_scores.mean(), test_scores.std()
    return mean, std
```

```jsx=
from tqdm import tqdm
scores = []
optimal_estimators = None
max_score = 0
for quantity in tqdm(range(100, 500, 50)):
    random_forest_clf = RandomForestClassifier(n_estimators=quantity)
    mean, std = cross_validation(random_forest_clf)
    scores.append(mean)
    if mean > max_score:
        max_score = mean
        optimal_estimators = quantity

plt.figure(figsize=(10, 7))
plt.plot(range(100, 500, 50), scores, color="green", linestyle="dashed", marker="o", markerfacecolor="orange", markersize=10)
plt.ylabel("Score")
plt.xlabel("Number of estimators in Random Forests")
```

![](https://i.imgur.com/9D7QHRP.png)


+ Tuning criterion (độ đo để xây dựng cây quyết định trong rừng ngẫu nhiên: ***gini***, ***entropy***):

```jsx=
title = f'Tuning criterion with n_estimators = {optimal_estimators}'
xlabel = "criterion"
X = []
Y = []
error = []

for criterion in tqdm(["gini", "entropy"]):
    random_forest_clf = RandomForestClassifier(criterion=criterion, n_estimators=optimal_estimators)
    mean, std = cross_validation(random_forest_clf)
    X.append(str(criterion))
    Y.append(mean)
    error.append(std)

plot(title, xlabel, X, Y, error)
plt.savefig('RandomForest_tunning_criterion.png', bbox_inches='tight')
plt.show()
```
![](https://i.imgur.com/k9LpEEV.png)

Nhìn vào hình vẽ trên, mình chọn ***gini*** cho mô hình để dự đoán dữ liệu

***Đầy đủ source code tại [đây]()***

#### 2.2.6 Dự đoán trên tập test

Dựa vào các tham số ***n_estimators*** và ***criterion*** chúng ta sử dụng xây dựng rừng ngẫu nhiên và đưa ra dự đoán cho tập test. Do số lượng các lớp khá chênh lệch nhau trong tập test nên mình dùng thêm ***confusion_matrix*** đánh giá khách quan hơn

```jsx=
random_forest_clf = RandomForestClassifier(criterion='entropy', n_estimators=optimal_estimators)
random_forest_clf.fit(adult_train_arr['data'], adult_train_arr['target'])
y_pred = random_forest_clf.predict(adult_test_arr['data'])
print("Accuracy_score:", accuracy_score(y_pred, adult_test_arr['target']))

c_matrix = confusion_matrix(adult_test_arr['target'], y_pred)
labels = adult_df.iloc[:, -1].value_counts().index
c_matrix_df = pd.DataFrame(c_matrix, index = labels, columns = labels)
plt.title("Confustion Matrix For Test Data")
sns.heatmap(c_matrix_df, annot=True)

```

```jsx=
Accuracy_score: 0.8478368970661363
```

![](https://i.imgur.com/k2Uo7fa.png)



## 3. Ưu điểm của mô hình rừng ngẫu nhiên

+ Mô hình dễ cái đặt vì tính ngẫu nhiên trong cách xây dựng mỗi cây quyết định

+ Mô hình rừng ngẫu nhiên có thể làm việc với dữ liệu có nhiều chiều, và việc phát triển tới độ sâu tối đã của mỗi cây quyết định không làm cho mô hình rừng ngẫu nhiên gặp trường hợp ***overfitting***.

Sự ngẫu nhiên chọn thuộc tính trong việc phát triển cây, sự ngẫu nhiên trong việc chọn dữ liệu huấn luyện vì sao lại khiến cho mô hình rừng ngẫu nhiên lại tốt như vây?

Ta thấy nếu chỉ dùng ***duy nhất 1 cây quyết định*** để mang đi dự đoán khả năng cao những sự ngẫu nhiên trên sẽ khiến cho mô hình bị ***underfitting*** hoặc ***overfitting***. Sở dĩ do mô hình sẽ chỉ học được một phần thông tin từ dữ liệu. Những khi két hợp tất cả lại với nhau, mô hình rừng ngẫu nhiên sẽ học được rất nhiều thông tin từ dữ liệu. Vì vậy đây là 1 phần lý do mô hình rừng ngẫu nhiên hoạt động tốt.

Một vài ví dụ đời sống cho lý lẽ này:

+ Bầu cử tổng thống
+ Tấm lý số đông
+ 1 đội chỉ chơi tốt nếu kết hợp các cầu thủ giỏi mỗi mảng lại với nhau

## 4. Tài liệu tham khảo

+ [Random Forests Algorithm - Tiệp Vũ, Tuấn Nguyễn](https://machinelearningcoban.com/tabml_book/ch_model/random_forest.html)
+ [Random Forests - Trần Quang Khoát](https://users.soict.hust.edu.vn/khoattq/ml-dm-course/L7-Random-forests.pdf)
+ [The wisdom of Crowds](https://en.wikipedia.org/wiki/The_Wisdom_of_Crowds)


