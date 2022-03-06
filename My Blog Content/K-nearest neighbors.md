# K-nearest neighbors

Chắc hẳn tất cả các bạn đang đọc bài viết này, đã từng nghe câu tục ngữ do ông cha ta đúc kết: ***"Gần mực thì đen, gần đèn thì sáng"***. Đối với ***Machine Learning***, cũng đã áp dụng những điều đơn giản trong cuộc sống vào một số bài toán củ thể. Đặc biệt, thuộc toán K láng giềng gần nhất ***(K-nearest neighbors)*** là một ví dụ điển hình.

## 1. Giới thiệu 

Thuật toán K-nearest neighbors (KNN) được biết là một trong những thuật toán đơn giản nhất của học máy và nó còn được gọi với 1 số cái tên khác như là: ***Lười học (Lazy learning)***, ***Học dựa trên bộ nhớ (Memory-based learning)***, ***Học dựa trên các ví dụ (Instance-based learning)***

Sở dĩ được gọi là 1 trong những phương pháp đơn giản nhất trong ML , vì ý tưởng của của phương pháp cũng cữ kì đơn giản:

+ Khác với những phương pháp khác, phương pháp KNN không có bất cứ 1 giả thuyết ***(assumption)*** nào cho hàm chúng ta cần phải học
***Ví dụ:*** Đối với phương pháp Linear Regression thì chúng ta xét dạng là hàm tuyến tính là hàm xấp xỉ 1 hàm số dúng để dự đoán chúng ta chưa biết. Và giả thuyết ở đây là ***tuyến tính***

+ 1 điểm đặc biệt nữa ở phương pháp KNN là trong giai đoạn học có thể nó là chúng ta không học hay làm bất cứ 1 điều gì đặ biệt ngoài ***lưu trữ dữ liệu huấn luyện (train data)***. Và đây cũng chính là lý do nhiều người nói rằng phương pháp KNN là ***phương pháp Lazy learning (lười học)***

+ Việc phán dựa vào ý nghĩa của câu tục ngữ ***Gần mực thì đen, gần đèn thì sáng***. Củ thể việc dự đoán cho 1 quan sát mới dựa vào tính chất của các láng giếng gần nhất của nó trong tập training.

Phương pháp ***phi tham số (non-parametric)*** là phương pháp chúng ta không đặt ra bất cứ 1 giả thuyết nào khi chúng ta học cả, và ***KNN*** là 1 trong những phương pháp phi tham số

***Chú ý:*** Phương pháp phi tham số không phải là 1 phương pháp không có tham số mà là phương pháp chúng ta không dùng bất cứ 1 dạng hàm nào củ thể về hàm chúng ta xấp xỉ (học). Hồi quy tuyến tính, hồi quy ***Ridge*** và hồi quy ***Lasso*** là các phương pháp có tham số

## 2. Yếu tố ảnh hưởng đến thuật toán KNN

+ Một trong những yếu tố quan trọng ảnh hưởng lớn đến thuật toán KNN là ***khoảng cách giữa 2 quan sát*** được tính như thế nào. Khoảng cách này sẽ đo được sự giống nhau và khác nhau giữa 2 mẫu quan sát

+ Số lượng hàng xóm cũng ảnh hưởng đến thuật toán. ***Bao nhiêu hàng xóm là đủ?*** 10, 20, 30, 50, thậm chí nhiều bài toán 1 hàng xóm là đủ và có thể đạt được 1 mức lỗi tối ưu.

+ Hàng xóm nào là hàng xóm quan trọng, có nên ***đánh trọng số*** cho từng hàng xóm, hay tất cả hàng xóm đều có ***vai trò*** như nhau.

## 3. Giải quyết bài toán

### 3.1. Bạn có thể hiểu ý tưởng thuật toán thông qua ví dụ này

![](https://i.imgur.com/8W0ulof.png)

Như chúng ta đã thấy, tất cả các mẫu dữ liệu dùng để phán đoán thuộc vào 2 lớp: ***Lớp A và lớp B***, và có 1 mẫu quan sát mới ***(ô vuông ? màu vàng)*** chúng ta cần phân lớp cho nó.

+ Nếu dùng đúng ***k=1 hàng xóm*** để dự đoán thì mẫu quan sát mới thuộc ***lớp A*** vì gần điểm thuộc ***lớp A*** nhất.

+ Nếu dùng ***k=3 hàng xóm*** để dự đoán thì mẫu quan sát mới thuộc ***lớp B***.

+ Nếu dùng ***k=7 hàng xóm*** để dự đoán thì mẫu quan sát mới thuộc ***lớp A***. (Trong ***7 hàng xóm*** gần nó nhất có 4 mẫu thuộc ***lớp A*** nhiều hơn 3 mẫu thuộc ***lớp B***)

### 3.2 KNN cho bài toán hồi quy và bài toán phân lớp

+ Quá trình học rất đơn giản: Chúng ta chỉ cần lưu bộ dữ liệu để huấn luyện
+ Quá trình dự đoán một mẫu quan sát mới ***z***:
    + Tính khoảng cách từ ***z*** tới tất cả ${x \in D}$
    + Với k là số hàng xóm chúng ta chọn ban đầu, chọn ra tập ***NB(z)*** là k hàng xóm gần nhất của ***z***

#### 3.2.1 Áp dụng thuật toán KNN cho bài toán phân loại
Hàng xóm thuộc loại lớp nào trong ***NB(z)*** thì lấy lớp đó là kết quả dự đoán cho mẫu quan sát mới ***z***

#### 3.2.1 Áp dụng thuật toán KNN cho bài toán hồi quy

Kết quả dự đoán cho mẫu quan sát mới ***z*** là:

\begin{equation}
    y_z = \frac{1}{k} \sum\limits_{x \in NB(z)} y_x
\end{equation}

### 3.3 Số hàng xóm thể nào là ổn?

+ Bạn có thể chọn hàng xóm theo số lượng k cho trước, có thể là: ***k = 1 hoặc k = 2 hoặc k = 10 hoặc k = 100, ...***
+ Cũng có thể chọn số hàng xóm theo vùng: ***Nghĩa là chọn trước bán kính với tâm là mẫu quan sát cần dự đoán, số hàng xóm sẽ là sô điểm nằm trong vùng bán kính đó***

***Chú ý:***
+ Nếu chọn ***k = 1***, mặc dù 1 hàng xóm có thể đạt được mức lỗi tối ưu tuy nhiên không thể tránh được ảnh hưởng của lỗi, nhiều ***(noise)*** trong tập dữ liệu
***-> Nên chọn số hàng xóm lớn hớn 1***

+ Nếu chọn ***k*** quá lớn thì kết quả dự đoán có thể ***rất tệ.*** Sở dĩ nếu chúng ta dụ đoán nhãn phân loại cho 1 con động vật trong 1 khu rừng ***(Tập huấn luyện)***, khi chọn số hàng xóm quá lớn thì nhãn phán đoán của chúng ta xác suất cao luôn luôn là phân loại vào lớp động vật nào có số lượng lớn nhất. Hơn thế nữa, chọn ***k*** quá lớn, chúng ta vô tình bỏ qua những thuộc tính ***(đặc trưng)*** tiềm ẩn xung quanh điểm dữ đoán ***z***. Khi đó ta có thể nói, mô hình của chúng ta đang gặp vấn đề ***underfitting***


Chọn ít cũng không được, chọn nhiều cũng không xong, chẳng lẽ chúng ta chọn bừa.

Để giải quyết vấn đề này có 1 tính chất rất hay ho mình học trong slide [K-nearest neighbors - Trần Quang Khoát](https://users.soict.hust.edu.vn/khoattq/ml-dm-course/L6-KNN.pdf) như sau:

Thuật toán KNN sẽ đạt được mức lỗi tối ưu nếu thỏa mãn các điều kiện sau:

+ Hàm số ẩn chúng ta đang cố gắng xấp xỉ hay giá trị ***Y*** bị chặn và liên tục
+ Tập training size ***M*** phải có ***kích thước lớn***
+ $k \to \infty$, $\frac{k}{M} \to 0$, $\frac{k}{logM} \to +\infty$ ***(${*}$)*** nghĩa là:
+ Chọn ***k lớn*** nhưng so với kích thước tập huấn luyện phải ***rất bé*** và ***k luôn luôn lớn hơn $log (M)$***

Vì vậy điểm bắt đầu khi chúng ta lựa chọn hằng số ***k*** trong các project thực tế, nên chọn bắt đầu từ ***log(M) (M: số lượng mẫu dữ liệu tập training)***

## 3.4 Độ đo khoảng cách hoặc độ đo tương đồng nào là phù hợp?

+ 1 độ đo được chọn có nghĩa là chúng ta đã giả sử phân bố dữ liệu của chúng ta phù hợp với độ đo khoảng cách đó. Ví dụ: nếu chúng ta xét bài toán phân loại xúc con người và dùng ***khoảng cách Euclid*** đồng nghĩa chúng ta đã giả sử rằng tập dữ liệu chúng ta đang xét nằm trong không gian ***Euclid***. Và điều đó ảnh hưởng rất nhiều tới việc dự đoán sau này vì không thể tính khoảng cách ***1 người trên tàu siêu tốc*** và ***1 người đang đi bộ bên cạnh cô bạn gái*** bằng khoảng cách Euclid được

+ Chúng ta nên chọn các khoảng cách hình học phù hợp đầu vào là các số thực
+ Với các dữ liệu đầu vào dạng nhị phân thì nên sử dụng khoảng cách ***Hamming***

Khoảng cách ***Hamming*** cho 2 xâu hoặc 2 vector có ***size*** bằng nhau có giá trị bằng ***số các phần tử tại cùng 1 vị trí khác nhau***

***Ví dụ:*** 
+ Khoảng cách Hamming giữa [1, 0, ***1***, 0, ***1***] và [1, 0, ***0***, 0, ***0***] là 2.
+ Khoảng cách Hamming giữa 2***14***3***8***96 và 2***23***3***7***96 là 3.

***Chú ý:*** Chúng ta có thể sử dụng độ Cosine cho các kiểu dữ liệu dạng ***văn bản*** hoặc dữ liệu ***rời rạc***

\begin{equation}
d(x, z) = \frac{x^Tz}{\lVert{x}\lVert.\lVert{z}\lVert}
\end{equation}

## 3.5 Đánh trọng số cho hàng xóm (Weighting neighbors)

+ Xét mẫu quan sát ***z*** cần được phán đoán
+ Chúng ta cần đánh trọng số cho hàng xóm $x_i \in NB(z)$ để nói lên sự quan trọng của các hàng xóm gần so với các hàng xóm xa thay vì chúng ta xem vai trò của ***k*** hàng xóm là như nhau
+ d(x, z) là khoảng cách giữa x và z
+ Chúng ta có 1 số cách đánh trọng số như sau:
    + ${v(x, z) = \frac{1}{\alpha + d(x, z)}}$
    + ${v(x, z) = \frac{1}{\alpha + [d(x, z)]^2}}$
    + ${v(x, z) = e^{-\frac{d(x, z)^2}{\sigma^2}}}$


***Kết quả dự đoán bây giờ sẽ trở thành:***

***Bài toán phân loại***

Thay vì chọn số lượng lớp nào lớn nhất trong ***k hàng xóm*** để mang ra phán đoán, ta phán đoán dựa vào công thức dưới đây:

\begin{equation}
c_z = arg \max\limits_{c_j \in C} \sum\limits_{x \in NB(z)} v(x, z) * Identical(c_j, c_x)
\end{equation}

Với:  ${Identical(a, b)\text{ bằng } 1 \text{ nếu }a = b\text{ và bằng }0\text{ nếu } a \neq b}$

***Bài toán hồi quy***

Thay vì lấy trung bình giá trị của ***k hàng xóm*** để mang ra phán đoán, ta phán đoán dựa vào công thức dưới đây:

\begin{equation}
y_z = \frac{\sum\limits_{x \in NB(z)}v(x, z)*y_x}{\sum\limits_{x \in NB(z)}v(x, z)}
\end{equation}
    
## 4. Thực hành

## 4.1 Áp dụng KNN cho bài toán phân lớp (KNN for Classification)

### 4.1.1 Giới thiệu bài toán

Chúng ta có 1000 mẫu dữ liệu được chia làm 2 lớp. Mục đích đơn giản là khi có 1 mẫu quan sát mới thì chúng ta phải dự đoán dữ liệu đó thuộc lớp nào trong 2 lớp trên

![](https://i.imgur.com/bHNmUwI.png)

***Bộ dữ liệu đầy đủ ở [đây]()***

### 4.1.2 Giải quyết bài toán
#### 4.1.2.1 Khai báo thư viện

+ scikit-learn(sklearn): thư viện hỗ trợ phong phú các mô hình học máy cùng với các hàm đánh giá và huẩn luyện đa dạng
+ pandas: thư viện xử lý dữ liệu dạng bảng
+ numpy: thư viện cho đại số tuyến tính dùng để biến đổi ma trận, làm việc vơi vector
+ matplotlib: thư viện phục vụ vẽ các đồ thị

```python=
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
```
#### 4.1.2.2 Đọc dữ liệu

```python=
df = pd.read_csv('./random_classified_data.csv', sep=',', index_col=0)
print("Số chiều của bộ dữ liệu: ", df.shape)
```

#### 4.1.2.3 Chuẩn hóa dữ liệu

Khi bộ dữ liệu có nhiều cột, hay nói cách khác có nhiều thuộc tính khác nhau và bên cạnh đó ứng với mỗi thuộc tính lại có độ lớn, min, max, đơn vị khác nhau, vì vậy điều này ảnh hưởng tới độ chính xác, quá trình hội tụ và thời gian tính toán của thuật toán. Chính vì vậy giải phát đơn giản nhất là đưa các đặc trưng này về chung một tỷ lệ nhất định. Đó là ***Data Normalization***.

***Trong bài viết này mình sử dụng [StandardScaler của gói preprocessing](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) trong sklearn để chuẩn hóa
bộ dữ liệu***

```python=
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

def scale_columns(df, cols):
    for col in cols:
        df[col] = pd.DataFrame(ss.fit_transform(pd.DataFrame(df[col])), columns=[col])
    return df

df_scaled = scale_columns(df, df.columns[:-1])
arr = df_scaled.to_numpy()
```

#### 4.1.2.4 Tách bộ dữ liệu

Đối với [bộ dữ liệu]() này gồm 1000 mẫu dữ liệu mình tách chúng ra ***700 mẫu để train*** và ***300 mẫu để test***


```python=
arr_train = {
    'data': arr[:700, :-1],
    'target': arr[:700, -1]
}

arr_test = {
    'data': arr[700:, :-1],
    'target': arr[700:, -1]
}
```

#### 4.1.2.5 Huần luyện và dự đoán

Sau đây, mình xét trường hợp đơn giản trước ***k = 1***

```python=
knn_clf = KNeighborsClassifier(n_neighbors=1)
knn_clf.fit(arr_train['data'], arr_train['target'])
y_pred = knn_clf.predict(arr_test['data'])

print("Print results for 25 test data samples in test:")
print("Predicted labels: ", y_pred[:25])
print("Ground truth:     ", arr_test["target"][:25])    
```

```ruby=
Print results for 25 test data samples in test:
Predicted labels:  [0. 0. 0. 1. 0. 1. 1. 1. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1. 1. 1. 0. 1. 0.
 1.]
Ground truth:      [0. 0. 0. 1. 0. 1. 1. 1. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1. 1. 1. 0. 1. 0.
 0.]
```

Tính độ chính xác khi dùng 1 hàng xóm và in ra ***confustion matrix***

```python=
print(f'Using 1 neighbor, accuracy score: {round(100 * accuracy_score(arr_test["target"], y_pred), ndigits=2)}%')
print("Confusion Matrix:\n", confusion_matrix(arr_test["target"], y_pred))
```

```ruby=
Using 1 neighbor, accuracy score: 93.67%
Confusion Matrix:
 [[130  12]
 [  7 151]]
```

Ta thấy độ chính xác 93.67% khá cao, và để tăng độ tin cậy mình đã in thêm ***confusion matrix(mình sẽ giới thiệu trong các bài viết tiếp theo)***. Tức là có ***12 điểm*** đáng lẽ thuộc lớp thứ 1 nhưng ta đã dự đoán vào lớp thứ 2 và có ***7 điểm*** đáng lẽ thuộc lớp thứ 2 nhưng đã dự đoán vào lớp thứ 1. ***Tỷ lệ đoán sai của các lớp thấp hơn nhiều so với tỷ lệ đoán đúng***

Tiếp theo, mình sẽ lựa chọn giá trị k nào sẽ hợp lý trong 1 khoangr giá trị cho trước

```python=
error_rate_list = []
for quantity in range(1, 51):
    knn_clf = KNeighborsClassifier(n_neighbors = quantity)
    knn_clf.fit(arr_train["data"], arr_train["target"])
    y_pred = knn_clf.predict(arr_test["data"])
    error_rate_list.append(np.mean(y_pred != arr_test["target"]))
```
Đồ thị biểu hiện sự thay đổi tỷ lệ đoán sai dựa trên sự thay đổi của ***số lượng k hàng xóm***

```python=
plt.figure(figsize=(10, 7))
plt.plot(range(1, 51), error_rate_list, color="blue", linestyle="dashed", marker="o", markerfacecolor="red", markersize=10)
plt.ylabel("Error Rate")
plt.xlabel("Number of Neighbors used in KNeighborsClassifier")
```

![](https://i.imgur.com/f5qUgwH.png)

Nhìn vào đồ thị ta chọn ***k = 20***

```python=
knn_clf = KNeighborsClassifier(n_neighbors=20)
knn_clf.fit(arr_train['data'], arr_train['target'])
y_pred = knn_clf.predict(arr_test['data'])

print("Print results for 25 test data samples in test:")
print("Predicted labels: ", y_pred[:25])
print("Ground truth:     ", arr_test["target"][:25])
```

```python=
Print results for 25 test data samples in test:
Predicted labels:  [0. 0. 0. 1. 0. 1. 1. 1. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1. 1. 1. 0. 1. 0.
 0.]
Ground truth:      [0. 0. 0. 1. 0. 1. 1. 1. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1. 1. 1. 0. 1. 0.
 0.]
```

```python=
print(f'Using 20 neighbors, accuracy score: {round(100 * accuracy_score(arr_test["target"], y_pred), ndigits=2)}%')
print("Confusion Matrix:\n", confusion_matrix(arr_test["target"], y_pred))
```

```python=
Using 20 neighbors, accuracy score: 96.67%
Confusion Matrix:
 [[135   7]
 [  3 155]]
```

Độ chính xác đã cao hơn 1 chút là 96.67% và tỷ lệ đoán sai ở các lớp cũng đã giảm xuống

Bạn có thể đánh trọng số cho các hàng xóm bằng cách thêm ***weights="distance"*** vào trong ***KNeighborsClassifier***, tham khảo code ở [đây]()

## 4.2 Áp dụng KNN cho bài toán hồi quy (KNN for Regression)

### 4.1.1 Giới thiệu bài toán

Chúng ta có dữ liệu về 100 bạn sinh viên năm nhất. Mục đích đơn giản là dự đoán điểm ***GPA** dựa vào 1 số thông tin cho trước như: ***High School GPA***, ***Years Off***, ***College***, ..vvv

![](https://i.imgur.com/hjSE4AJ.png)

***Bộ dữ liệu đầy đủ ở [đây]()***


### 4.1.2 Xử lý các trường categorical

Chúng ta sử dụng ***hàm get_dummies của pandas***

### 4.1.3 Huấn luyện và dự đoán

Tương tự như trên, xét k = 1 hàng xóm

```python=
knn_regr = KNeighborsRegressor(n_neighbors=1, p=2)
knn_regr.fit(freshmen_arr_train["data"], freshmen_arr_train["target"])
y_pred = knn_regr.predict(freshmen_arr_test["data"])

print("Print results for 5 test data samples in test:")
print("Predicted GPA: ", y_pred[:5])
print("Real GPA:         ", freshmen_arr_test["target"][:5])
```
```python=
Print results for 5 test data samples in test:
Predicted GPA:  [3.27 3.12 1.81 1.72 0.35]
Real GPA:          [2.57 2.8  1.33 1.73 2.25]
```

Tính trung bình lỗi bình phương khi dùng 1 hàng xóm
```python=
print(f'Using 1 neighbor, mean squared error: {mean_squared_error(freshmen_arr_test["target"], y_pred)}')
```
```ruby=
Using 1 neighbor, mean squared error: 1.3301733333333334
```

Chọn ***k hàng xóm phù hợp***

```python=
error_list = []
for quantity in range(1, 51):
    knn_regr = KNeighborsRegressor(n_neighbors=quantity, p=2)
    knn_regr.fit(freshmen_arr_train["data"], freshmen_arr_train["target"])
    y_pred = knn_regr.predict(freshmen_arr_test["data"])
    error_list.append(mean_squared_error(freshmen_arr_test["target"], y_pred))

```


Đồ thị biểu hiện sự thay đổi trung bình bình phương lỗi dựa trên sự thay đổi của ***số lượng k hàng xóm***


```python=
plt.figure(figsize=(10, 7))
plt.plot(range(1, 51), error_list, color="blue", linestyle="dashed", marker="o", markerfacecolor="red", markersize=10)
plt.ylabel("Mean Squared Error")
plt.xlabel("Number of Neighbors used in KNeighborsRegressor")
```

![](https://i.imgur.com/WSiMNYr.png)

Chọn ***k = 6***

```python=
knn_regr = KNeighborsRegressor(n_neighbors=6, p=2)
knn_regr.fit(freshmen_arr_train["data"], freshmen_arr_train["target"])
y_pred = knn_regr.predict(freshmen_arr_test["data"])

print("Print results for 5 test data samples in test:")
print("Predicted GPA: ", y_pred[:5])
print("Real GPA:     ", freshmen_arr_test["target"][:5])
```

```python=
Print results for 5 test data samples in test:
Predicted GPA:  [2.52       2.00666667 2.23166667 2.17166667 1.32      ]
Real GPA:      [2.57 2.8  1.33 1.73 2.25]
```


Tính trung bình lỗi bình phương khi dùng 1 hàng xóm
```python=
print(f'Using 6 neighbors, mean squared error: {mean_squared_error(freshmen_arr_test["target"], y_pred)}')
```
```ruby=
Using 6 neighbors, mean squared error: 0.7094427777777774
```

Chúng ta thấy ***k = 6*** lỗi đã thấp hơn khi sử dụng 1 hàng xóm

***Toàn bộ source code bạn xem ở [đây]()***


***Chú ý:***
Một điểm mình đã nói ở trên về cách chọn ***K***, thì ta nên chọn ***K bắt đầu từ ${log(M)}$, M là size của tập training***.

Đối chiếu 2 bài toán trên ta có:

+ Bài thứ nhất: M = 700, ta nên chọn k = 9 là điểm bắt đầu
+ Bài thứ 2: M = 70, ta nên chọn k = 6 là điểm bắt đầu 


## 5. Ưu và nhược điểm

### 5.1 Ưu điểm
+ Quá tình training không mất nhiều thời gian
+ KNN có khả năng loại bỏ nhiều khi sử dụng số hàng xóm k > 1
+ Rất linh hoạt với nhiều độ đo khác nhau

### 5.2 Nhược điêm
+ Quá trình đưa ra dự đoán, tính toán mất khá nhiều thời gian nếu tập training lớn
+ Lựa chọn K, độ đo khoảng cách, độ đo tương đồng cho phù hợp là 1 vấn đề không hề dễ dàng

## 6. Tài liệu tham khảo

+ [K-neareast Neighbors](https://www.cs.cornell.edu/courses/cs4780/2017sp/lectures/lecturenote02_kNN.html)
+ [K-nearest neighbors - Trần Quang Khoát](https://users.soict.hust.edu.vn/khoattq/ml-dm-course/L6-KNN.pdf)
+ [Khoảng cách Hamming - Wiki](https://vi.wikipedia.org/wiki/Kho%E1%BA%A3ng_c%C3%A1ch_Hamming)
