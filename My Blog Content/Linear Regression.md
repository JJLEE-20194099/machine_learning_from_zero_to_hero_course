# Linear Regression

Đến với bài này, mình sẽ giới thiệu tới các bạn mô hình cơ bản nhất và có thể nói rằng đơn giản nhất của Machine Learning. Thuật toán cho dạng mô hình này là thuật toán học có giám sát (***Supervised learning***) và thường được gọi là Hồi Quy Tuyến Tính (***Linear Regression***).

## 1. Đặt vấn đề

Các thông số của 1 bệnh nhân như tuổi (***Age***), giới tính (***Sex***), chỉ số khối cơ thể (***Body mass index***), huyết áp trung bình (***Average Blood Pressure***), và các thông số đo huyết thanh khác thì có ***chỉ số đánh giá mức độ tiến triển sau 1 năm điều trị*** là bao nhiêu . Giả sử chúng ta có thông số của 400 bệnh nhân, khi đó 1 giả thuyết đặt ra rằng: "Với các thông số tuổi, giới tính, chỉ số khối cơ thể, huyết áp trung bình, ..." thì chúng ta có thể dự đoán được chỉ số đánh giá mức độ tiến triển sau 1 năm điều trị của bệnh nhân hay không?

Điều đầu tiên chúng ta có thể nghĩ đến là tìm 1 hàm số với những thông số ***x*** cho trước sao cho đầu ra ***y*** có thể xấp xỉ với ***chỉ số đánh giá mức độ tiến triển của bệnh nhân sau 1 năm***  

Củ thể: 
+ ***x*** = [x1, x2, x3, x4, ..., x10] là 1 vector hàng trong không gian 10 chiều chứa thông tin input,  
+ ***y*** là 1 số thực biểu diễn output (tức là chỉ số tiến triển của bệnh nhân mắc bệnh tiểu đường sau 1 năm)

Khi đó bài toán dự đoán được đưa về bài toán sau:

***Học một hàm số y = f(x)*** từ 1 bộ dữ liệu cho trước D = ***{***(***$x_1$***, $y_1$), (***$x_2$***, $y_2$), ..., (***$x_M$***, $y_M$)***}*** sao cho $y_i \approx f(x_i)$ với mọi ***1 <= i <= M***

***Mô hình tuyến tính:***

\begin{equation}
\hat{y} = f(x) = w_0 + w_1x_1 + w_2x_2 + ... + w_{10}x_{10} (1)
\end{equation}

***Trong đó:***
+ ${w_1}$, ${w_2}$, ${w_3}$,..., ${w_{10}}$ là ***coefficients/weights*** (trọng số)
+ ${w_0}$ là bias

***Chú ý: Bài toán học hàm dự đoán f(x) chính là ước lượng (tối ưu) các hệ số ${w_i}$***

+ $\hat{y}$ là giá trị mô hình tuyến tính dự đoán được, ${y}$ là giá trị thực tế, chúng ta mong muốn $\hat{y} \approx y$
+ Mô hình tuyến tính khi được mô phỏng lên không gian là những hình dáng chúng ta đã học từ cấp 2 và cấp 3 đó là ***đường thẳng*** đối với khồng gian 2 chiều, ***mặt phẳng*** với không gian 3 chiều và được gọi là ***siêu phẳng*** trong không gian nhiều chiều hơn

## 2. Xác định mô hình

### 2.1 Mục đích
Mục đích của chúng ta là bằng 1 phương pháp nào đó cũng phải tìm được hàm ***$f(x) \approx y$*** và hàm số ***f(x)*** phải có khả năng tổng quát hóa trong tương lai

***Khó khăn gặp phải:***
+ Phải học hàm ***f(x)*** như thế nào trong vô số các hàm ***f(x)*** trong không gian của nó?
+ Một đại lượng củ thể có thể tính toán được và làm thước đo cho việc so sánh hàm ***f(x)*** có thực sự tốt hơn hàm ***g(x)***

### 2.2 Đại lượng tính toán sai số

***Hàm lỗi (Loss Function) (2)***

\begin{equation}
RSS(f) = {\frac{1}{M} \sum\limits_{i=1}^{M} (y_i - \hat{y}_i)^2} 
\end{equation}

***Chú ý:*** ${\hat{y_i} = f({x_i}) = {x_i}w}$, với w là bộ trọng số

### 2.3 Phương pháp OLS (Ordinary least squares)

+ Với bộ dữ liệu, tập các quan sát D cho trước, ***phương pháp OLS*** sẽ đi cực tiểu hóa ***hàm lỗi L(w)***

\begin{equation}
f^* =  arg \min\limits_{f \in S} RSS(f)
\end{equation}

+ Việc tối ưu hàm mất mát sẽ giúp ta tìm được điểm tối ưu (các trọng số tối ưu trong mô hình hồi quy) ***(3)***

\begin{equation}
w^* =  arg \min\limits_{w} \sum\limits_{i=1}^M (y_{i} - w_{0} - w_{1}x_{i1} - ... - w_{n}x_{in})^2
\end{equation}

***Chú ý: mẫu quan sát ${x_i}$ có n chiều*** 

#### 2.4 Nghiệm của bài toán

Như chúng ta đã biết, để tìm cực trị của 1 hàm số thì đạo hàm tại điểm đó phải bằng 0 hoặc không xác định đạo hàm tại điểm đó và đổi dấu đạo hàm khi đi qua điểm đó.

Hơn thế nữa, do mô hình tuyến tính khá đơn giản nên việc tính đạo hàm khá dễ dàng

+ Phương pháp tối ưu hàm lỗi còn được gọi là phương pháp bình phương tối thiểu
+ Nghiệm tối ưu ${w^{*}}$ được tìm bằng cách giải phương trình ${RSS^{'}} = 0$ và sau khi giải được ta có nghiệm duy nhất là:
\begin{equation}
w^* = (A^TA)^{-1}A^Ty (4)
\end{equation}
***Trong đó:***
    + M là số lượng mẫu dữ liệu, n là số chiều của 1 mẫu
    + A là ma trận có kích thước ***Mx(n + 1)***, với hàng i là ***${A_i} = (1, x_{i1}, x_{i2}, ..., x_{in})$***; $X^{-1}$ gọi là ma trận khả nghịch của X; ***$y = (y_{1}, y_{2}, ..., y_{M})^T$***
+ Phương pháp này chỉ thành công khi ma trân ${A^TA}$ khả nghịch
+ Sau khi tìm được điểm tối ưu ${w^*}$ thì ta dự đoán cho 1 quan sát mới ***x = ${x_{1}, x_{2}, ..., x_{n}}$*** như sau: 
\begin{equation}
y_x = w_0^* + w_1^*x_1 + ... + w_n^*x_n(5)
\end{equation}

***Nhược điểm:***
+ Phương pháp OLS sẽ không thể thực hiện được nếu ma trận ${A^TA}$ không ***khả nghịch (invertible)***. Điểu này xảy ra các thuộc tính của bộ dữ liệu hay còn gọi là các cột tồn tại 1 sự phụ thuộc vào nhau tuyến tính vào nhau
+ Việc có được ma trận nghich đảo cần chi phí tính toán cao
+ Việc cố gắng tìm nghiệm tối ${w^*}$ thỏa mãn ${RSS^{'}} = 0$ dẫn đến hiện tượng overfitting vì quan trình học chỉ tập trung vào việc tối ưu hàm lỗi trên tập train (Đọc thêm vấn đề này tại đây)

## 3.Thực hành

### 3.1 Giới thiệu bài toán

Trong bài học này, mình sẽ giới thiệu tới các bạn một ví dụ cơ bản như sau:

Chúng ta có thông số của 442 bệnh nhân, với thông tin của từng bệnh nhân bao gồm ***Tuổi(Age), Giới tính(Sex), Chỉ số khối cơ thể(Body Mass Index), Huyết Áp Trung Bình (Average Blood Pressure BP), các chỉ số huyết thanh (S1, S2, S3, S4, S5, S6)***, và  1 cột cuối cùng đánh giá mức độ tiến triển của bệnh nhân mắc bệnh tiểu đường sau 1 năm điểu trị.

![](https://i.imgur.com/CJoAgfN.png)
***Bộ dữ liệu đầy đủ ở [đây](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt)***

Chúng ta cần dự đoán được mức độ tiến triển của bệnh nhân mắc bệnh tiểu đường sau 1 năm điều trị (cột 11) dựa vào thông số của 10 cột đầu tiên.

### 3.2 Giải quyết bài toán

#### 3.2.1 Khai báo thư viện cần sử dụng

+ scikit-learn(sklearn): thư viện hỗ trợ phong phú các mô hình học máy cùng với các hàm đánh giá và huấn luyện đa dạng
+ pandas: thư viện xử lý dữ liệu dạng bảng
+ numpy: thư viện cho đại số tuyến tính dùng để biến đổi ma trận, làm việc với vector
+ matplotlib: thư viện phục vụ vẽ các đồ thị
+ math: thư viện hỗ trợ các hàm số tính toán

```python=
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
```

#### 3.2.2 Đọc dữ liệu 

```python=
diabetes = pd.read_csv('./data.txt', sep='\t')
print("Số chiều của bộ dữ liệu: ", diabetes.shape)
```


#### 3.2.3 Điều chỉnh tỷ lệ bộ dữ liệu

Khi bộ dữ liệu có nhiều cột, hay nói cách khác có nhiều thuộc tính khác nhau và bên cạnh đó ứng với mỗi thuộc tính lại có độ lớn, min, max, đơn vị khác nhau, vì vậy điều này ảnh hưởng tới độ chính xác, quá trình hội tụ và thời gian tính toán của thuật toán. Chính vì vậy giải phát đơn giản nhất là đưa các đặc trưng này về chung một tỷ lệ nhất định. Đó là ***Data Scaling***. Và thông thường ta đưa giá trị các thuộc tính về khoảng giá trị chung [0, 1]. 

***Trong bài viết này mình sử dụng [MinMaxScaler của gói preprocessing](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) trong sklearn để điều chỉnh tỷ lệ bộ dữ liệu***

```python=
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

def scale_columns(df, cols):
    for col in cols:
        df[col] = pd.DataFrame(mms.fit_transform(pd.DataFrame(df[col])), columns=[col])
    return df

diabetes_scaled = scale_columns(diabetes, diabetes.columns[:-1])
diabetes_arr = diabetes_scaled.to_numpy()
```

***Chú ý:*** Mình chỉ scale lại 10 cột đầu tiên của bộ dữ liệu và sẽ giữ nguyên cột cuối cùng.

#### 3.2.4 Tách bộ dữ liệu

Đối với [bộ dữ liệu](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt) này gồm 442 mẫu dữ liệu mình tách chúng ra ***400 mẫu để train*** và ***42 mẫu để test***

```python=
diabetes_train = {
    'data': diabetes_arr[:400, :-1],
    'target': diabetes_arr[:400, -1]
}

diabetes_test = {
    'data': diabetes_arr[400:, :-1],
    'target': diabetes_arr[400:, -1]
}
```

#### 3.2.5. Tính nghiệm tối ưu theo công thức

Trong mục này, mình sẽ tính nghiệm tới ưu w = [${w_0}$, ${w_1}$, ..., ${w_{10}}$] theo công thức số (5) ở phía trên bài viết.

```python=
# Chèn giá trị 1 vào đầu của mỗi mẫu dữ liệu quan sát, giá trị này tương ứng với trọng số w0 
one_train_arr = np.ones((diabetes_train['data'].shape[0], 1))
A_train = np.concatenate((one_train_arr, diabetes_train['data']), axis=1) 

#Tính toán w theo công thức đã cho
Q = np.dot(A_train.T, A_train)
J = np.dot(A_train.T, diabetes_train['target'])
w = np.dot(np.linalg.pinv(Q), J)

print("[w0, w1,..., wn] = ", w)

```

```python=
[w0, w1,..., wn] = [   1.89845203    1.09602218  -22.72575768  136.05975295   73.31637773
 -211.10687859  139.94765717   23.67057925   48.55148826  183.46255187
   24.24508984]
```

Bây giờ, ta dùng bộ số tới ưu w này để đi dự đoán cho 42 mẫu bệnh nhân còn lại, in ra kết quả thực tế và dự đoán của 7 bệnh nhân trong 42 mẫu bệnh nhân trên:

```ruby=
one_test_arr = np.ones((diabetes_test['data'].shape[0], 1))
A_test = np.concatenate((one_test_arr, diabetes_test['data']), axis=1) 

diabetes_test_target_predicted = np.dot(A_test, w)

print("Kết quả dự đoán chỉ số của 7 bệnh nhân đầu tiên trong 42 bệnh nhân của tập test: \n", diabetes_test_target_predicted[:7])
print("Kết quả thực tế chỉ số của 7 bệnh nhân đầu tiên trong 42 bệnh nhân của tập test: \n", diabetes_test['target'][:7])
```

```python=
Kết quả dự đoán chỉ số của 7 bệnh nhân đầu tiên trong 42 bệnh nhân của tập test: 
 [185.39410409  90.34025895 152.32680043 250.8659631  198.45798721
 281.11608385  50.83212934]
Kết quả thực tế chỉ số của 7 bệnh nhân đầu tiên trong 42 bệnh nhân của tập test: 
 [175.  93. 168. 275. 293. 281.  72.]
```

***Nhìn qua kết quả trên, chỉ số dự đoán tương đối gần với chỉ số thực tế***

#### 3.2.6 Tính nghiệm tối ưu bằng thư viện

Trong mục này, mình sẽ tính toán nghiệm bằng cách sử dụng thư viện scikit-learn, củ thể là ***gói linear_model***

```python=
# Train mô hình bằng hàm fit
regr = linear_model.LinearRegression()
regr.fit(diabetes_train['data'], diabetes_train['target'])

print("[w1, w2,..., wn] = ", regr.coef_)
print("w0 = ", regr.intercept_)
```

```python=
[w1, w2,..., wn] =  [   1.09602218  -22.72575768  136.05975295   73.31637773 -211.10687859
  139.94765717   23.67057925   48.55148826  183.46255187   24.24508984]
w0 =  1.8984520285978874
```

Bây giờ, ta dùng bộ số tới ưu w này để đi dự đoán cho 42 mẫu bệnh nhân còn lại, in ra kết quả thực tế và dự đoán của 7 bệnh nhân trong 42 mẫu bệnh nhân trên:

```python=
diabetes_test_target_predicted = regr.predict(diabetes_test['data'])

print("Kết quả dự đoán chỉ số của 7 bệnh nhân đầu tiên trong 42 bệnh nhân của tập test: \n", diabetes_test_target_predicted[:7])
print("Kết quả thực tế chỉ số của 7 bệnh nhân đầu tiên trong 42 bệnh nhân của tập test: \n", diabetes_test['target'][:7])
```

```python=
Kết quả dự đoán chỉ số của 7 bệnh nhân đầu tiên trong 42 bệnh nhân của tập test: 
 [185.39410409  90.34025895 152.32680043 250.8659631  198.45798721
 281.11608385  50.83212934]
Kết quả thực tế chỉ số của 7 bệnh nhân đầu tiên trong 42 bệnh nhân của tập test: 
 [175.  93. 168. 275. 293. 281.  72.]
```

***[Toàn bộ code ở đây]()***

## 3. Nhược điểm của mô hình hồi quy tuyến tính
+ Một số trường hợp dễ xảy ra hiện tượng overfitting và không có khả năng tổng quát hóa trong tương lai, mình sẽ giới thiệu các bạn khái niệm hiệu chỉnh cùng với 2 mô hình hồi quy Ridge và hồi quy LASSO trong bài viết tiếp theo
+ Mô hình hình hồi quy rất nhạy cảm với nhiễu hay còn gọi là ***Outlier***

![](https://i.imgur.com/YHlR9Vq.png)


Để giải quyết vấn đề này chúng ta cần có các bước tiền xử lý để lọc và loại bỏ nhiều trước khi đưa vào mô hình Linear Regression

+ Với các bộ dữ liệu không có quan hệ tuyến tính ***(Non Linear Relationship)*** thì phương pháp Linear Regression không thể sử dụng được vì dữ liệu trong thực tế nó rất phức tạp

![](https://i.imgur.com/J4bY615.png)

## 4. Tài liệu tham khảo

+ [Linear Regression Analysis using SPSS Statistics](https://statistics.laerd.com/spss-tutorials/linear-regression-using-spss-statistics.php)
+ [Linear Regression](https://machinelearningcoban.com/2016/12/28/linearregression/)






