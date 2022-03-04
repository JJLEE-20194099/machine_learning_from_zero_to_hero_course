# Regularization
Đến với bài viết này, mình sẽ giới thiệu tới các bạn kỹ thuật hiệu chỉnh. Đây là một lỹ thuật khá quan trọng và đóng vai trò lớn trong việc giảm số lượng các mô hình cần huyến luyện và phần nào tránh được hiện tượng ***overfitting*** (khái niệm mà mình sẽ giới thiệu ngay dưới đây)

## 1. Overfitting và Underfitting

![](https://i.imgur.com/pzLDH1b.png)


***Chú ý: Độ lỗi trên tập train và test dựa trên độ phức tạp của mô hình***

Khi chúng ta tăng độ phức tạp của mô hình thì lỗi trên tập test cũng có xu hướng giảm dần, tuy nhiên khi chúng ta tăng tiếp độ phức tạp của mô hình lên nữa thì có thể độ lỗi sẽ tăng lên và mô hình có thể rất tệ. Đây được gọi là ***Overfitting***


***Overfitting*** là hiện tượng mà mô hình chúng ta tìm được quá  phù hợp ***(fit)*** với dữ liệu. Hiện tượng này đồng nghĩa với việc mô hình của chúng ta không tốt vì mô hình thay vì chỉ tập trung cố gắng mô tả được dữ liệu thì mô hình đã mô tả luôn cả nhiễu ***(noise)*** của dữ liệu

***Underfitting*** là hiện tượng xảy ra khi chúng ta cố gắng mô tả dữ liệu phức tạp bằng các mô hình đơn giản ví dụ như mô hình hồi quy tuyến tính chúng ta đã học ở bài học trước. Mô hình gặp phải hiện tượng này đồng nghĩa với việc mô hình không học được các trọng số w quan trọng hay mô hình xây dựng chưa có độ chính xác cao đối với dữ liệu huấn luyện và kiểm tra.

Một trong những phương pháp rất phổ biến và hiệu quả trong học máy hiện nay là hiệu chỉnh ***(Regularization)***

## 2. Giới thiệu hiệu chỉnh (Regularization)
Regularization là 1 kỹ thuật mà chúng ta cố gắng đưa các tri thức bên ngoài vào và các tri thức này có thể đến từ nhiều nguồn khác nhau.

Mục đích của phương pháp này là bằng cách nào đó có thể giảm được hiện tượng overfitting trong Machine Learning.

Có rất nhiều góc nhìn khác nhau khi chúng ta làm hiểu chỉnh và việc sử dụng ***kỹ thuật hiệu chỉnh*** đồng nghĩa với việc chúng ta đưa thêm các ***tri thức bên ngoài*** vào mô hình bằng cách ***mã hóa*** thành ***1 đại lượng phạt lên sự phức tạp của mô hình (penalty on the complexity)*** chúng ta cần học.

***Chú ý:*** Khi mà 1 hàm phức tạp hay 1 mô hình phức tạp thì chúng sẽ bị phạt nhiều hơn

## 3. Hiệu chỉnh trong các mô hình hồi quy

Như chúng ta đã biết ở bài học hồi quy tuyến tính, ***phương pháp bình phương tối thiểu OLS (Ordinary Least Squares)*** sẽ đi tối thiểu mỗi hàm lỗi để tìm ra nghiệm tối ưu:

\begin{equation}
w^* =  arg \min\limits_{w} \sum\limits_{i=1}^M (y_{i} - w_{0} - w_{1}x_{i1} - ... - w_{n}x_{in})^2
\end{equation}

Vì thế mà phương pháp ***OLS*** rất dễ gây ra hiện tượng overfitting.

Do vậy sau đây mình sẽ đưa các đại lượng hiệu chỉnh, và tương ứng với ***từng kỹ thuật hiểu chỉnh*** ta sẽ có ***từng mô hình hình khác nhau*** 

***Nhắc lại:***
+ Xét tập dữ liệu ${D = {(x_{1}, y_{1}), (x_{2}, y_{2}), ...,(x_{M}, y_{M})}}$
+ Mỗi quan sát là 1 vector n chiều:
${x_i} = ({x_{i1}}, {x_{i2}}, ...,{x_{in}})$
+ ${A_i} = (1, {x_{i1}}, {x_{i2}}, ...,{x_{in}})$
+ Hàm lỗi thực nghiệm:
\begin{equation}
RSS(f) = {\frac{1}{M} \sum\limits_{i=1}^{M} (y_i - \hat{y}_i)^2} 
\end{equation}
***Chú ý:*** ${\hat{y_i} = f({x_i}) = {x_i}w}$, với w là bộ trọng số


### 3.1 Phương pháp hồi quy Ridge

Thay vì chỉ đi tối ưu mỗi hàm lỗi, thì chúng ta đưa vào 1 đại lương hiểu chỉnh là chuẩn ${L^2}$ ***(${L^2}$ norm)*** đối với bộ trong số ***w*** (ký hiệu là: ***${||w||_{2}^2}$***), cùng với đó là 1 hằng số hiệu chỉnh (hằng số phạt) cho trước ${\lambda > 0}$ ***(Regularization Constant)*** để tiến hành tìm nghiệm tối ưu ${w^*}$, bài toán (1): 
\begin{equation}
w^* = arg \min\limits_{w} \sum\limits_{i = 1}^M(y_i - A_iw)^2 + \lambda \sum\limits_{j = 0}^n w_{j}^2
\end{equation}

Công thức trên chính là ***bài toán của hồi quy Ridge***

Vậy mục đích củ thể của hiểu chỉnh trong công thức trên là gì:

+ Khi ${\lambda}$ rất lớn, giả sử có 1 trọng số ${w_i}$ lớn một chút thì tổng ${\lambda \sum\limits_{j = 0}^nw_{j}^2}$ áp đảo ***hàm lỗi cực tiểu RSS(f)***, nghĩa là khi chúng ta cực tiểu hóa theo công thức trên thì thành phần lỗi thực nghiệm không có nhiều ý nghĩa trong việc tối ưu. Do vậy để việc cực tiểu hàm ${\sum\limits_{i = 1}^M(y_i - A_iw)^2 + \lambda \sum\limits_{j = 0}^n w_{j}^2}$ mang nhiều ý nghĩa thì ***w*** phải bé. Hay mô hình có khả năng gặp trường hợp ***underfitting***

+ Hằng số ${\lambda}$ sẽ có ý nghĩa phạt độ lớn của ***w***. Hay nói cách khác, Hằng số ${\lambda}$ sẽ liên quan đến độ phức tạp của mô hình.

Bài toán (1) tương đương với bài toán (2) sau:

\begin{equation}
w^* = arg \min\limits_{w} \sum\limits_{i = 1}^M(y_i - A_iw)^2
\end{equation}

\begin{equation}
\text{Với điều kiện: }\sum\limits_{j = 0}^n w_{j}^2 <= t, \text{t là hằng số}
\end{equation}

Từ bài toán (2):
+ Ta có thể thấy rõ hơn kỹ thuật hiểu chỉnh giúp chúng ta giảm không gian tìm kiếm giá trị tối ưu ***${w^*}$***, đồng nghĩa với ảnh hưởng tới quá trình hội tụ của mô hình

+ Kỹ thuật hiệu chỉnh giúp chúng ta giảm ảnh hưởng của lỗi nhiễu, củ thể việc giảm không gian tìm kiếm của ***w*** khiến cho mô hình của chúng ta sẽ không bị các điểm lỗi, nhiễu ***(outliers)*** kéo lệch khỏi vùng dữ liệu chuẩn.

+ Ta có thể thấy rằng có thể mô hình của chúng ta khi thêm hiệu chỉnh sẽ giảm độ khớp ***(fitting)*** với tập dữ liệu huấn luyện ***D***, tuy nhiên mô hình lại tăng được ***khả năng tổng quát hóa*** trong tương lai


Từ bài toán (1), việc tính nghiệm tối ưu ***${w^*}$*** bằng cách tích đạo hàm và giải phương trình đó bằng 0 (mình đã nói ở bài viết trước), ở đây chúng ta có nghiệm cho bài toán (1) là:

\begin{equation}
w^* = (A^TA + \lambda I_{n + 1})^{-1}A^Ty
\end{equation}

Trong đó: 
+ A là ma trận có ***size Mx(n+1)*** và ${A_i} = (1, {x_{i1}}, {x_{i2}}, ...,{x_{in}})$
+ ${I_{n+1}}$ là ma trận đơn vị
+ ${B^{-1}}$ là ma trận khả nghịch
+ ${y} = ({y_{1}}, {y_{2}}, ...,{y_{M}})^T$

***Chú ý:*** Ma trận ${(A^TA + \lambda I_{n + 1})^{-1}}$ luôn tồn tại nghịch đảo với ${\lambda > 0}$

### 3.2 Phương pháp hồi quy LASSO

Như chúng ta đã đề cập ở trên, hồi quy Ridge ***(Ridge Regression)*** sử dụng hiệu chỉnh ***${L^2}$*** và khi chúng ta thay hiểu chỉnh ***${L^2}$*** bằng hiểu chỉnh ***${L^1}$*** thì ta được một phương pháp khác có tên là hồi quy LASSO ***(Lasso Regression)***

Bài toán cần giải quyết đới với phương pháp hồi quy Lasso như sau (3):

\begin{equation}
w^* = arg \min\limits_{w} \sum\limits_{i = 1}^M(y_i - A_iw)^2 + \lambda \sum\limits_{j = 0}^n |w_{j}|
\end{equation}


***hoặc***

\begin{equation}
w^* = arg \min\limits_{w} \sum\limits_{i = 1}^M(y_i - A_iw)^2
\end{equation}

\begin{equation}
\text{Với điều kiện: }\sum\limits_{j = 0}^n |w_{j}| <= t, \text{t là hằng số}
\end{equation}

***Chú ý:***
+ Ý nghĩa của ${\lambda}$ sẽ giống như ***hồi quy Ridge***
+ Đối với hồi quy LASSO thì hàm lỗi phức tạp hơn nhiều so với 2 phương pháp chúng ta đã biết là bình phương tối thiêủ ***(OLS)*** và ***hồi quy Ridge***, sở dĩ hàm lỗi của phương pháp ***hồi quy LASSO*** không tồn tại đạo hàm tại 1 số điểm do chuẩn ***${L^1}$***

+ Nếu các bạn đã biết về bài toán quy hoạch lồi trong tối ưu thì nếu ${x^*}$ là nghiệm tối ưu địa phương thì ${x^*}$ cũng là điểm tối ưu toàn cục. Đối chiều bài toán ***(3)***:
    + Hàm ${\sum\limits_{i = 1}^M(y_i - A_iw)^2}$ là hàm lồi theo ${w}$
    + Tập xác định: ${\sum\limits_{j = 0}^n |w_{j}| <= t, \text{t là hằng số}}$ cũng là tập lồi theo ${w}$
    + Vì vậy bài toán ***(3)*** là bài toán quy hoạch lồi

+ Việc tính nghiệm tối ưu theo phương pháp hồi quy ***Lasso*** bằng cách lặp (mình sẽ giới thiệu phương pháp phổ biến giảm ngược hướng đạo hàm ***Gradient Descent*** những bài viết tiếp tới)

***Điểm gì khiến cho phương pháp LASSO trở nên nổi tiếng?***

![](https://i.imgur.com/4ZJnzJj.png)

***(Hình ảnh được tham khảo trong bài giảng của Xuezhi Wang [ở đây](http://alex.smola.org/teaching/cmu2013-10-701/slides/13_recitation_lasso.pdf))***

Tập xác định đối với phương pháp hồi quy ***Ridge*** (bài toán (2)) là ***hình cầu*** còn tập xác định đối với phương pháp hồi quy ***Lasso*** (bài toán (3)) là ***hình thoi***.

Phương pháp ***hồi quy Lasso*** thường tìm được nghiệm tối ưu thưa (có nhiều hệ số ${w_i^*} = 0$), mặc dù ***hồi quy Lasso*** vẫn có thể tìm được những nghiệm thưa như vậy tuy nhiên ***xác suất là khá thấp.***

 Mình sẽ nhắc lại 1 ít kiến thức để giải thích vấn đề trên như sau:

+ Điểm cực biên của ${D}$ là điểm không nằm giữa hai điểm nào trong tập, hay nói cách khác: x là điểm cực biên của ${D}$ nếu không tồn tại 2 điểm a, b và ${\lambda \in (0, 1)}$ sao cho ${x = \lambda a + (1 - \lambda)b}$

+ Đầu tiên ta có 1 tính chất sau: Nếu một hàm lồi đạt ***tối ưu*** trên 1 tập lồi có điểm cực biên thì điểm tối ưu đó sẽ là môt điểm cực biên của tập lồi đó ***(${*}$)***

Như chúng ta thấy 2 phương pháp ***hồi quy Ridge*** và ***hồi quy Lasso*** đều đối mặt với 2 bài toán quy hoạch lồi tuy nhiên điểm khác biệt lớn nhất là:

+ Tập lồi của bài toán trong phương pháp ***hồi quy Ridge*** có vô số điểm cực biên ***(các điểm trên biên hình cầu)*** và xác suất gặp 1 điểm cực biên thưa là rất ít

+ Tập lồi của bài toán trong phương pháp ***hồi quy Lasso*** có hữu hạn điểm cực biên ***(các đỉnh của hình thoi)*** và mỗi điểm cực biên của tập đều là các điểm có 1 số chiều có giá trị = 0

Vì vậy theo tính chất ***(${*}$)*** ta đã chứng minh được rằng ***phương pháp hồi quy Lasso*** thường tìm thấy các nghiệm thưa ***(spare solutions)***, và xác suất cao hơn nhiều so với ***phương pháp hồi quy Ridge***


***Việc tìm được các nghiệm thưa ${w^* = (w_0^*, w_1^*, ...,w_n^*)}$ giúp chúng ta chọn được các thuộc tính nào quan trọng, thuộc tính nào không quan trọng dựa vào ${w_i^* \neq 0}$ để thực hiện đưa vào hàm f(x)***

## 4. Thực hành 

### 4.1 Giới thiệu bài toán
Chúng ta có thông số của 67 quan sát về bệnh ung thư tuyến tiền liệt ***(prostate cancer)*** (mình cũng không biết bệnh này vì mình thi môn sinh tốt nghiệp được 1,25 điểm), bộ dữ liệu có 8 thuộc tính ứng với 8 cột đầu tiên và cột cuối cùng dùng để dự đoán.

![](https://i.imgur.com/mfhJVY3.png)

***Bộ dữ liệu đầy đủ ở [đây]()***

### 4.2 Giải quyết bài toán
#### 4.2.1 Khai báo thư viện cần sử dụng

+ scikit-learn(sklearn): thư viện hỗ trợ phong phú các mô hình học máy cùng với các hàm đánh giá và huẩn luyện đa dạng
+ pandas: thư viện xử lý dữ liệu dạng bảng
+ numpy: thư viện cho đại số tuyến tính dùng để biến đổi ma trận, làm việc vơi vector
+ matplotlib: thư viện phục vụ vẽ các đồ thị
+ math: thư viện hỗ trợ các hàm số tính toán
+ tqdm: thư viện giúp ta biết được tiến độ chạy, ví dụ vòng for đã chạy được bao nhiêu %

```python=
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
```

#### 4.2.2 Đọc dữ liệu

```python=
prostate_cancer_df = pd.read_csv('./prostate_cancer.csv', sep=',')
print("Số chiều của bộ dữ liệu: ", prostate_cancer_df.shape)
print(prostate_cancer_df)
```

#### 4.2.3 Điều chỉnh tỷ lệ bộ dữ liệu

Kiểm tra thông tin min, max,... của các thuộc tính:

```python=
prostate_cancer_df.describe()
```

![](https://i.imgur.com/GrPUNcy.png)

Ta thấy thuộc tính ***age(cột 3)*** có giá trị min, max và trung bình khá lớn, nên mình quyết định scale lại thuộc tính này

```python=
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

def scale_columns(df, cols):
    for col in cols:
        df[col] = pd.DataFrame(mms.fit_transform(pd.DataFrame(df[col])), columns=[col])
    return df

prostate_cancer_scaled = scale_columns(prostate_cancer_df, [prostate_cancer_df.columns[2]])
prostate_cancer_arr = prostate_cancer_scaled.to_numpy()
```

#### 3.2.4 Tách bộ dữ liệu

Đối với [bộ dữ liệu]() này gồm 97 mẫu dữ liệu mình tách chúng ra ***67 mẫu để train*** và ***30 mẫu để test***

```python=
prostate_cancer_train = {
    'data': prostate_cancer_arr[:67, :-1],
    'target': prostate_cancer_arr[:67, -1]
}

prostate_cancer_test = {
    'data': prostate_cancer_arr[67:, :-1],
    'target': prostate_cancer_arr[67:, -1]
}
```

#### 3.2.5 Tính nghiệm tối ưu bằng thư viện

Trong mục này, mình sẽ tính toán nghiệm bằng cách sử dụng thư viện scikit-learn, củ thể là ***gói linear_model***

##### 3.2.5.1 Mô hình Linear Regression

Huấn luyện mô hình:
```python=
linear_regr = linear_model.LinearRegression()
linear_regr.fit(prostate_cancer_train['data'], prostate_cancer_train['target'])
print("[w1, w2,..., wn] = ", linear_regr.coef_)
print("w0 = ", linear_regr.intercept_)
```

```typescript=
[w1, w2,..., wn] =  [ 0.43781781  0.57463133 -0.96486058  0.15850887 -0.30144711 -0.1077378
  0.24364124  0.00309586]
w0 =  -1.6998065617549987
```

Dự đoán kết quả trên tập test và tính toán lỗi dựa trên độ đo RMSE ***(Lỗi trung bình bình phương đo sự khác biệt giữa các giá trị dự đoán và giá trị thực tế)***:

```python=
prostate_cancer_test_target_predicted = linear_regr.predict(prostate_cancer_test['data'])
rmse = math.sqrt(mean_squared_error(prostate_cancer_test['target'], prostate_cancer_test_target_predicted))
print(f'RMSE = {rmse}')
```

```typescript=
RMSE = 1.4846333766283948
```

##### 3.2.5.2 Mô hình Ridge Regression

Huấn luyện mô hình:

```python=
ridge_reger = linear_model.Ridge(alpha=5)
ridge_reger.fit(prostate_cancer_train['data'], prostate_cancer_train['target'])
print("[w1, w2,..., wn] = ", ridge_reger.coef_)
print("w0 = ", ridge_reger.intercept_)
```

```python=
[w1, w2,..., wn] =  [ 0.388475    0.33012034 -0.21776869  0.15313623 -0.09345342 -0.09010693
  0.15651654  0.00316383]
w0 =  -0.6433202760629273
```

Dự đoán kết quả trên tập test và tính toán lỗi dựa trên độ đo RMSE ***(Lỗi trung bình bình phương đo sự khác biệt giữa các giá trị dự đoán và giá trị thực tế)***:

```python=
prostate_cancer_test_target_predicted = ridge_reger.predict(prostate_cancer_test['data'])
rmse = math.sqrt(mean_squared_error(prostate_cancer_test['target'], prostate_cancer_test_target_predicted))
print(f'RMSE = {rmse}')
```

```typescript=
RMSE = 1.4328803753886048
```

***Chú ý:***
+ Tại ***alpha=5*** thì lỗi trên tập test là thấp nhất
![](https://i.imgur.com/x36rSRZ.png)

+ Dưới đây là ảnh hưởng của ***alpha** lên các thuộc tính của bộ dữ liệu:
![](https://i.imgur.com/9lmEgsL.png)
     + Khi ***alpha = 0*** thì phương pháp hồi quy Ridge sẽ trở thành phương pháp OLS (bình phương tối thiểu)
     + Khi ***alpha tăng dần*** thì vùng giá trị của ***w*** và khi ***alpha đủ lớn*** thì ***w dần tiến về 0*** (trường hợp này mô hình bị ***underfitting***)

+ Kết quả RMSE ở phương pháp hồi quy Ridge đã thấp hơn hồi quy tuyễn tính

##### 3.2.5.3 Mô hình Laso Regression

Huấn luyện mô hình:

```python=
lasso_regr = linear_model.Lasso(alpha=0.1)
lasso_regr.fit(prostate_cancer_train['data'], prostate_cancer_train['target'])
print("[w1, w2,..., wn] = ", lasso_regr.coef_)
print("w0 = ", lasso_regr.intercept_)
```

```python=
[w1, w2,..., wn] =  [ 0.30384977  0.         -0.          0.14273201 -0.         -0.
  0.          0.00536446]
w0 =  1.52756804010382523
```

Dự đoán kết quả trên tập test và tính toán lỗi dựa trên độ đo RMSE ***(Lỗi trung bình bình phương đo sự khác biệt giữa các giá trị dự đoán và giá trị thực tế)***:

```python=
prostate_cancer_test_target_predicted = lasso_regr.predict(prostate_cancer_test['data'])
rmse = math.sqrt(mean_squared_error(prostate_cancer_test['target'], prostate_cancer_test_target_predicted))
print(f'RMSE = {rmse}')
```

```typescript=
RMSE = 1.429488124089856
```

***Chú ý:***
+ Tại ***alpha=0.1*** thì lỗi trên tập test là thấp nhất
![](https://i.imgur.com/Gjfxs5y.png)


+ Dưới đây là ảnh hưởng của ***alpha** lên các thuộc tính của bộ dữ liệu:
![](https://i.imgur.com/iYjIVCR.png)


+ Kết quả RMSE ở phương pháp hồi quy Lasso đã thấp hơn hồi quy tuyễn tính, và thấp hơn hồi quy Ridge. Hai mô hình hồi quy Ridge và hồi quy Lasso sẽ phù hợp với từng kiểu bài toán riêng. Có khi đối với kiểu dữ liệu này thì mô hình hồi quy Ridge tốt hơn, có khi đối với kiểu dữ liệu khác thì mô hình hồi quy Ridge lại tệ hơn

***Sự so sánh tương quan giữa 3 phương pháp OLS, hồi quy Ridge và hồi quy Lasso về phương diện nghiệm tối ưu ${w^*}$ tìm được:***


|w | OLS | Ridge | Lasso|
| -------- | -------- | -------- |--------|
| ${w_0}$     |   -1.6998   | -0.6433     |1.5276|
|   lcavol   | 0.4378     | 0.3885     |0.3038|
|    lweight  | 0.5746     | 0.3301     |0|
|     age |  -0.9649     | -0.2178     |0|
|     lbph | 0.1585     | 0.1531     |0.1427|
|     svi | -0.3014     | -0.0934     |0|
|     lcp | -0.1077     | -0.0901  |0|
|     gleason | 0.2436     |  0.1565     |0|
|     pgg45 | 0.0031     | 0.0032     |0.0054|
|     RMSE | 1.4846     | 1.4329     |1.4295|

***Như chúng ta đã nhận xét những note phía trên, hai phương pháp Ridge và Lasso đã tốt hơn phương pháp hồi quy tuyến tính thông thường và chúng ta thấy nghiệm tối ưu tìm được trong phương pháp hồi quy Lasso là nghiệm thưa có nhiều trong số ${w_i = 0}$***

## 5. Tài liệu tham khảo

+ [Avoid Overfitting - SHADAB HUSSAIN](https://www.kaggle.com/shadabhussain/avoid-overfitting)
+ [Ridge/Lasso Regression, Model selection, Xuezhi Wang](http://alex.smola.org/teaching/cmu2013-10-701/slides/13_recitation_lasso.pdf)
+ [Luận văn hàm lồi và tập lồi](http://math.ac.vn/training/images/TTDaotao/Caohoc/Luanvan/19_Ha_Thi_Thao.pdf)
+ [Overfitting - Machine learning cơ bản](https://machinelearningcoban.com/2017/03/04/overfitting/)
