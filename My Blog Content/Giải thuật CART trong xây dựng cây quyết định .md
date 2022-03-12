# Giải thuật CART trong xây dựng cây quyết định 

Nếu bạn chưa biết cây quyết định là gì thì mình nghĩ bạn nên đọc bài viết [Decision Tree và thuật toán ID3]() trước khi đọc bài viết này.

Mình giả sử các bạn đều đã biết cây quyết định và mục đích của bài viết này là tìm hiểu ***giải thuật CART cơ bản*** mà sklearn dựa trên ý tưởng này để xây dựng mô hình cây quyết định


## 1. Xây dựng cây quyết định bằng giải thuật CART

Việc xây dựng cây quyết định đồng nghĩa với việc học 1 mô hình từ dữ liệu tập huấn luyện

Mục tiêu tại mỗi nút, thuật toán ***CART*** tìm ra các điều kiện giúp cây quyết định có thể phân loại tốt nhất bộ dữ liệu. Những điều kiện như vâỵ gọi là điều kiện có tính tác biệt ***(discriminative)***

### 1.1 Thế nào là thuộc tính có tính tách biệt

![](https://i.imgur.com/fLp0AnH.png)
***(Hình ảnh lấy tại [bài giảng ML and DM](https://users.soict.hust.edu.vn/khoattq/ml-dm-course/L7-Random-forests.pdf) của thấy Trần Quang Khoát - ĐH BKHN)***

Bộ dữ liệu training có ***60 mẫu dữ liệu*** với:

+ 35 mẫu thuộc lớp ***c1***, 25 mẫu thuộc lớp ***c2***
+ mẫu dữ liệu có 2 thuộc tính cần được xét đến là ***A1, A2***

Ứng với mỗi nhánh sau khi chọn thuộc tính, dữ liệu sẽ được phân lớp lại vào các ***node*** con ứng với mỗi nhánh đó. Ví dụ:

+ Nếu chúng ta chọn thuộc tính ***A1*** và đi theo nhánh ***v11***, thì bây giờ dữ liệu cần phân loại của chúng ta bao gồm ***30 mẫu dữ liệu*** với ***21 mẫu lớp c1*** và ***9 mẫu lớp c2***

+ Nếu chúng ta chọn thuộc tính ***A2*** và đi theo nhánh ***v21***, thì bây giờ dữ liệu cần phân loại của chúng ta bao gồm ***33 mẫu dữ liệu*** với ***27 mẫu lớp c1*** và ***6 mẫu lớp c2***

Một ***thuộc tính*** tại ***mỗi đỉnh*** được gọi là phân biệt nếu tạo đỉnh đó chúng ta có khả năng phân loại dữ liệu tốt nhất so với với các ***thuộc tính khác***


Như chúng ta thấy:

+ Nếu chọn thuộc tính ***A2*** và đi theo cả 2 nhánh ***v21*** hoặc ***v22*** thì chúng ta có thể tách biệt dữ liệu bởi vì theo nhánh ***v21*** thì xác suất 1 mẫu rơi vào phân lớp ***c1*** là $\frac{27}{33}$ rất cao và tương tự xác suất 1 mẫu rơi vào phân lớp ***c2*** là $\frac{19}{27}$ rất cao

+ Nếu chọn thuộc tính ***A1*** và đi theo cả 2 nhánh ***v12*** hoặc ***v13*** thì chúng ta ***khó*** có thể tách biệt dữ liệu bởi vì xác suất 1 mẫu dữ liệu rơi vào ***c1*** hoặc ***c2*** là ngang ngửa nhau. (Hay dữ liệu không có tính tách biệt)

Vậy 1 thuộc tính có tính tách biệt là 1 thuộc tính gây ra được hiện tượng tách biệt dữ liệu (***Xác suất*** rơi vào 1 phân lớp nào đó cao và ***chênh lệch*** so với các phân lớp khác) ở các nhánh con của thuộc tính đó.


### 1.2 Gini là gì

#### 1.2.1 Công thức

Gini mô tả ***độ tạp chất trong tập dữ liệu*** nói cách khác:
+ Nếu tất cả các mẫu dữ liệu đều thuộc cùng 1 lớp duy nhất thì độ tạp chất trong tập dữ liệu là thấp nhất
+ Nếu các mẫu dữ liệu nó thuộc càng nhiều lớp thì độ tạp chất trong tập dữ liệu cao

+ Xét một tập ***S*** gồm các ***mẫu dữ liệu*** được phân loại vào ***c lớp***
Gọi $p_i$ là xác suất khi chọn 1 mẫu dữ liệu rơi vào lớp $i, 1 \le i \le c$. Khi đó $p_1 + p_2 + ... + p_c = 1$.
\begin{equation}
Gini(S) = \sum\limits_{i = 1}^cp_i(1 - p_i) = 1 - \sum\limits_{i = 1}^c (p_i)^2
\end{equation}

Ta có:

\begin{equation}
0 < \sum\limits_{i = 1}^c (p_i)^2 \le (\sum\limits_{i = 1}^c p_i)^2 = 1
\end{equation}

Hay nói cách khác:

+ Giá trị $0 \le Gini(S) \le 1$
+ $Gini(S) = 0$, nghĩa là độ tập chất trong tập $S$ thấp nhất. Tất cả các mẫu dữ liệu trong $S$ đều thuộc về cùng 1 lớp
+  $Gini(S)$ lớn nhất  khi mà các mẫu dữ liệu trong $S$ được phân phối ngẫu trên các lớp.


### 1.3 Thuật toán CART

Thuật toán ***Cart*** được sử dụng cho cả bài toán hồi quy và bài toán phân loại

Để cho ***quá trình phân loại tốt*** thì tại mỗi node khi xây dựng cây sao cho ***Gini*** ứng với dữ liệu tại node đó phải bé nhất. Đây chính là điểm mấu chốt của thuật toán ***CART***.

***Chú ý:***
+ ID3 sử dụng ***Infomation Gain*** để xây dựng cây
+ CART sử dụng ***Gini Index*** để xây dựng cây, củ thể sử dụng nó để tách tách ***node cha*** ra các ***node con***

Như kiến thức mình đã nêu trong ***Giải thuật ID3***, 1 ***node*** có thể có ***nhiểu node con*** ứng với ***số lượng giá trị của thuộc tính của node cha***.

Tuy nhiên thuật toán ***CART*** xây dựng ra 1 cây nhị phân ***(Cây nhị phân là cây có mỗi nút trong chỉ có 2 nút con)***

Tương tự như ***ID3*** thì node lá sẽ chứa giá trị mà chúng ta đem đi 


#### 1.3.1 Nguyên tắc xây dựng cây của thuật toán

+ Xây dựng cây dựa trên tập huấn luyện và sử dựng cây này để dự đoán cho tập dữ liệu kiểm tra
+ Với những thuộc tính có sẵn trong tập train, thuật toán tìm cách ***gán thuộc tính kiểm tra và ngưỡng giới hạn threshold*** cho từng ***node*** để từ đó tách tập dữ liệu thành 2 nhánh: ***left_branch*** và ***right_branch*** ứng với 2 ***node con nhỏ hơn***. Tiếp tục xây dựng như vậy với những ***node con***.

Vậy làm cách nào mà thuật toán ***Cart*** có thể tìm được cách gán như vây và tách các ***node cha*** thành cách ***node con***. Mình sẽ sử dụng ví dụ siêu siêu dễ hiểu để giải thích vấn đề này

#### 1.3.2 Ví dụ để mô tả bản chất của giải thuật

##### 1.3.2.1 Nêu ví dụ

Ta có 10 mẫu dữ liệu dự đoán hôm nay có ***nên (Yes)*** hoăc ***không nên (No)*** đi dạo ***hồ tây*** hay không dựa vào 2 trường thuộc tính ***Humidity*** và ***Wind***


| Humidity |  Wind | Class |
| -------- | -------- | -------- |
| 5.4   | 3.8    | No     |
| 5.0    | 3.5    | Yes    |
| 4.9    | 1.8    | No    |
| 5.3    | 3.9    | Yes     |
| 3.7    | 0.5    | No    |
| 1.8    | 0.4    | No     |
| 1.9    | 0.5    | Yes     |
| 1.8    | 0.7    | No     |
| 4.2    | 0.7    | No     |
| 1.8    | 0.5    | Yes     |

##### 1.3.2.2 Bản chất của giải thuật

Mục đích của giải thuật ***Cart*** là tại mỗi node của cây sẽ gắn liền với ***1 thuộc tính*** và ***ngưỡng giá trị(Threshold)*** để có thể tách tập dữ liệu của chúng ta thành ***2 tập dữ liệu nhỏ hơn*** mà đảm bảo việc ***phân tách này*** là ***tốt cho quá trình phân loại***.

Nghĩa là, ta sẽ tìm ***điều kiện*** để thực hiện việc phân tách giống như ***IF-ELSE*** vậy

![](https://i.imgur.com/QldVqti.png)



Nhìn hình vẽ trên bạn có thể thấy dựa vào 1 số điểu kiện như: ***Wind > 3.85*** hay ***Humidity > 2.8*** hoặc ***Condition X, Condition Y*** ta có thể tách bộ dữ liệu của chúng ta thành làm hai phần.

***Chú ý trong hình vẽ:***

+ Mỗi node sẽ có dữ liệu tương ứng
+ Ngoại trừ node lá ***(leaf node)*** thì mỗi node gắn liền 1 điều kiện để thực hiện quá trình tách về sau với những trường dữ liệu sau:
    + Attribubte: Thuộc tính điều kiện(Ví dụ: ***Wind*** trong điều kiện ***Wind > 3.85***)
    + Threshold; Ngưỡng điều kiện(Ví dụ: ***3.85*** trong điều kiện ***Wind > 3.85***)
    + Có thể có thêm giá trị ***Gini***

+ Nút lá là nút bạn không thể thực hiện được quá trình phân tách nữa hoặc số mẫu dữ liệu nhỏ hơn 1 ngưỡng cho trước. Mình sẽ nhắc ở mục phía dưới

Như vậy từ hình vẽ trên chúng ta cũng đã biết được là ***thuật toán Cart*** sẽ cố gắng ***đi tìm những điều kiện*** để xây dựng ***cây quyết định***

Vấn đề còn lại là chúng ta cần tìm ra ***những điểu kiện*** có thể thể giúp chúng ta phân loại tốt. 

Việc ***tìm ra những điều kiện này*** chúng ta phải nhớ lại khái niệm ***gini*** mình đã nhắc ở phía trên.

Phân loại tốt 1 tập dữ liệu khi mà tập dữ liệu chúng ta có giá trị ***Gini*** bé nhất có thể. Nghĩa là sẽ tồn tại 1 số lớp có nhiều mẫu dữ liệu hơn các lớp còn lại.

Vậy để có 1 cây quyết định tối ưu, thuật toán ***Cart*** sử dụng 1 cách ***tham lam*** tại các bước ***node*** như sau:

Giá trị trung bình ***gini*** sau khi phân tách thành ***2 nhánh con*** là bé nhất có thể. Như thế mới hi vọng chúng ta có thể phân loại dữ liệu tốt được.

Ta có công thức tính giá trị ***Gini*** sau khi phân tách, giá trị này còn được gọi là ***Gini weighted average***:

\begin{equation}
newgini = \frac{m_{left} * Gini_{left} + m_{right} * Gini_{right}} {m}
\end{equation}

Việc tính toán ***newgini*** phụ thuộc vào 2 nhánh con ***left*** và ***right*** được quản lý bởi 2 node con trái và phải.

***Ví dụ:*** dữ liệu (có $m = 10$ mẫu dữ liệu) trên với cách chia dựa vào điều kiện ***Wind > 3.85***, gọi $p_{yes}, p_{no}$ là xác suất khi lựa chọn 1 mẫu dữ liệu rơi vào lớp ***Yes, No*** tương ứng

+ Tập dữ liệu ở nhánh con trái có ***1 mẫu lớp Yes*** , ***0 mẫu lớp No:***
    + $m_{left} = 1$
    + $p_{yes} = 1, p_{no} = 0$
    + $Gini_{left} = 1 - (p_{yes})^2 - (p_{no})^2 = 0$

+ Tập dữ liệu ở nhánh con phải có ***3 mẫu lớp Yes*** , ***6 mẫu lớp No:***
    + $m_{right} = 9$
    + $p_{yes} = \frac{1}{3}, p_{no} = \frac{2}{3}$
    + $Gini_{right} = 1 - (p_{yes})^2 - (p_{no})^2 = \frac{4}{9}$

+ $newgini = \frac{m_{left} * Gini_{left} + m_{right} * Gini_{right}} {m} =  \frac{1 * 0 +  9 * \frac{4}{9}} {10}=0.4$

Vậy ***tách nào cực tiểu hóa $newgini$ tại mỗi bước*** ?

***Cách đơn giản nhất là xét tất cả các trường thuộc tính và threshold một cách phù hợp dựa vào cột thuộc tính tương ứng***

+ Tập dữ liệu có 2 thuộc tính: ***Wind*** và ***Humidity***
+ Ứng với từng thuộc tính $attr$, để tìm ra ngưỡng $threshold$  để tạo ta 1 điều kiện $attr > threshold$ phân tách tốt:
    + ***(1)*** Sắp xếp từ ***bé đến lớn theo giá trị*** trong cột thuộc tính $attr$, mình mặc định là tập dữ liệu đã được chuyển sang dạng số, ta được tập $D = [val_1, val_2, .., val_m], val_1 \le val_2 \le ... \le val_m$. Lý do ta sắp xếp để tiện cho việc chọn 2 tập $left$ và $right$
    
    +  ***(2)*** Do ngưỡng $threshold$ dùng để tách thành 2 tập $left$ và $right$ sao cho giá trị($val$) tại $attr$ của tất cả mẫu dữ liệu của tập $left$ nhỏ hơn $threshold$ và giá trị($val$) tại $attr$ của tất cả mẫu dữ liệu của tập $right$ lớn hơn hoặc bằng $threshold$ 
    
    +  ***(3)*** Từ điều kiện trên ta chỉ cân duyệt các trường hợp tách  
    ***{***$(left, right) \mid left = [x_1, x_2, ..., x_k], right = [x_{k + 1}, x_{k + 2}, ...,x_{m}], \forall 1 \le k \le m-1$***}***, với $X = [x_1, x_2, ..., x_m]$ là bộ dữ liệu của tập cha sau khi sắp xếp theo thứ tự tăng dần với thuộc tình $attr$ đang xét
    
    +  ***(4)*** Trong quá trình duyệt trên, cập nhật các giá trị $min_{newgini}$, ***thuộc tính tốt nhất***. Giả sử tại vị trí $k$ ta đạt được cách ***tách tối ưu*** (tối thiểu $newgini$), $1 \le k \le m-1$. Tập ***left*** = ***{***$[x_1, x_2, ..., x_k]$***}*** và tập ***right*** = ***{***$[x_{k + 1}, x_{k + 2}, ...,x_{m}]$***}***
    Ta có: từ ***(2)***, $x_k < threshold < x_{k + 1}$.
    ***Chọn giá trị $threshold = \frac{x_k + x_{k + 1}}{2}$*** 


##### 1.3.2.3 Áp dụng giải thuật vào ví dụ

| Humidity |  Wind | Class |
| -------- | -------- | -------- |
| 5.4   | 3.8    | No     |
| 5.0    | 3.5    | Yes    |
| 4.9    | 1.8    | No    |
| 5.3    | 3.9    | Yes     |
| 3.7    | 0.5    | No    |
| 1.8    | 0.4    | No     |
| 1.9    | 0.5    | Yes     |
| 1.8    | 0.7    | No     |
| 4.2    | 0.7    | No     |
| 1.8    | 0.5    | Yes     |

Xét thuộc tính ***Wind***

+ ***Sắp xếp theo thứ tự tăng dần bộ dữ liệu theo cột Wind***

    | Humidity |  Wind | Class |
    | -------- | -------- | -------- |
    | 1.8    | 0.4    | No     |
    | 3.7    | 0.5    | No    |
    | 1.8    | 0.5    | Yes     |    
    | 1.9    | 0.5    | Yes     |
    | 1.8    | 0.7    | No     |
    | 4.2    | 0.7    | No     |
    | 4.9    | 1.8    | No    |
     | 5.0    | 3.5    | Yes    |
    | 5.4   | 3.8    | No     |
    | 5.3    | 3.9    | Yes     |
   
    
+ Duyệt các cách tách thành 2 bộ ***left*** và ***right***
Gán $min_{newgini} = 1$, gọi $x_i$ là giá trị tại mẫu dữ liệu i của thuộc tính ***Wind*** của bộ dữ liệu (i bắt đầu từ 1)

     + ***Iteration 1: k = 1, $x_k = 0.4 \neq x_{k + 1} = 0.5$*** 

        | Thông số | Tập left | tập right |
        | -------- | -------- | -------- |
        | Danh sách hàng | [1]     | [2, 3, 4, 5, 6, 7, 8, 9, 10]|
        | Số lượng mẫu dữ liệu | 1    | 9 |
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***Yes*** |$p_{yes} = 0$|   $p_{yes} = 4/9$|
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***No*** |$p_{no} = 1$|   $p_{no} = 5/9$|
        | $Gini = 1 - (p_{yes})^2 - (p_{no})^2$ |$G_{left} = 0$|   $G_{right} = \frac{40}{81}$|

        
        + $newgini = \frac{m_{left} * Gini_{left} + m_{right} * Gini_{right}} {m} =  \frac{1 * 0 +  9 * \frac{40}{81}} {10}= \frac{4}{9}$
        + $min_{newgini} = min(min_{newgini}, newgini) = \frac{4}{9}$
        + best_attribute_idx (chỉ số của thuộc tính) = 0 (ứng với thuộc tính ***Wind***)
        + best_splitting_threshold = $\frac{x_1 + x_2}{2} = \frac{0.4 + 0.5}{2} = 0.45$
        
     + ***Iteration 2: k = 2, $x_k = 0.5 = x_{k + 1}$*** 

        | Thông số | Tập left | tập right |
        | -------- | -------- | -------- |
        | Danh sách hàng | [1, 2]     | [3, 4, 5, 6, 7, 8, 9, 10]|
        | Số lượng mẫu dữ liệu | 2    | 8 |
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***Yes*** |$p_{yes} = 0$|   $p_{yes} = 1/2$|
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***No*** |$p_{no} = 1$|   $p_{no} = 1/2$|
        | $Gini = 1 - (p_{yes})^2 - (p_{no})^2$ |$G_{left} = 0$|   $G_{right} = \frac{1}{2}$|
        
        Ta chỉ cập nhật các thông số $min_{newgini}$, best_attribute_idx, best_splitting_threshold khi mà $x_k \neq x_{k + 1}$
    
    + ***Iteration 3: k = 3, $x_k = 0.5 = x_{k + 1}$*** 
        
        Ta chỉ cập nhật các thông số $min_{newgini}$, best_attribute_idx, best_splitting_threshold khi mà $x_k \neq x_{k + 1}$
        
     + ***Iteration 4: k = 4, $x_k = 0.5 \neq x_{k + 1} = 0.7$*** 

        | Thông số | Tập left | tập right |
        | -------- | -------- | -------- |
        | Danh sách hàng | [1, 2, 3, 4]     | [5, 6, 7, 8, 9, 10]|
        | Số lượng mẫu dữ liệu | 4    | 6 |
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***Yes*** |$p_{yes} = 1/2$|   $p_{yes} = 1/3$|
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***No*** |$p_{no} = 1/2$|   $p_{no} = 2/3$|
        | $Gini = 1 - (p_{yes})^2 - (p_{no})^2$ |$G_{left} = 1/2$|   $G_{right} = \frac{4}{9}$|

        
        + $newgini = \frac{m_{left} * Gini_{left} + m_{right} * Gini_{right}} {m} =  \frac{4 * \frac{1}{2} +  6 * \frac{4}{9}} {10}= \frac{7}{15}$
        + $min_{newgini} = min(min_{newgini}, newgini) = min(\frac{4}{9}, \frac{7}{15}) = \frac{4}{9}$
        + Giá trị $min_{newgini}$ không đổi nên ta không cập nhật các thông số $min_{newgini}$, best_attribute_idx, best_splitting_threshold 
    
    + ***Iteration 5: k = 5, $x_k = 0.7 = x_{k + 1}$*** 
        
        Ta chỉ cập nhật các thông số $min_{newgini}$, best_attribute_idx, best_splitting_threshold khi mà $x_k \neq x_{k + 1}$
    
    + ***Iteration 6: k = 6, $x_k = 0.7 \neq x_{k + 1} = 1.8$*** 

        | Thông số | Tập left | tập right |
        | -------- | -------- | -------- |
        | Danh sách hàng | [1, 2, 3, 4, 5, 6]     | [7, 8, 9, 10]|
        | Số lượng mẫu dữ liệu | 6    | 4 |
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***Yes*** |$p_{yes} = 1/3$|   $p_{yes} = 1/2$|
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***No*** |$p_{no} = 2/3$|   $p_{no} = 1/2$|
        | $Gini = 1 - (p_{yes})^2 - (p_{no})^2$ |$G_{left} = 4/9$|   $G_{right} = 1/2$|

        
        + $newgini = \frac{m_{left} * Gini_{left} + m_{right} * Gini_{right}} {m} =  \frac{6 * \frac{4}{9} +  4 * \frac{1}{2}} {10}= \frac{7}{15}$
        + $min_{newgini} = min(min_{newgini}, newgini) = min(\frac{4}{9}, \frac{7}{15}) = \frac{4}{9}$
        + Giá trị $min_{newgini}$ không đổi nên ta không cập nhật các thông số $min_{newgini}$, best_attribute_idx, best_splitting_threshold
    
     + ***Iteration 7: k = 7, $x_k = 1.8 \neq x_{k + 1} = 3.5$*** 

        | Thông số | Tập left | tập right |
        | -------- | -------- | -------- |
        | Danh sách hàng | [1, 2, 3, 4, 5, 6, 7]     | [8, 9, 10]|
        | Số lượng mẫu dữ liệu | 7    | 3 |
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***Yes*** |$p_{yes} = 2/7$|   $p_{yes} = 2/3$|
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***No*** |$p_{no} = 5/7$|   $p_{no} = 1/3$|
        | $Gini = 1 - (p_{yes})^2 - (p_{no})^2$ |$G_{left} = 20/49$|   $G_{right} = 4/9$|

        
        + $newgini = \frac{m_{left} * Gini_{left} + m_{right} * Gini_{right}} {m} =  \frac{7 * \frac{20}{49} +  3 * \frac{4}{9}} {10}= \frac{44}{105}$
        + $min_{newgini} = min(min_{newgini}, newgini) = min(\frac{4}{9}, \frac{44}{105}) = \frac{44}{105}$
        + best_attribute_idx (chỉ số của thuộc tính) = 0 (ứng với thuộc tính ***Wind***)
        + best_splitting_threshold = $\frac{x_7 + x_8}{2} = \frac{1.8 + 3.5}{2} = 2.65$
    
    + ***Iteration 8: k = 8, $x_k = 3.5 \neq x_{k + 1} = 3.8$*** 

        | Thông số | Tập left | tập right |
        | -------- | -------- | -------- |
        | Danh sách hàng | [1, 2, 3, 4, 5, 6, 7, 8]     | [9, 10]|
        | Số lượng mẫu dữ liệu | 8    | 2 |
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***Yes*** |$p_{yes} = 3/8$|   $p_{yes} = 1/2$|
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***No*** |$p_{no} = 5/8$|   $p_{no} = 1/2$|
        | $Gini = 1 - (p_{yes})^2 - (p_{no})^2$ |$G_{left} = 15/32$|   $G_{right} = 1/2$|

        
        + $newgini = \frac{m_{left} * Gini_{left} + m_{right} * Gini_{right}} {m} =  \frac{8 * \frac{15}{32} +  2 * \frac{1}{2}} {10}= \frac{19}{40}$
        + $min_{newgini} = min(min_{newgini}, newgini) = min(\frac{44}{105}, \frac{19}{30}) = \frac{44}{105}$
        + Giá trị $min_{newgini}$ không đổi nên ta không cập nhật các thông số $min_{newgini}$, best_attribute_idx, best_splitting_threshold
    
    + ***Iteration 9: k = 9, $x_k = 3.8 \neq x_{k + 1} = 3.9$*** 

        | Thông số | Tập left | tập right |
        | -------- | -------- | -------- |
        | Danh sách hàng | [1, 2, 3, 4, 5, 6, 7, 8, 9]     | [10]|
        | Số lượng mẫu dữ liệu | 9    | 1 |
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***Yes*** |$p_{yes} = 1/3$|   $p_{yes} = 1$|
        | Xác suất 1 mẫu dữ liệu thuộc lớp ***No*** |$p_{no} = 2/3$|   $p_{no} = 0$|
        | $Gini = 1 - (p_{yes})^2 - (p_{no})^2$ |$G_{left} = 4/9$|   $G_{right} = 0$|

        
        + $newgini = \frac{m_{left} * Gini_{left} + m_{right} * Gini_{right}} {m} =  \frac{9 * \frac{4}{9} +  1 * 0} {10}= \frac{4}{10}$
        + $min_{newgini} = min(min_{newgini}, newgini) = min(\frac{44}{105}, \frac{4}{10}) = \frac{4}{10}$
        + best_attribute_idx (chỉ số của thuộc tính) = 0 (ứng với thuộc tính ***Wind***)
        + best_splitting_threshold = $\frac{x_9 + x_10}{2} = \frac{3.8 + 3.9}{2} = 3.85$
    
Xét thuộc tính ***Humidity***

+ ***Sắp xếp theo thứ tự tăng dần bộ dữ liệu theo cột Humidity***

    | Humidity |  Wind | Class |
    | -------- | -------- | -------- |
    | 1.8    | 0.4    | No     |
    | 1.8    | 0.5    | Yes     | 
    | 1.8    | 0.7    | No     |
    | 1.9    | 0.5    | Yes     |
    | 3.7    | 0.5    | No    |
    | 4.2    | 0.7    | No     |    
    | 4.9    | 1.8    | No    |
    | 5.0    | 3.5    | Yes    |
    | 5.3    | 3.9    | Yes     |
    | 5.4   | 3.8    | No     |
    

+ Việc duyệt tương tự như trên, bạn tính cho dễ nhớ kiến thức nhé!!!

Sau khi ***duyệt toàn bộ thuộc tính***, ta có thông tin cho ***node gốc của cây quyết định*** như sau:

+ Điều kiện để tách: ***Wind > 3.85***
+ min_gini = 0.4

Sau khi tách ta có:

![](https://i.imgur.com/ppiK9Kf.png)


***Chú ý:***
+ tách node đầu tiên ta được node con trái là node lá
+ Tiếp tục tách node con bên phải theo thuật toán ***Cart*** như trên do node con bên phải không phải lá



#### 1.3.3 Điều kiện dừng của giải thuật

Nếu chúng ta quá ***tập trung vào việc phát triển cây***, và thực tế tập dữ liệu có ***số chiều khá lớn***, điều này đồng nghĩa với việc cây quyết định chúng ta tạo ra cũng ***khá lớn***. 

Nếu đánh giá 1 cách khách quan hơn, so sánh với mô hình ***Linear Regression***, do chúng ta đã quá tập trung ***tối ưu hóa hàm lỗi thực nghiệm*** dẫn đến mô hình bị ***overfit*** với dữ liệu. Điều này cũng xảy ra tương tự với cây quyết định khi chúng ta xây dụng 1 cây quyết định đầy đủ đến mức mà có thể ***fit 100%*** với tập huấn luyện , điều kiện ***trong tập dữ liệu không xác định các mẫu dữ liệu không nhất quán ví dụ: cùng giá trị các thuộc tính, nhưng mà giá trị cột quyết định lại khác nhau***

Vì vấn đề quá rõ ràng trên thì trong học máy cũng sẽ tìm cách để ***regularization*** cho cây quyết định để tránh hiện tượng ***overfitting***.

***Cắt tỉa (Prunning)*** là một tỏng những kỹ thuật như vậy

##### 1.3.2.1 Phương pháp cắt tỉa - Prunning

![](https://i.imgur.com/UFx0vKy.png)

Phương pháp ***cắt tỉa cây*** là 1 kỹ thuật thuật giảm kích thước cây quyết định thay vì cố gắng phát triển hết toàn bộ cây. Điều này cũng phần nào giảm được sự phức tạp và thời gian ***phân loại*** ở bước ***test***


Vậy ***regularization*** thế nào là hợp lý, hay nói 1 cách dễ hiểu hơn là kích thước tối ưu cho cây quyết định là gì?

Một cây ***quá lớn*** cũng dẫn tới:
+ Hiện tượng overfitting
+ Khó tổng quát hóa được trong tương lai cho các ***mẫu dữ liệu quan sát mới***

Một cây ***quá bé*** cũng dẫn tới ***hiện tượng undefitting***, cây không học được các thuộc tính để phân loại trong tập huấn luyện

***Jì Zậy Trời !!! Tôi phải làm như thế nào cho vừa lòng anh đây ?***

Cách có thể nghĩ tới ngay chúng ta cứ phát triển cây và cho đến khi nào tại mỗi ***node*** ta gặp trường hợp tại mỗi ***node*** ấy có ***số lượng mẫu dữ liệu ít hơn 1 ngưỡng min_samples_split nào đó*** (1 thuộc tính trong lớp DecisionTreeClassifier), ta sẽ ***cắt tỉa-prunning*** để bỏ đi node đó. Do chúng ta đã lựa chọn các ***thuộc tính cho node*** mang tính ***phân loại dữ liệu cao*** tại những bước đầu tiên với ***số lượng mẫu dữ liệu tại node đó cũng đủ lớn*** nên các ***những ít mẫu dữ liệu và mang những thuộc tính khác*** không mang thêm nhiều thông tin trong việc phân loại của chúng ta.

Trong các viết tiếp tới, mình sẽ giới thiệu tới các bạn vấn đề này.

##### 1.3.2.2 Một số kỹ thuật khác

Bạn cũng có thể không phát triển cây tại 1 ***node*** nếu thỏa mãn điều kiện sau đây:

+ Nếu node có quá ít mẫu dữ liệu ta sẽ không phân tách thành các nhánh nữa và sẽ xem đó là ***node lá - leaf node*** với nhãn là lớp chiếm phần đa trong mẫu dữ liệu còn lại đó.
+ Nếu số lượng nút lá đã vượt quá giới hạn cho phép thì ta cũng dừng việc phát triển cây.


## 2. Thực hành

### 2.1 Tự cài đặt giải thuật CART bằng python

Dưới đây là lập trình của mình cho giải thuật Cart, làm việc với cả dữ liệu categorical và dữ liệu liên tục. Bạn có thể xem đầy đủ source code [tại đây]()

<b>Tạo <span style="color:brown">class Node</span></b> 

```ruby=
class Node:
    def __init__(self, label):
        self.label = label # nhãn của node nếu node đó là node lá
        self.best_attribute_idx = 0 # thuộc tính để phân tách dữ liệu thành 2 node con
        self.best_splitting_threshold = 0 # ngưỡng phân tách dữ liệu thành 2 node con
        self.left_branch = None # node con trái
        self.right_branch = None # node con phải
        self.gini = 0 # giá trị gini

    def set_left_branch(self, left_branch):
        self.left_branch = left_branch
    
    def set_right_branch(self, right_branch):
        self.right_branch = right_branch
```

<b>Hàm tính<span style="color:brown"> Gini</span></b> 

```python=
def calc_gini(freq_list):
    arr = np.array(freq_list)

    no_samples = np.sum(arr)
    
    tmp = 0
    for freq in arr:
        tmp += (freq/no_samples) ** 2
    return 1 - tmp
```

Tất cả code bạn xem [tại đây]()

Ta sẽ sừ dụng bộ dữ liệu ***iris***. Việc huấn luyện ***decision tree*** dựa trên giải thuật ***Cart***

```python=
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    print("Kích thước bộ dữ liêu:", X.shape)
    skf = StratifiedKFold(n_splits=5)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        clf = CartDecisionTree(max_depth = 20)
        train_data = X[train_idx]
        train_target = y[train_idx]
        test_data = X[test_idx]
        test_target = y[test_idx]

        clf.fit(train_data, train_target)
        y_pred = clf.predict(test_data)
        score = accuracy_score(y_pred, test_target)
        scores.append(score)
    scores = np.array(scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
```

```python=
Kích thước bộ dữ liêu: (150, 4)
0.92 accuracy with a standard deviation of 0.03
```

***Chú ý: Do mẫu dữ liệu khá nhỏ nên mình dùng phương pháp lấy mẫu phân tầng để đánh giá mô hình***. Mình sẽ giới thiệu vấn đề này trong các bài sau.

### 2.2 Sử dụng thư viện sklearn

Ở đây mình sử dụng lớp ***DecisionTreeClassifier*** của gói ***tree*** trong thư viên scikit-learn

```python=
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    print("Kích thước bộ dữ liêu:", X.shape)
    skf = StratifiedKFold(n_splits=5)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        # clf = CartDecisionTree(max_depth = 20)
        clf = tree.DecisionTreeClassifier(max_depth=20)
        train_data = X[train_idx]
        train_target = y[train_idx]
        test_data = X[test_idx]
        test_target = y[test_idx]

        clf.fit(train_data, train_target)
        y_pred = clf.predict(test_data)
        score = accuracy_score(y_pred, test_target)
        scores.append(score)
    scores = np.array(scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
```

```python=
Kích thước bộ dữ liêu: (150, 4)
0.96 accuracy with a standard deviation of 0.03
```

***Nhận xét***: Như chúng ta thấy khi sử dụng thư viện sklearn thì độ chính xác khi dự đoán có cao hơn một chút.
Sở dĩ khi mình cài đặt thuật toán ***Cart*** bằng python thì mới chỉ cài đặt ***ý tưởng cở bản nhất*** của giải thuật. Chứ chưa có các kỹ thuật tối ưu như trong thư viện sklearn

## 3. Thuật toán Cart và ID3

Đối với cả 2 giải thuật, trong quá trình xây dựng cây tại mỗi bước thì chỉ có thể tách dựa vào 1 thuộc tính duy nhất

***Khác nhau:***

+ Thuật toán ***Cart*** xây dựng 1 cây quyết định nhị phân, mỗi node trong có 2 node con còn thuật toán ***ID3*** xây dựng 1 cây quyết định mà mỗi node trong có thể có nhiều hơn 2 node con
    + Thuật toán ***Cart*** sử dụng ***Gini*** trong quá trình xây dựng cây còn thuật toán ***ID3*** sử dụng ***Information Gain***
+ Khi nhận 1 giá trị liên tục:
    + Thuật toán ID3 phải chia khoẳng giá trị của thuộc tính thành khá nhiều thành phần, mỗi phần có 1 khoảng dữ liệu ***(Rời rạc hóa)***
    + Trong khi đó thuật toán Cart không cần làm điểu này
+ Cart làm việc dễ dàng hơn so với các kiểu dữ liệu ***numerical*** và ***categorical*** hơn ID3

## 4. Tài liệu tham khảo

+ [COMPARATIVE STUDY ID3, CART AND C4.5 DECISION TREE ALGORITHM: A SURVEY - Sonia Singh](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.685.4929&rep=rep1&type=pdf)
+ [Entropy, Information gain, and Gini Index; the crux of a Decision Tree](https://blog.clairvoyantsoft.com/entropy-information-gain-and-gini-index-the-crux-of-a-decision-tree-99d0cdc699f4)
+ [Gini Index: Decision Tree, Formula, and Coefficient](https://blog.quantinsti.com/gini-index/)
+ [Understanding the Gini Index and Information Gain in Decision Trees](https://medium.com/analytics-steps/understanding-the-gini-index-and-information-gain-in-decision-trees-ab4720518ba8)
+ [Decision tree pruning - Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_pruning)