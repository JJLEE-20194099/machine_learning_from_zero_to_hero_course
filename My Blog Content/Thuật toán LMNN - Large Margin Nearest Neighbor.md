# Thuật toán LMNN - Large Margin Nearest Neighbor

Trong bài viết này mình sẽ giới thiệu cho các bạn một giải thuật nhằm nâng cao độ chính xác cho thuật toán KNN ***(K-nearest Neighbors)*** mình đã giới thiệu ở bài viết trước. Thuật toán này thay vì chúng ta sử dụng các độ đo ***(metric)*** khoảng cách như ***manhattan_distance*** hay ***euclid_distance*** thì thuật toán ***LMNN*** học 1 độ đo riêng cho bài toán ***Classification***

## 1. Khái niệm metric và pseudometric

***Metric*** hay ***Độ đo*** trên 1 tập ${X}$ là 1 ***ánh xạ***:

$d: X \times X$ &rarr; $R$

Thỏa mãn:
1. $d(x_i, x_j) + d(x_j, x_k) \geq d(x_i, x_k)$

2. $d(x_i, x_j) = d(x_j, x_i)$
3. $d(x_i, x_j) \geq 0$
4. $d(x_i, x_j) = 0$ &harr; $x_i = x_j$

***Pseudometric*** là 1 độ đo chỉ cần thỏa mãn điểu kiện 1, 2, 3 ở trên. Và có thể 2 điểm $x_i, x_j$ khác nhau nhưng mà độ đo giữa chúng có thể bằng 0.

## 2. Bản chất của thuật toán

Thuật toán ***LMNN*** sẽ học một độ đo ***pseudometric*** có dạng $d(x_i, x_j) = (x_i - x_j)^TM(x_i - x_j)$

***Trong đó:*** M là ma trận nửa xác định dương

Ma trận M kích thước $n \times n$ được gọi là ma trận nửa xác định dương ***(positive semi-definite)*** khi và chỉ khi $x^TMx \geq 0,  \forall x \in R^n$

Trong giải thuật ***LMNN*** có 2 loại hàng xóm là ***Target Neighbor*** và ***Impostor***

Xét tập training data $D = [(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)]$

### 2.1 Target Neighbor

Thông qua độ đo chúng ta học ***(Ánh xạ d)***, những điểm target neighbor của 1 điểm $x_i$ là những điểm gần nhất với điểm $x_i$ đó và những điểm này có cùng lớp với $x_i$. $x_j$ là target neighbor của $x_i$ thì chưa chắc $x_i$ là target neighbor của $x_j$. Tập những target neighbors của $x_i$ là $N_i$

### 2.2 Impostor

Một điểm $x_j$ được gọi là ***Impostor*** của $x_i$ nếu điểm $x_j$ là 1 trong những hàng xóm gần nhất của $x_i$ mà khác lớp với $x_i$

![](https://i.imgur.com/yeY4We4.png)

Hình ảnh lấy từ [wiki](https://en.wikipedia.org/wiki/Large_margin_nearest_neighbor#:~:text=Large%20margin%20nearest%20neighbor%20) mô tả điểm target neighbor và điểm impostor


### 2.3 Cơ chế hoạt động

Để cho thuật toán ***KNN*** hiểu quả, đối với từng điểm $x_i$ thì các hàng xóm ***target neighbors*** dưới ánh xạ ***d*** nên gần hơn so với các điểm có nhãn khác với $x_i$. Vì vậy có thể nói rằng: Chúng ta cần học ***1 ánh xạ*** hay ***1 hàm độ đo*** sao cho tất cả các điểm ***target neighbors*** của $x_i$ sẽ tạo ra 1 vùng không gian bao quanh điểm $x_i$ và không chứa bất cứ điểm nào ***khác phân lớp*** với $x_i$. 

Nếu chúng ta học được 1 ánh xạ ***d*** như vậy khiến cho ***tất cả mẫu dữ liệu trong tập training*** được bao quanh bới ít nhất ***k*** điểm có cùng lớp với mẫu tương ứng thì khi đó lỗi dự đoán của chúng ta sẽ tối ưu

Vậy khi chúng ta mô hình hóa mong muốn trên về bài toán tối ưu nghĩa là phải cực tiểu hóa số lượng các điểm không cùng lớp với $x_i$ ***(Impostors)***.

Thuật toán ***LMNN*** có thực hiện việc cực tiểu hóa số lượng các điểm ***Impostors*** bằng cách luôn duy trì 1 giá trị lớn được gọi là lề ***(margin)***  giữa vùng không gian của ***target neighbors*** và ***impostors***.

Đó là lý do mà thuật toán có tên là ***Large Margin Nearest Neighbor***

Ta định nghĩa lại $x_l$ là 1 hàng xóm ***Impostor*** của $x_i$ thì:

\begin{equation}
d(x_i, x_l) \leq d(x_i, x_j) + 1, \forall x_j \in \text{ tập các target neighbors}
\end{equation}

Nói cách khác, ***Impostor*** $x_j$ cảu $x_i$ là những điểm khác phân lớp với $x_i$ và thuộc vào vùng không gian (tổng không gian chứa các ***target neighbor*** và 1 giá trị ***margin*** nhất định, lựa chọn giá trị lề là 1)

![](https://i.imgur.com/95PIUka.png)

Nhìn vào hình vẽ trên ta thấy, ứng với mỗi điểm $x_i$ trước khi học, ***k hàng xóm*** của nó sẽ bao gồm cả ***Impostors*** và ***Target neighbors***. Tuy nhiên sau khi học, mục đích của chúng ta sẽ ***pull*** những hàng xóm ***target neighbors*** lại gần $x_i$ và đẩy những hàng xóm ***impostors*** ra xa $x_i$. Hơn nữa sau quá trình học sẽ hình thành 1 vùng ***margin*** giữa ***Impostors*** và ***Target Neighbors***.

### 2.4 Xây dựng hàm lỗi trong giải thuật LMNN

Như đã nói ở trên giải thuật LMNN tối ưu thuật toán ***KNN*** bằng cách xây dựng một ánh xạ ***d*** sao cho với một giá trị k cho trước, bất cứ 1 điểm $x_i$ nào cũng nên chỉ được bao quanh bởi ***k target neighbors***.

Việc tối ưu hàm lỗi sẽ được chia là 2 giai đoạn: 

+ ***Kéo (pull)*** những ***target neighbors*** lại gần $x_i$

+ ***Đẩy (push)*** những ***impostors*** ra xa $x_i$


Việc ***Kéo*** những điểm ***target neighbors*** lại càng gần $x_i, đồng nghĩa với việc tối ưu hàm sau:

\begin{equation}
f_1(d) = \sum\limits_{i, j \in N_i}d(x_i, x_j)
\end{equation}

***d là pseudometric*** là giải thuật ***LMNN*** cần phải học

Trước khi học các ***Impostor*** nằm trong vùng không gian ***Target Neighbor*** cộng với phần ***margin***

\begin{equation}
d(x_i, x_l) \leq d(x_i, x_j) + 1, \forall x_j \in \text{ tập các target neighbors}
\end{equation}

Việc ***đẩy*** những điểm ***Impostor*** ra xa $x_i$ đồng nghĩa:

\begin{equation}
d(x_i, x_j) + 1 - d(x_i, x_l) < 0, \forall x_j \in \text{ tập các target neighbors}
\end{equation}

Quá trình ***đẩy*** những điểm ***Impostor*** đồng nghĩa với việc tối ưu hàm lõi sau:


\begin{equation}
f_2(d) = \sum\limits_{i, j \in N_i, l, y_l \neq y_i} [d(x_i, x_j) + 1 - d(x_i, x_l)]_+
\end{equation}



***Chú ý:*** 
+ Hàm số $[x]_+ = max(x, 0)$

+ Việc chọn giá trị ***margin*** có thể thay đổi bằng 1 số dương khác bằng việc điểu chỉnh lại tỷ lệ các ***weight*** của ma trận ***M*** mình cần học trong ánh xạ ***d***

Từ 2 quá trình trên, ta cần tối ưu hàm lỗi sau:

\begin{equation}
L(d) = \mu f_1(d) + (1 - \mu) f_2(d), \mu \in [0, 1]
\end{equation}

***$ \mu $ cũng là siêu tham số chúng ta cần chọn trước, có vai trò đánh trọng số mức độ thành phần quan trọng trong hàm lỗi L(d)***