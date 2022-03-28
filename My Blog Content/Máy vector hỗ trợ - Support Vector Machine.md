# Máy vector hỗ trợ - Support Vector Machine

Chúng ta đã cùng nhau trải qua 1 số mô hình thú vị để giải quyết những bài toán tỏng ***học máy***. Trong bài viết ngày hôm nay mình sẽ đề cập tới một mô hình khá là quan trọng và phổ biến để giải quyết ***cả bài toán phân loại và bài toán hồi quy***, ***đó chính là mô hình máy vector hỗ trợ - Support Vector Machine (viết tắt là SVM)***

## 1. Bài toán SVM tuyến tính

### 1.1 Giới thiệu tổng quan bài toán

Xét bài toán với tập dữ liệu training $D$ có $r$ mẫu dữ liệu. Bài toán cần phân dữ liệu ra ***2 lớp*** được đại diện bởi ***2 con số 1 và -1***. Ta gọi ***1 có nghĩa là possitive*** và ***-1 có nghĩa là negative***.

Vì sao lại chọn ***-1 và 1***, mình sẽ giải thích sau trong bài viết này.

Đối với ***mô hình hồi quy tuyến tính*** giả thuyết đặt ra là dữ liệu tuân theo ***mối quan hệ tuyến tính*** nào đó, thì trong ***SVM tuyến tính*** ta cũng có giả thuyết rằng ***luôn tồn tại một siêu phẳng(hyperplane)*** có thể chia tách tốt bộ dữ liệu thành ***2 cụm lớp -1 và 1***

![](https://i.imgur.com/W86VUvk.png)

***Hình ảnh được tham khảo từ [Support Vector Machine](https://tjmachinelearning.com/lectures/1718/svm/)***

***Một siêu phẳng*** tách bộ dữ liệu của chúng ta thành hai phần, đó là lớp ***dương và âm*** có dạng:

$$\langle w, x \rangle + b = 0$$

Trong nhiều trường hợp thì họ cũng gọi $\langle w, x \rangle + b$ là ***decision boundary***. Tuy nhiên bạn có thể nhìn thấy rằng, có thể có ***vố số siêu phẳng có thể thỏa mãn*** những gì chúng ta mong muốn. Vậy ***siêu phẳng nào là tốt nhất trong các siêu phẳng*** hay nói cách khác ***siêu phẳng nào có tính tổng quát hóa cao cho những điểm dữ liệu trong tương lai***

Củ thể hơn:

Ta sẽ đi tìm một siêu phẳng $f(x) = \langle w, x \rangle + b$ thỏa mãn:

Xét bộ dữ liệu training $D = [(x_1, y_1), (x_2, y_2), ..., (x_r, y_r)]$

$\forall x_i \in D$ thì:

+ Nếu điểm $x_i$ thuộc vào lớp ***positive*** hay $y_i = 1$ thì $f(x_i) = \langle w, x_i \rangle + b \ge 0$
+ Nếu điểm $x_i$ thuộc vào lớp ***negative*** hay $y_i = -1$ thì $f(x_i) = \langle w, x_i \rangle + b < 0$

Những điểm thuốc lớp positive thì sẽ nằm trên hàm $f(x)$, những điểm thuộc lớp negative tì sẽ nằm dưới hàm $f(x)$

***Hàm nào sẽ tốt nhất ?***. Tiếp đây mình sẽ giới thiệu đến ***nguyên lý lề cực đại - max margin***.

### 1.2 Nguyên lý lề cực đại - max margin

#### 1.2.1 Định nghĩa

***Lề - margin*** là cái gì vậy? Khái niệm ***margin*** chắc hầu hết ai cũng biết là ***một thuộc tính quan trọng trong CSS*** mang ý nghĩa là ***đệm thêm một lượng pixel*** cho một ***thẻ-tag*** nào đó.

Đối với ***nguyên lý lề cực đại*** thì khái niệm ***margin*** có khác đi một chút xíu. Trước khi đi vào chi tiết, mình có một vài khái niệm cơ bản sau đây:

![](https://i.imgur.com/2Bxm5Wq.png)

+ $H_0$ là siêu phẳng $\langle w,x \rangle + b = 0$ 
+ $H_+$ là siêu phẳng dương ***(positive hyperplane)***: $H_+$ được xác định bằng cách di chuyển song song cho đến khi ***chạm vào lớp dương***
+ $H_-$ là siêu phẳng âm ***(negative hyperplane)***: $H_-$ được xác định bằng cách di chuyển song song cho đến khi ***chạm vào lớp âm***

***Khi đó margin trong trường hợp này chính là khoảng các giữa hai siêu phẳng $H_+$ và $H_-$***

***Ký hiệu:***

+ Điểm tiếp xúc của ***positive hyperplane*** với ***lớp dương*** là $(x_+, 1)$
+ Điểm tiếp xúc của ***negative hyperplane*** với ***lớp âm*** là $(x_-, -1)$
***Nói cách khác:***

+ $\langle w, x_+ \rangle + b = 1$
+ $\langle w, x_- \rangle + b = -1$

Do $(x_+, 1)$ và $(x_-, 1)$ là các điểm tiếp xúc lần lượt với ***siêu phẳng dương*** và ***siêu phẳng âm*** nên ***các điểm thuộc lớp dương*** sẽ nẳm trên ***siêu phẳng dương*** và ***các điểm thuộc lớp âm*** sẽ nẳm dưới ***siêu phẳng âm*** 

+ $\langle w, x_i \rangle + b >= 1$ nếu $y_i = 1$
+ $\langle w, x_i \rangle + b <= -1$ nếu $y_i = -1$

***Nguyên lý lề cực đại sẽ đi tìm một siêu phẳng sao cho lề (khoảng cách giữa 2 siêu phẳng $H_-$ và $H_+$) lớn nhất***

Khoảng cách giữa 2 siêu phẳng $H_+$ và $H_-$ là:

$$d = d_1 + d_2$$

Với:
+ $d_1$ là khoảng cách từ siêu phẳng dương tới $H_0$
$$d_1 = d(H_+, H_0) = d((x_+, 1), H_0) = \frac{|\langle w, x_+ \rangle + b - 0|}{\parallel w \parallel} = \frac{1}{\parallel w \parallel}$$
+ $d_2$ là khoảng cách từ siêu phẳng âm tới $H_0$
$$d_1 = d(H_-, H_0) = d((x_-, 1), H_0) = \frac{|\langle w, x_- \rangle + b - 0|}{\parallel w \parallel} = \frac{1}{\parallel w \parallel}$$


***Khi đó: $d = \frac{2}{\parallel w \parallel}$***

Theo nguyên lý lề cực đại ta cần tìm $d$ lớn nhất hay $\parallel w \parallel$ nhỏ nhất

***Tổng kết lại ta có bài toán tối ưu sau đây:***

Hãy tìm giá trị $w, b$ sao cho biểu thức dưới đây đạt giá trị lớn nhất

$$d = margin =  \frac{2}{\parallel w \parallel}$$

Trong đó: $w, b$ thỏa mãn hai điều kiện sau:
+ $\langle w, x_i \rangle + b \ge 1$ nếu $y_i = 1$, ***tức là đối với các điểm thuộc lớp dương***
+ $\langle w, x_i \rangle + b \le -1$ nếu $y_i = -1$, ***tức là đối với các điểm thuộc lớp âm***

***Chắc các bạn còn nhớ câu hỏi vì sao lại chọn số 1 và -1 đúng hông?***

Câu trả lời mình trả lời cho các bạn là: ***Các bạn chọn số mấy cũng được***.

Ví dụ các bạn chọn $\lambda$ và $-\lambda$ với $\lambda > 0$, khi đó điều kiện của bài toán tối ưu trở thành:

+ $\langle w, x_i \rangle + b \ge \lambda$ nếu $y_i = \lambda$, ***tức là đối với các điểm thuộc lớp dương***
+ $\langle w, x_i \rangle + b \le -\lambda$ nếu $y_i = -\lambda$, ***tức là đối với các điểm thuộc lớp âm***


Đến đây các bạn đặt:
+ $w' = \frac{w}{\lambda}$
+ $b' = \frac{b}{\lambda}$

Khi đó điều kiện tối ưu lại trở về như lúc ban đầu:
+ $\langle w', x_i \rangle + b' \ge 1$ nếu $y_i = \lambda$, ***tức là đối với các điểm thuộc lớp dương***
+ $\langle w', x_i \rangle + b' \le -1$ nếu $y_i = -\lambda$, ***tức là đối với các điểm thuộc lớp âm***

Đối với bài toán phân loại hai lớp giả sử ***nam và nữ*** thì lúc đó với mỗi điểm dữ liệu $(x_i, y_i)$ , giá trị $y_i$ không phải là số mà chỉ là chúng ta quy ước. ***-1 là nữ, 1 là nam***. 

***Ta có thể đưa bài toán của chúng ta về bài toán tương đương sau:***

Xét tập dữ liệu $D = [(x_1, y_1), (x_2, y_2), ..., (x_r, y_r)]$

***Cực tiểu hóa $\frac{\langle w, w \rangle}{2}$ với điều kiện $y_i(\langle w, x_i \rangle + b) \ge 1, \forall i = 1...r$***

Ta thấy rằng:

+ $\frac{\langle w, w \rangle}{2}$  là ***hàm lồi***
+ Miền ràng buộc là giao của hai siêu phẳng $\langle w, x_i \rangle + b \ge 1$ và $\langle w, x_i \rangle + b \le -1$. Bài toán của chúng ta đã giả sử rằng ***luôn tồn tại một siêu phẳng mà nó tách biệt 2 lớp*** vì vậy nên ***miền ràng buộc là miền khác rỗng***

Từ hai tính chất trên ta suy ra được bài toán tối ưu của chúng ta là bài ***toán quy hoạch lồi***. Tính chất này đã giúp chúng ta một vấn đề siêu quan trọng chính là ***nghiệm tối ưu $w^*$ tìm được là nghiệm tối ưu toàn cục***

Ta nhận thấy đây là bài toán tối ưu có ràng buộc vì vậy ta có thể đưa bài toán về bài toán trên về bài toán sau đây ***(1)***:

$$arg \min\limits_{w, b} \max\limits_{\alpha \ge 0} L(w, b, \alpha)= arg \min\limits_{w, b} \max\limits_{\alpha \ge 0}(\frac{1}{2} \langle w, w\rangle - \sum\limits_{i = 1}^r\alpha_i[y_i (\langle w, x_i\rangle + b) - 1])$$

Hàm $L(w, b, \alpha) = \frac{1}{2} \langle w, w\rangle - \sum\limits_{i = 1}^r\alpha_i[y_i (\langle w, x_i\rangle + b) - 1]$ là hàm ***Lagrange*** với $\alpha_i \ge 0, \forall i = 1...r$


#### 1.2.2 Giải quyết bài toán tối ưu

##### 1.2.2.1 Áp dụng định lý KKT 

Bạn nên đọc bài viết ***[Những kiến thức tối ưu cơ bản]()*** để có thể hiểu rõ hơn và dễ dàng ôn tập những gì mình viết ở dây

Định lý ***KKT*** là điều kiện cần để ***một điểm là điểm tối ưu địa phương***. Tuy nhiên đối với ***bài toán quy hoạch lồi*** thì điều kiện ***KKT*** chính là ***điều kiện cần và đủ để 1 điểm là điểm tối ưu toàn cục*** 

Áp dụng các điều kiện ***KKT*** vào bài toán ***(1)*** ta có

$w, b$ là nghiệm tối ưu toàn cục khi và chỉ khi phải thỏa mãn tất cả những điểu kiện sau đây (bài toán của chúng ta là bài toán ***quy hoạch lồi***)

***a.*** $\frac{\partial L}{\partial w} = 0$
***b.*** $\frac{\partial L}{\partial b} = 0$
***c.*** $y(\langle w, x \rangle + b) - 1 \ge 0$
***d.*** $\alpha_i \ge 0, \forall i = 1...r$
***e.*** $\alpha_i[y_i (\langle w, x\rangle + b) - 1] = 0$

Nhìn vào điều kiện ***e*** ta thấy:

Khi $\alpha_i > 0$ thì ta có: 

$$y_i (\langle w, x\rangle + b) - 1 = 0$$

$$\iff$$

$$y_i (\langle w, x\rangle + b) = 1$$

***Hay nói các khác thì điểm $(x_i, y_i)$ trong tập dữ liệu $D$ nằm trên một trong hai siêu phẳng dương hoặc âm***

Những điểm $(x_i, y_i)$ như trên người ta gọi là vector hỗ trợ ***(support-vector)***

Và những vector không hỗ trợ ***(non-support vector)*** thì sẽ tương ứng với $\alpha_i = 0$

Tuy nhiên việc giải 5 điều kiện ***a, b, c, d, e*** khá phức tạp và khó do xuất hiện thêm các ***bất phương trình***, vì vậy sẽ cần 1 ***cách giải quyết khác hiệu quả*** cho bài toán ***(1)***

#### 1.2.2.2. Sử dụng đạo hàm

Ta thấy bài toán ***(1)*** của chúng ta có dạng ***Minimax***, củ thể ***min theo $w, b$*** và ***max theo $\alpha$***

$$arg \min\limits_{w, b} \max\limits_{\alpha \ge 0} L(w, b, \alpha)= arg \min\limits_{w, b} \max\limits_{\alpha \ge 0}(\frac{1}{2} \langle w, w\rangle - \sum\limits_{i = 1}^r\alpha_i[y_i (\langle w, x_i\rangle + b) - 1])$$

Hàm $L(w, b, \alpha) = \frac{1}{2} \langle w, w\rangle - \sum\limits_{i = 1}^r\alpha_i[y_i (\langle w, x_i\rangle + b) - 1]$ là hàm ***Lagrange*** với $\alpha_i \ge 0, \forall i = 1...r$

***Ta sẽ đi tìm min theo $w, b$ được kết quả là $K(\alpha)$ và tìm max của $K(\alpha)$ theo biến $\alpha$***

Ta có:

$$\frac{\partial L(w, b, \alpha)}{\partial w} = w - \sum\limits_{i = 1}^r\alpha_iy_ix_i = 0$$

$$\iff$$

$$w = \sum\limits_{i = 1}^r\alpha_iy_ix_i$$

***Chú ý:*** $w, x_i$ là các ***vector có cùng chiểu*** với $(x_i, y_i)$ là 1 điểm dữ iệu thuộc tập ***training $D$***

$$\frac{\partial L(w, b, \alpha)}{\partial b} = - \sum\limits_{i = 1}^r\alpha_ix_i = 0$$
$$\iff$$

$$\sum\limits_{i = 1}^r\alpha_ix_i = 0$$

Ta có các ***điểm tối ưu $(w, b)$*** thỏa mãn $w = \sum\limits_{i = 1}^r\alpha_iy_ix_i$ và $\sum\limits_{i = 1}^r\alpha_ix_i = 0$, thay vào phương trình $L(w, b, \alpha)$ ban đàu ta được:

$$K(\alpha) = \frac{1}{2}(\sum\limits_{i = 1}^r\alpha_iy_ix_i)^2 - \sum\limits_{i = 1}^r\alpha_i[y_i(\langle  \sum\limits_{j = 1}^r\alpha_jy_jx_j, x_i\rangle + b) - 1]$$

$$\iff$$

$$K(\alpha) = \sum\limits_{i = 1}^r\alpha_i - \frac{1}{2}\sum\limits_{i,j = 1}^r\alpha_i\alpha_jy_iy_j\langle x_i, x_j\rangle$$


Cách giải bài toán minimax như trên giúp ta tìm được ***bài toán đối ngẫu*** của bài toán gốc. Do vậy ta có ***bài toán đối ngẫu*** như sau:

***Cực đại hóa***

$$K(\alpha) = \sum\limits_{i = 1}^r\alpha_i - \frac{1}{2}\sum\limits_{i,j = 1}^r\alpha_i\alpha_jy_iy_j\langle x_i, x_j\rangle$$


***Với điều kiện***

+ $\alpha_i \ge 0, \forall i = 1...r$
+ $\sum\limits_{i = 1}^r\alpha_iy_i = 0$


***Điều kiện $\sum\limits_{i = 1}^r\alpha_iy_i = 0$ được suy ra từ $\sum\limits_{i = 1}^r\alpha_ix_i = 0$ theo quy tắc đối ngẫu***

Ta thấy việc giải quyết bài toán đối ngẫu ***đơn giản hơn rất nhiều*** so với bài toán ban đầu

Giả sử chúng ta đã tìm được giá trị $\alpha^*$ tối ưu, khi đó có thể suy ngược lại giá trị tối ưu $w^*, b^*$ theo điều kiện ***KKT***

+ $w^* = \sum\limits_{i = 1}^r\alpha_iy_ix_i$
+ Từ điểu kiện ***e*** trong 5 điều kiện ở mục ***1.2.2.1*** ta có:
  $\alpha_i[y_i (\langle w, x\rangle + b) - 1] = 0$
  
  Với giá trị $\alpha_i \neq 0$ thì suy được giá trị:
  $b^* = b = y_i - \langle w^*, x_i \rangle$
  
  Khi đó: ***Với một mẫu dữ liệu mới z đến*** thì ta sẽ tính giá trị $f(z) = \langle w^*, z \rangle + b$

Nếu $f(z) > 0$ thì ta sẽ phân loại vào lớp ***positive***, ngược lại ta sẽ phân loại vào lớp ***negative***

Chúng ta có một tính chất sau: ***Số lượng các giá trị $\alpha_i \neq 0$ rất ít***. Và như mình đã nói trên: $\alpha_i \neq 0$ sẽ tương ứng với ***vector hỗ trợ***, do đó:

Gọi $SV$ là tập các vector hỗ trợ

+ $w_*$ là ***tổ hợp tuyến tính của các vector hỗ trợ***: $w^* = \sum\limits_{x_i \in SV}\alpha_iy_ix_i$
+ $f(z) = \langle w^*, z \rangle + b =  \sum\limits_{x_i \in SV}\alpha_iy_i \langle x_i, z \rangle + b^*$. ***Do số lượng các vector hỗ trợ ít nên biểu thức này thể hiện tính chất thưa của phương pháp SVM***


## 2. Giải quyết nhiếu lỗi trong cách giải quyết trên

Như chúng ta vừa đọc ***mục 1*** trên thì ta thấy để áp dụng những phương pháp trên thì chúng ta phải có 1 giải thuyết là ***tập dữ liệu của chúng ta có thể được tách hoàn toàn*** bới 1 siêu phẳng.

Vậy nếu ***không tách được bằng siêu phẳng thì thế nào***?

+ ***Do dữ liệu có quá nhiều nhiễu khiến cho 2 lớp dẽ bị chồng lấn và giao thoa với nhau***

+ ***Phải dùng mặt cong mới tách được***

Để giải quyết vấn đề này, mình xin giới thiệu một kỹ thuật cải thiện cho ***nguyên lý lề cực đại*** đó là ***Soft-margin SVM***

### 2.1 Soft-margin SVM

Với trường hợp tập dữ liệu có thể ***phân tách tuyến tính với nhau được*** thì miền xác định dưới đây ***luôn luôn khác rỗng***

$$y_i(\langle w, x_i \rangle + b) \ge 1, \forall i = 1...r$$

Tuy nhiên nếu ***tập dữ liệu không tách được tuyến tính*** thì ***miền xác đinh trên là miền rỗng***

Phương pháp ***Soft-margin SVM*** sinh ra để giải quyết vấn đề này.

Củ thể, tại mỗi bất phương trình $y_i(\langle w, x_i \rangle + b) \ge 1, \forall i = 1...r$ thì phương pháp sẽ có ***thêm một tham số*** $\xi_i$ để cho ***miền định không rỗng***

Nhìn vào hình vẽ trên ta thấy có một số điểm lỗi như:

+ ***Điểm thuộc lớp xanh thì lại chạy sang lớp đỏ***
+ ***Điểm thuộc lớp đỏ thì lại chạy sang lớp xanh***

Vì vậy:

+ Nhưng điểm nào lỗi thì ta sẽ cho giá trị $\xi_i > 1$ vào bất phương trình
+ Những điểm nào không lỗi thì ta sẽ cho giá trị $\xi_i = 0$ vào bất phương trình

Củ thể:

+ $\langle w, x_i \rangle + b \ge 1 - \xi_i$, nếu $y_i = 1$
+ $\langle w, x_i \rangle + b \le -1 + \xi_i$ nếu $y_i = -1$

***Và miền xác định của chúng ta bây giờ là:***


+ $y_i(\langle w, x_i \rangle + b) \ge 1 - \xi_i, \forall i = 1...r$
+ $\xi_i \ge 0$

Có một vấn đề đặt ra như sau: ***Ta chỉ đi cực tiêu hóa $\langle w, w \rangle$***

Ta luôn luôn tồn tại những giá trị $\xi_i > 0$ thỏa mãn miền ràng buộc của chúng ta vì khi cho ***$\xi_i$*** cang lớn thì ta thấy $w$ ***luôn luôn tiến về 0*** sẽ thỏa mãn và ***giá trị nhỏ nhất có thể ngày càng tiến về 0***. Việc quá tập trung cực tiểu hóa  $\langle w, w \rangle$ sẽ dẫn đến cho mô hình bị ***overfitting*** và ***không có khả năng tổng quát hóa trong tương lai***

Ta cần sử dụng một ***đại lượng hiệu chỉnh***. Hay nói cách khác ta cần ***phạt*** các hệ số $\xi_i$ 

Củ thể:

Ta sẽ đi ***cực tiểu hóa $\frac{1}{2} \langle w, w \rangle + C \sum\limits_{i=1}^r\xi_i$ với $C > 0$*** thay vì chỉ tập trung ***cực tiểu hóa $\langle w, w \rangle$***

Ý nghĩa của hằng số phạt $C$ là nếu $C$ càng lớn thì độ lớn của các hệ số $\xi_i$ sẽ bị ***giới hạn trong một miền giá trị nào đó chứ không quá lớn được***

***Tổng kết lại ta có bài toán sau:***

Với hằng số phạt $C > 0$ chọn trước

***Cực tiểu hóa***

$$\frac{1}{2}\langle w, w \rangle + C \sum\limits_{i=1}^r\xi_i$$

***Với điều kiện:***

+ $y_i(\langle w, x_i \rangle + b) \ge 1 - \xi_i, \forall i = 1...r$
+ $\xi_i \ge 0, \forall i = 1...r$

### 2.2 Giải quyết bài toán

Ta có hàm ***Lagrange*** cho bài toán trên như sau:

$$L = \frac{1}{2} \langle w, w\rangle + C \sum\limits_{i=1}^r\xi_i -  \sum\limits_{i=1}^r\alpha_i[y_i(\langle w, x_i \rangle + b) - 1 + \xi_i] - \sum\limits_{i = 1}^r\mu_i\xi_i$$

***Trong đó: $\alpha_i \ge 0, \mu_i \ge 0, \forall i = 1...r$***

Ta có các điểu kiện ***KKT*** cho hàm Lagrange như sau:

+ $\frac{\partial L}{\partial w} = w - \sum\limits_{i = 1}^r\alpha_iy_ix_i = 0$
+ $\frac{\partial L}{\partial b} = -\sum\limits_{i = 1}^r\alpha_iy_i = 0$
+ $\frac{\partial L}{\partial \xi_i} = C - \alpha_i - \mu_i = 0$
+ $y_i(\langle w, x_i \rangle + b) \ge 1 - \xi_i$
+ $\xi_i \ge 0$
+ $\alpha_i \ge 0, \mu_i \ge 0$
+ $\mu_i\xi_i = 0$
+ $\alpha_i(y_i(\langle w, x_i\rangle) - 1 + \xi_i) = 0$


Tương tự như mục trên ta có được bài toán đối ngẫu là:

***Cực đại hóa***

$$K(\alpha) = \sum\limits_{i = 1}^r\alpha_i - \frac{1}{2}\sum\limits_{i, j = 1}^r\alpha_i\alpha_jy_iy_j\langle x_i, x_j \rangle$$

***Với điều kiện***

+ $\sum\limits_{i = 1}^r\alpha_iy_i = 0$
+ $0 \le \alpha_i \le C, \forall i = 1...r$

Ta thấy rằng $\xi_i$ và $\mu_i$ không xuất hiện trong hàm đối ngẫu

Ta xét một số trường hợp đặc biệt:

+ $\alpha_i = 0 \Rightarrow \mu_i = C > 0 \Rightarrow \xi_i = 0 => y_i(\langle w, x_i \rangle + b) \ge 1$
+ $0 < \alpha_i < C \Rightarrow \mu_i = C - \alpha_i > 0 \Rightarrow \xi_i = 0 => y_i(\langle w, x_i \rangle + b) = 1$
+ $\alpha_i = C \Rightarrow y_i(\langle w, x_i \rangle + b) \le 1$

Từ các trường hợp đặc biệt trên ta có 1 số nhận xét sau:

+ Hầu hết các điểm nằm đúng nghĩa là các điểm thuộc lớp dương sẽ nằm gần như tất cả ở phần dương và nhưng điểm thuộc phần âm sẽ nằm gần như tất cả ở phân âm. Trường hợp này ứng với $\alpha_i = 0$
+ Một số điểm được gọi là ***vector hỗ trợ - support vector***nằm trên các siêu phẳng âm và dương. Trường hợp này ứng với trường hợp $0 < \alpha_i < C$
+ Một số điểm nhiễu lỗi tức là các điểm thuộc lớp dương nhưng lại nhảy sang vùng lớp âm và những điểm thuộc lớp âm lại nhảy sang vùng lớp dương. Trường hợp này ứng với $\alpha_i = C$

## 3. SVM phi tuyến

Vấn đề đặt ra ở mục này mình muốn đề cập đến đó chính là ***Nếu không tồn tại siêu phẳng để tách tập dữ liệu mà tồn tại mặt cong để tách tập dữ liệu thì phải giải quyết như thế nào?***

![](https://i.imgur.com/kqW3DeM.png)

***Hình ảnh thứ 2 mô tả vấn đề mà chúng ta đang nói đến***

***Chú ý:*** Nếu chúng ta cứ dùng ***một siêu phẳng*** cho trường hợp như thế này thì ***lỗi nhận được rất cao và mô hình không có khả năng tổng quát hóa trong tương lai***

Vậy ý tưởng ở đây là gì?

+ Bằng 1 cách nào đó chúng ta phải đưa bộ dữ liệu ***sang 1 chiều không gian mới*** sao cho ***bộ dữ liệu mới có thể được phân tách bằng siêu phẳng***
+ Giải quyết trên bộ dữ liệu mới bằng các phương pháp cho ***bài toán SVM tuyến tính*** đã đề cập phía trên

### 3.1. Một số vấn đề khi chuyển đổi không gian dữ liệu

![](https://i.imgur.com/pmuy9TT.png)


Như chúng ta thấy ở hình vẽ chúng ta thấy rằng: Ban đầu bộ dữ liệu trong không gian 2 chiều ***không thể được tách bằng 1 siêu phẳng được***, tuy nhiên sau khi chuyển sang 1 chiều không gian khác có số chiều là 3 thì ***bộ dữ liệu đã có thể được tách bằng một siêu phẳng***

Ta có 1 số thuật ngữ sau:

+ ***Input space:*** không gian ban đầu của bộ dữ liệu
+ ***Feature space***: Không gian đặc trưng - không gian sau khi bị chuyển đổi từ ***Input space***

    $$\text{Input space } \rightarrow \text{ Feature space}$$

Nên chọn ***Feature space*** có số chiều cao hơn ***Input space***. Vì sao lại như vậy?


+ Với dữ liệu phân tách tuyến tính trong không gian ***Input space*** được thì luôn tồn tại không gian ***Feature space*** có số chiều cao hơn mà nếu chuyển dữ liệu ban đầu sang không gian ***Feature space*** đó thì bộ dữ liệu thu được tách được tuyến tính.
    + Các ***chiều được tăng thêm*** trong không gian mới có vai trò kéo 2 lớp điểm ***positive*** và ***negative*** về hai phía. Phép chuyển đổi phải làm cho các giá trị trên các ***chiều không gian tăng thêm*** được chia làm 2 vùng cho 2 lớp điểm ***positive*** và ***negative***
    + Ví dụ như hình vẽ trên: Phép chiếu các điểm sau khi được ***chuyển sang không gian 3 chiều*** lên trục ***z*** tạo thành ***2 vùng điểm màu xanh và đỏ*** tách biệt nhau.

+ ***Học đại số ai cũng biết:*** Nếu $A \rightarrow B$ thì $\bar{B} \rightarrow \bar{A}$. Khi đó ***theo dấu chấm thứ nhất*** thì: Với dữ liệu không phân tách tuyến tính trong chiều không gian ***Feature space*** thì cũng không phân tách được tuyến tính trên không gian ***Input space*** có số chiều nhỏ hơn.

Ta có: ***Phép biến đổi $\phi$ giúp chúng ta thỏa mãn tập dữ liệu sau khi chuyển đổi phân tách tuyến tính được.***

Giả sử bộ dữ liệu ban đầu $D = [(x_1, y_1), (x_2, y_2), ..., (x_r, y_r)]$. Khi đó bộ dữ liệu sau khi được chuyển đỗi là: $F = [(\phi(x_1), y_1), (\phi(x_2), y_2), ..., (\phi(x_r), y_r)]$

![](https://i.imgur.com/nX9svyq.png)

### 3.2 Giải quyết bài toán

Sau khi đã có được bộ dữ liệu chuyển đổi có thể phân tách tuyến tín được thì cách giải quyết tương tự như ***bài toán SVM tuyến tính***. Củ thể:

+ Bài toán gốc:
***Cực tiểu hóa***
$$L = \frac{1}{2}\langle w, w\rangle + C\sum\limits_{i = 1}^r\xi_i$$
***Với điều kiện:***
    + $y_i(\langle w, \phi(x_i) \rangle + b) \ge 1 - \xi_i, \forall i = 1...r$
    + $\xi_i \ge 0, \forall i = 1...r$

+ Bài toán đối ngẫu:
***Cực đại hóa***
$$L_D = \sum\limits_{i = 1}^r - \frac{1}{2}\sum\limits_{i, j = 1}^r\alpha_i\alpha_jy_iy_j\langle \phi(x_i), \phi(x_j) \rangle$$
***Với điều kiện:***
    + $\sum\limits_{i = 1}^r\alpha_iy_i = 0$
    + $0 \le \alpha_i \le C, \forall i = 1...r$

+ Dự đoán:
$$f(z) = \langle w^*, \phi(z) \rangle + b^* = \sum\limits_{x_i \in SV} \alpha_iy_i\langle \phi(x_i), \phi(z) \rangle + b^*$$
    + $f(z) > 0 \Rightarrow z \in \text { Positive class}$
    + $f(z) < 0 \Rightarrow z \in \text { Negative class}$

Khi chuyển dữ liệu từ không gian ***Input space*** sang không gian ***Feature space*** có nhiều chiều hơn thì chúng ta gặp phải vấn đề gì?

Cho dù tập dữ liệu ban đầu có nhiều điểm dữ liệu thi khi chuyển đổi sang không gian có nhiều chiều hơn thì số điểm dữ liệu so với không gian ấy có thể nói như ***Những hạt cát trong sa mạc***. Số điểm dữ liệu ít như vậy hay nói cách khác chúng ta sẽ phải làm việc trên ***bộ dữ liệu rất thưa***. 

Chúng ta cũng có thể biết rằng khi ***dữ liệu quá ít*** thì dễ dẫn đến trường hợp ***overfitting***. 

### 3.3 Kernel trong SVM là gì? 

Như chúng ta thấy trong các biếu thức của mục trên đều có các đại lượng tích vô hướng có dạng $\langle \phi(x), \phi(y)\rangle$. Các đại lượng dạng như thế này goi là ***hàm nhân - kernel function***

Có rất nhiều loại hàm nhân, trong ***SVC của gói svm thuộc thư viện sklearn*** hỗ trợ một số hàm nhân như ***linear, poly, rbf, sigmoid***. Ở này mình sẽ giới thiệu một số hàm nhân nhé.

+ ***Linear Kernel - linear:***
$$K(x, y) = \langle x, y\rangle$$
+ ***Polynomial Kernel - poly:***
$$K(x, y) = (\langle x, y\rangle + \theta)^d, \theta \in R, d \in N$$
+ ***Gaussian radial basis function - rbf:***
$$K(x, y) = e ^ {-\frac{\parallel x - y \parallel ^ 2}{2 \sigma^2}}, \sigma > 0$$
+ ***Sigmoid - sigmoid:***
$$K(x, y) = tanh(\beta\langle x, y\rangle  + \lambda) = 1 - \frac{2}{\exp(2\beta\langle x, y\rangle + 2\lambda + 1)}, \beta, \lambda \in R$$

## 4. Thực hành

### 4.1 Giới thiệu bài toán

Trong mục này mình giới thiệu một ví dụ cơ bản như sau:

***Pulsars*** là một loại ngôi sao hiếm và có nhiều ứng dụng quan trọng. Nhiệm vụ của chúng ta là phát hiện hay nói cách khác tìm ra các ngối sao đó

Chúng ta có thông số của 12528 mẫu dữ liệu. Thông tin của từng mẫu dữ liệu bao gồm: ***Mean of the integrated profile, Standard deviation of the integrated profile, Excess kurtosis of the integrated profile, ...*** và cột cuối cùng là cột ***target_class*** dùng để biết ngôi sao đó có phải là ngôi sao ***Pulsars*** không. 

![](https://i.imgur.com/wIE9miB.png)
***Bộ dữ liệu đầy đủ ở [đây]()***

### 4.2 Giải quyết bài toán

#### 4.2.1 Khai báo một số thư viện cần sử dụng

+ scikit-learn(sklearn): thư viện hỗ trợ phong phú các mô hình học máy cùng với các hàm đánh giá và huấn luyện đa dạng
+ pandas: thư viện xử lý dữ liệu dạng bảng
+ numpy: thư viện cho đại số tuyến tính dùng để biến đổi ma trận, làm việc với vector
+ matplotlib: thư viện phục vụ vẽ các đồ thị
+ seaborn: thư viện được xây dựng trên matplotlib, dùng để vẽ hình đẹp hơn
+ os: thư viện cung cấp chức năng tương tác với hệ điều hành và cũng như có được các thông tin về nó

```jsx=
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import os
import seaborn as sns
```

#### 4.2.2 Đọc dữ liệu


```jsx=
pulsar_df = pd.read_csv('Data/pulsar_data.csv', sep=',')
print("Kích thước bộ dữ liệu:", pulsar_df.shape)
pulsar_df.head()
```
![](https://i.imgur.com/DjNBRnB.png)

#### 4.2.3 Khám phá và tiền xử lý dữ liệu

+ Kiểm tra số lượng ngôi sao ***Pulsar*** trong bộ dữ liệu ***(Ứng với target_class=1.0)***
```jsx=
pulsar_df.iloc[:, -1].value_counts().plot(kind='bar')
```
![](https://i.imgur.com/LqkeFbZ.png)
***Ta nhận thấy bộ dữ liệu đang gặp phải vấn đề imbalance***

+ Kiểm tra thiếu dữ liệu hay không ***(missing value)***
```jsx=
pulsar_df.isnull().sum()
```
![](https://i.imgur.com/k65Yjc2.png)
Mình giải quyết vấn đề này theo cách đơn giản là ***loại bỏ các mẫu dữ liệu nào có xuất hiện thuộc tính bị thiếu đi***
```jsx=
pulsar_df.dropna(inplace=True)
```

+ Kiểm tra nhiễu, outliers của bộ dữ liệu
![](https://i.imgur.com/fBgGdpH.png)
Ta thấy bộ dữ liệu có quá nhiều ***outliers***. Do vậy chúng ta phải dung ***soft-margin SVM***

+ Do các trương thuộc tính của bộ dữ liệu đều là trường số nên ta ***chuẩn hóa dữ liệu*** để tính toán và tránh những trường hợp không mong muốn  


#### 4.2.4 Phân tách tập train và tập test

```jsx=
from sklearn.preprocessing import StandardScaler

X = pulsar_df.drop(pulsar_df.columns[-1:], axis=1)
y = pulsar_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
```

#### 4.2.5 Huấn luyện và dữ đoán

```jsx=
svm_clf = svm.SVC()
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:", confusion_matrix(y_test, y_pred))
```

```typescript=
Accuracy Score: 0.9784366576819407
Confusion matrix: [[1670    5]
                   [  35  145]]
```

Ở đoạn code trên thì mình dùng ***các tham số mặc định của thư viện***. Chúng ta không được dễ dãi như vậy, cân phải ***đánh giá và tìm tham số một cách cẩn thận hơn***

Trước đó mình có nói tới một vấn đề là ***bộ dữ liệu đang gặp phải vấn đề mất cân bằng*** nên mình quyết định dùng phương pháp ***cross-validation*** để quá trình ***lựa chọn tham số và train mô hình*** mang tính khách quan hơn. ***(Nếu bạn chưa biết cross validation là gì thì có thể tham khảo bài viết của mình tại [đây]())***

```jsx=
def cross_validation(estimator):
    _, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=10, n_jobs=-1, train_sizes=[1.0], scoring='accuracy')
    test_scores = test_scores[0]
    mean, std = test_scores.mean(), test_scores.std()
    return mean, std
```

+ Lựa chọn ***kernel***
```jsx=
title = "Tunning kernel, C = 1"
xlabel = "kernel"
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
mean_scores = []
errors = []

for kernel in tqdm(kernels):
    svm_clf = svm.SVC(kernel = kernel, C = 1)
    mean, std = cross_validation(svm_clf)
    mean_scores.append(mean)
    errors.append(std)

plot(title, xlabel, kernels, mean_scores, errors)

if (os.path.exists('images') == False):
    os.makedirs('images')

plt.savefig('images/svm_tunning_kernel.png', bbox_inches="tight")
plt.show()
```
![](https://i.imgur.com/S2vyAXk.png)
***Ta thấy linear, rbf, poly có vẻ là tốt trong khi đó sigmoid có vẻ là không phù hợp với bộ dữ liệu***. Mình lựa chọn ***kernel function là linear***
+ Tunning tham số $C$

```jsx=
title = "Tunning C, kernel = linear"

xlabel = "C"
C_list = [0.1, 1.0, 2.0, 5.0, 10.0, 100.0]
mean_scores = []
errors = []

for C in tqdm(C_list):
    svm_clf = svm.SVC(kernel = "linear", C = C)
    mean, std = cross_validation(svm_clf)
    mean_scores.append(mean)
    errors.append(std)

plot(title, xlabel, C_list, mean_scores, errors)

if (os.path.exists('images') == False):
    os.makedirs('images')

plt.savefig('images/svm_tunning_C.png', bbox_inches="tight")
plt.show()
```
![](https://i.imgur.com/mgVRf4d.png)

Mình lựa chọn ***$C$ = 2***

***Huấn luyện và dự đoán lại bộ dữ liệu sau khi lựa chọn tham số***

```jsx=
svm_clf = svm.SVC(kernel = "linear", C = 2)
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
```
```jsx=
Accuracy Score: 0.9789757412398922
```

***Độ chính xác cao hơn một chút sau khi được lựa chọn tham số***

+ So sánh độ chính xác với tỷ lệ lớp chiếm nhiều hơn
```jsx=
class_freq_list = np.bincount(y_test)

null_accuracy_score = class_freq_list[0] / (class_freq_list[0] + class_freq_list[1])
print("Null Accuracy Score:", null_accuracy_score)
```
```json=
Null Accuracy Score: 0.9029649595687331
```
***Ta thấy tỷ lệ lớp 0 (lớp chứa đa số mẫu dữ liệu) thấp hơn hẳn độ chính xác khi ta dự đoán bằng mô hình SVM***. Chứng tỏ mô hình được học ***đang làm việc tốt***

+ Kiểm tra số lượng mẫu dự đoán đúng và sai ở mỗi lớp 
```jsx=
cm_matrix = pd.DataFrame(data=confusion_matrix(y_test, y_test_pred))
sns.heatmap(cm_matrix, annot=True, fmt='d')
```
![](https://i.imgur.com/dpRmZxJ.png)
***Nhận xét***: Mặc dù số lượng mẫu dữ liệu thuộc lớp 1 cực ít so với lớp 0 nhưng số lượng mẫu thuộc lớp 1 dự đoán sai vào lớp 0 cũng không nhiều ***(35 mẫu)*** so với đoán đúng ***(145 mẫu)***

***Chú ý: Nếu bạn chưa biết ý nghĩa của confusion matrix thì mới bạn đọc tại [đây]()***

***Xem toàn bộ source code tại [đây]()***

## 5. Tài liệu tham khảo

+ [Kernel SVM - machinelearningcoban](https://machinelearningcoban.com/2017/04/22/kernelsmv/)
+ [SVM - Trần Quang Khoát - ĐHBKHN](https://users.soict.hust.edu.vn/khoattq/ml-dm-course/L8-SVM.pdf)