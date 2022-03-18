# Mẫu thống kê và vấn đề ước lượng trong xác suất

Trong bài viết này, mình đề cập tới một nội dung ***quan trọng*** vì trong bài viết chứa kiến thức nền tảng để bạn có thể hiểu rõ hơn về hai phương pháp ***MLE*** và ***MAP*** (Hai phương pháp cực kỳ quan trọng đối với các mô hình ***xác suất*** trong học máy.

## 1. Mẫu là gì? 

+ Thu nhập dữ liệu, sắp xếp, tổng hợp và xử lý số liệu được gọi là ***thống kê mô tả***
+ Tổng hợp các số liệu thống kê trên ta có khái niệm ***Mẫu dữ liệu***.
+ Tập cha của tất cả ***mẫu dữ liệu*** là ***tập nền***.
+ Nói một cách nôm na, một ***mẫu dữ liệu*** sẽ phản ánh 1 phần thông tin của ***tập nền***. Vì vậy để có được đầy đủ thông tin về vấn đề liên quan đến dữ liệu thu thập được thì ta phải thao tác trên tập ***nền***. Thế thì sinh ra ***cái định nghĩa mẫu dữ liệu để làm cái gì?***

    + Bạn có đủ tài nguyên để có thể ***handle*** được một tập dữ liệu lớn ?
    + Bạn có kịp thích ứng với sự thay đổi của dữ liệu trong thời đại công nghệ như thế này không ?

Chắc là bạn cũng đã biết câu trả lời. ***Đáp án là không***. Làm việc trên tập ***nền*** là điều không thể. Vì vậy đã sinh ra các giải pháp ***lựa chọn mẫu*** và ***nghiên cứu trên mẫu***.

### 1.1. Lựa chọn mẫu

Có rất nhiều phương pháp để lấy mẫu, và phương pháp nào cũng sẽ phù hợp với từng kiểu dữ liệu.

#### 1.1.1 Simple Random Sampling - Lấy mẫu ngẫu nhiên đơn giản

Như tên gọi, ta lấy một cách ngẫu nhiên một ***tập con*** từ ***tập nền*** một cách ngẫu nhiên:

+ Có hoàn lại: Khi lấy một điểm dữ liệu ra khỏi tập nền thì trong tập nền sinh ra một điểm dữ liệu giống y đúc điểm được lấy ra. Nói cách khác ***lấy mẫu ngẫu nhiên có hoàn lại*** cho phép các phép các điểm dữ liệu được lặp lại

+ Không hoàn lại: Mỗi điểm dữ liệu chỉ được chọn một lần

#### 1.1.2 Clustering Sampling - Lấy mẫu theo nhóm

Chia tập nền thành ***k*** nhóm. Thực hiện lấy mẫu ngẫu nhiên cho ***k*** nhóm này. Tập hợp các mãu này lại cho ta kết quả ***mẫu theo nhóm***. Khi số lượng mẫu mỗi nhóm ***chênh lệch*** với nhau quá, thì ta nên sử dụng phương pháp lấy mẫu này. Trong học máy, lúc chia ***tập train và tập test** để tránh mắc phải vấn đề ***số điểm thuộc mỗi nhóm quá chênh lệch*** ta cũng có phương pháp mang ý tưởng ***lấy mẫu theo nhóm*** đó là ***lấy mẫu phân tầng (stratified Sampling)***

#### 1.1.3 Chọn mẫu có suy luận
Đây là phương pháp chọn mẫu dựa trên ý kiến của chuyên gia về đối tượng nghiên cứu.

### 1.2 Mẫu ngẫu nhiên
Một mẫu được gọi là mẫu ngẫu nhiên có kích thước ***n*** từ ***tập nền*** có biến ngẫu nhiên gốc $X$ nếu mẫu đó là một tập các biến $X_1, X_2, ..., X_n$ thỏa mãn:

+ ***n*** biến $X_1, X_2, ..., X_n$ độc lập với nhau
+ ***n*** biến $X_1, X_2, ..., X_n$ cùng phân phối xác suất với $X$.

Vì các biến ***độc lập với nhau*** nên các ***hàm xác suất hoặc hàm mật độ cho nhiều biến*** được tính bằng ***các hàm xác suất hoặc hàm mật độ thành phần***:

+ Nếu X là biến rời rạc thì:

\begin{equation}
p_n(x_1, x_2, ..., x_n) = P(X_1=  x_1, X_2 = x_2, ..., X_n = x_n) = \prod_{i = 1}^np(X_i = x_i)
\end{equation}

+ Nếu X là biến liên tục thì:

\begin{equation}
f_n(x_1, x_2, ..., x_n) = \prod_{i = 1}^nf(x_i)
\end{equation}

Chắc hẳn chúng ta đã nghe nhiều đến thuật ngữ ***thống kê***. Vậy ***thống kê là gì***.

Nói đơn giản, nói đến thống kê là nói đến hàm số. Một hàm số $Y = g(X_1, X_2, ..., X_n)$ phụ thuộc vào tập giá trị của mẫu ngẫu nhiên.

***Ví dụ:***

+ $g(X_1, X_2. .., X_n) = \frac{1}{n} \sum\limits_{i = 1}^nx_i = \bar{X}$ là một thống kê

+ $g(X_1, X_2. .., X_n) = \frac{1}{n} \sum\limits_{i = 1}^n(X_i - \bar{X})^2$ cũng là một thống kê

***Vì $X_i$ cũng là 1 biến đại diện cho các giá trị $x_i$ nên mình thay $X_i$ cho $x_i$ cho dễ nhìn trong các công thức toán học sau***

## 2. Đặc trưng của mẫu

### 2.1 Trung bình mẫu

\begin{equation}
\bar{X} = \frac{1}{n} \sum\limits_{i = 1}^nx_i
\end{equation}

Do $X_i$ là các biến ngẫu nhiên nên ***$\bar{X}$ cũng là biến ngẫu nhiên***.
Giả sử biến ngẫu nhiên gốc $X$ ***(biến ngẫu nhiên của tập nền)*** có $EX = \alpha$ và $VX = \sigma^2$.

Do các biến $X_i$ thuộc mẫu ngẫu nhiên của tập nền nên $X_i$ có cùng phân phối với biến $X$ (thoe định nghĩa) nên $EX_i = \alpha$ và $VX_i = \sigma^2$.


+ $E\bar{X} = E[\frac{1}{n} \sum\limits_{i = 1}^nX_i] = \frac{1}{n}(EX_1 + EX_2 + ... + EX_n) = \alpha$
+ $V\bar{X} = \frac{1}{n^2}(VX_1 + VX_2 + ... + VX_n) = \frac{\sigma^2}{n}$ (Do các biến $X_i$ đôi một độc lập với nhau)

Từ bài trước mình đã có nhắc tới ý nghĩa của $VX$ là ***đặc trưng cho độ phân tán dữ liệu*** mà $V\bar{X} < VX_i, 1 \le n \le n$ nên ***các giá trị của $\bar{X}$ sẽ ổn định quanh kì vọng hơn so với $X$***

### 2.2 Phương sai mẫu

\begin{equation}
S^2 = \frac{1}{n}\sum\limits_{i = 1}^n(x_i - \bar{X})^2
\end{equation}

Ta có:

\begin{equation}
S^2 = \frac{1}{n}\sum\limits_{i = 1}^n(X_i - \bar{X})^2 = \frac{X_1^2 + X_2^2 + ... + X_n^2}{n} - \frac{2(X_1 + X_2 + ... + X_n)^2}{n} + \frac{(X_1 + X_2 + ... + X_n)^2}{n} = \frac{1}{n} \sum\limits_{i = 1}^nX_i^2 - \frac{1}{n^2}(\sum\limits_{i = }^nX_i)^2
\end{equation}

Do $(X_1 + X_2 + ... + X_n)^2 = \sum\limits_{i = 1}^nX_i^2 + \sum\limits_{i \ne j}X_iX_j$, nên:

\begin{equation}
S^2 = \frac{n - 1}{n^2}\sum\limits_{i = 1}^nX_i^2 - \frac{1}{n^2}\sum\limits_{i \ne j}X_iX_j
\end{equation}

Do các biến $X_i$ đôi một độc lập với nhau và có cùng phân phối với biến gốc $X$ nên: 
+ $E(X_iX_j) = EX_iEX_j = \alpha^2$
+ $E(X_i^2) = E(X^2)$
Khí đó:

\begin{equation}
ES^2 = \frac{n - 1}{n^2}E(X^2) - \frac{n(n - 1)(EX)^2}{n^2} = \frac{n - 1}{n}(EX^2 - (EX)^2) = \frac{n - 1}{n} VX = \frac{n - 1}{n} \sigma^2 
\end{equation}

Ta thấy $ES^2 \ne \sigma^2$ nên người ta sẽ hiệu chỉnh $S^2$ sao cho $ES^2 = \sigma^2$. Gọi đó là ***phương sai mẫu hiểu chỉnh $s^2$***:

\begin{equation}
s^2 = \frac{n}{n-1}S^2 = \frac{1}{n - 1}\sum\limits_{i = 1}^n(x_i - \bar{X})^2
\end{equation}

Ta có:

\begin{equation}
Es^2 = \frac{n}{n-1}ES^2 = \sigma^2
\end{equation}

## 3. Bài toán ước lượng trong xác suất

### 3.1. Ước lượng điểm

#### 3.1.1 Ước lượng tham số.

Giả sử biến ngẫu nhiên $X gốc đã biết ***luật phân phối*** (có thể là tuân theo phân phối chuẩn, phân phối đều, ...), nhưng chưa biết các tham số $\theta$ ứng với công thức ***hàm xác suất hoặc hàm mật độ***.

***Ước lượng tham số là quá trình dựa vào các mẫu quan sát $x_1, x_2, ...x_n$n ta xác định được $\theta$***. Gọi kết quả ước lượng của $\theta$ là $\hat{\theta}$:

+ Nếu $\hat{\theta}$ là một điểm thì đây là quá trình ***ước lượng điểm***
+ Nếu $\hat{\theta}$ là một khoảng thì là quá trình ***ước lượng khoảng***

#### 3.1.2. Làm sao để ước lượng $\theta$

Rõ ràng ta luôn muốn ước lượng giá trị $\theta$ là giá trị tốt nhất.

Với mẫu quan sát $x_1, x_2, ..., x_n$ đã biết trước, nghĩa là có tồn tại giá trị $\theta$, chỉ là mình không biết nó có giá trị như thế nào?

Vì vậy ta cần tìm giá trị $\hat{\theta}$ sao cho giá trị này phải khiến cho ***hàm mật độ đối với biến liên tục hoặc hàm xác suất ứng với biến rời rạc*** khớp nhất với xác suất thu được mẫu quan sát $x_1, x_2, ..., x_n$. Nghĩa là:

***Giá trị ước lượng tốt nhất cho $\theta$ là giá trị khiến cho xác suất  xảy ra mẫu các quan sát $x_1, x_2, ..., x_n$ là cao nhất.***

Giả sử biến gốc $X$ có hàm xác suất hoặc hàm mật độ là $f(x \mid \theta)$.

Do mẫu $X_1, X_2, ..., X_n$ là các biến có cùng phân phối với $X$, ***các biến này độc lập với nhau*** nên xác suất xảy ra các quan sát $x_1, x_2, ..., x_n$ tương ứng là các giá trị của các biến ngẫu nhiên $X_1, X_2, ..., X_n$ là:

\begin{equation}
L(x \mid \theta) = L(X_1=x_1, X_2=x_2, ..., X_n=x_n \mid \theta) = \prod\limits_{i=1}^nf(x_i \mid \theta)
\end{equation}

***Ta có:***



Hay: $L(x \mid \hat{\theta}) \ge L(x \mid \theta), \forall\theta$

Như chúng ta đã biết để tìm điểm cực trị thì ta tình ***đạo hàm cấp 1 và so sánh giá trị đạo hàm cấp 2 với 0***.

***Chú ý:*** Việc tính đạo hàm trên tích sẽ khó hơn trên tổng nên mình sẽ lấy ***ln*** hai về của $L(x \mid \theta)$ trước khi lấy đạo hàm. Do hàm $ln(x)$ là hàm đồng biến với $x > 0$ nên ***giá trị tìm được sau khi ln sẽ bằng giá trị trước khi ln***

#### 3.1.3 Các ví dụ về ước lượng điểm cho tham số trong một số phân phối đã học ở [bài trước]()

##### 3.1.3.1 Ước lượng tham số $\lambda$ của phân bố Poisson

Do biến gốc $X$ tuân theo phân phối ***Possion*** nên các biến củ thể $X_i, \forall 1 \le i \le n$ cũng tuân theo phân phối ***Possion***

Ta có công thức hàm xác suất cho mỗi biến $X_i$:

\begin{equation}
p(X_i = x_i \mid \lambda) = e^{-\lambda} \frac{\lambda^{x_i}}{x_i!}, \lambda > 0
\end{equation}

Khi đó:

\begin{equation}
L(x \mid \lambda) = L(X_1=x_1, X_2=x_2, ..., X_n=x_n \mid \lambda) = \prod\limits_{i=1}^nf(x_i \mid \lambda) =  e^{-n\lambda} \frac{\lambda^{\sum\limits_{i = 1}^nx_i}}{\prod\limits_{i = 1}^nx_i!}
\end{equation}

***Do $L(x \mid \lambda) > 0$ nên lấy logarit cơ số e hai vế, ta có:***

\begin{equation}
ln(L(x \mid \lambda)) = ln(e^{-n\lambda} \frac{\lambda^{\sum\limits_{i = 1}^nx_i}}{\prod\limits_{i = 1}^nx_i!}) = -n\lambda + ln\lambda \times \sum\limits_{i = 1}^nx_i - ln(\prod\limits_{i = 1}^nx_i!)
\end{equation}

***Do hàm $ln(L(x \mid \lambda))$ liên tục $\forall \lambda > 0$, nên ta lấy đạo hàm cấp một thoải mái như con gà mái luôn:***

\begin{equation}
\frac{\partial ln(L(x \mid \lambda))}{\partial \lambda} = -n + \frac{\sum\limits_{i = 1}^nx_i}{\lambda}
\end{equation}

Ta có: 

\begin{equation}
\frac{\partial ln(L(x \mid \lambda))}{\partial \lambda} = 0 \Leftrightarrow \lambda = \frac{\sum\limits_{i = 1}^nx_i}{n}
\end{equation}

Ta có:

\begin{equation}
\frac{\partial^2 ln(L(x \mid \lambda))}{\partial \lambda^2} = 0 \Leftrightarrow \lambda = -\frac{1}{\lambda^2}\sum\limits_{i = 1}^nx_i < 0
\end{equation}


***Chú ý: Do $X_i$ đêu tuân theo phân phối Possion nên giá trị $x_i > 0$***

***Vậy giá trị ước lượng tốt nhất cho tham số $\lambda$ là $\hat{\lambda} = \frac{\sum\limits_{i = 1}^nx_i}{n}$***

##### 3.1.3.2 Ước lượng tham số $\alpha, \sigma^2$ của phân bố chuẩn $N(\alpha, \sigma^2)$

Do biến gốc $X$ tuân theo ***phân phối chuẩn*** nên các biến củ thể $X_i, \forall 1 \le i \le n$ cũng tuân theo ***phân phối chuẩn***

Ta có công thức hàm mật độ cho mỗi biến $X_i$:

\begin{equation}
f(x_i \mid \{\alpha, \sigma \}) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x_i - \alpha)^2}{2\sigma^2}}
\end{equation}

Khi đó:

\begin{equation}
L(x \mid \{\alpha, \sigma \}) = L(X_1=x_1, X_2=x_2, ..., X_n=x_n \mid \{\alpha, \sigma \}) = \prod\limits_{i=1}^nf(x_i \mid \{\alpha, \sigma \}) = (2\pi\sigma^2)^{\frac{-n}{2}}e^{-\frac{\sum\limits_{i = 1}^n(x_i - \alpha)^2}{2\sigma^2}}
\end{equation}

***Do $L(x \mid 
\{\alpha, \sigma \}) > 0$ nên lấy logarit cơ số e hai vế, ta có:***

\begin{equation}
ln(L(x \mid 
\{\alpha, \sigma \})) = ln((2\pi\sigma^2)^{\frac{-n}{2}}e^{-\frac{\sum\limits_{i = 1}^n(x_i - \alpha)^2}{2\sigma^2}}) = -\frac{n}{2}ln(2\pi\sigma^2) -\frac{\sum\limits_{i = 1}^n(x_i - \alpha)^2}{2\sigma^2}
\end{equation}

Do $\sigma = \sqrt{VX} > 0$ nên $ln(L(x \mid 
\{\alpha, \sigma \}))$ khả vi với $\forall \alpha, \sigma > 0$, lấy đạo hàm hai vế ta được:

\begin{equation}
\frac{\partial ln(L(x \mid 
\{\alpha, \sigma \}))}{\partial \alpha} = \frac{1}{\sigma^2}\sum\limits_{i = 1}^n(x_i - \alpha)
\end{equation}

\begin{equation}
\frac{\partial ln(L(x \mid 
\{\alpha, \sigma \}))}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{\sum\limits_{i = 1}^n(x_i - \alpha)^2}{2\sigma^4}
\end{equation}

Giải các phương trình: $\frac{\partial ln(L(x \mid 
\{\alpha, \sigma \}))}{\partial \alpha} = 0, \frac{\partial ln(L(x \mid 
\{\alpha, \sigma \}))}{\partial \sigma^2} = 0$, ta có:

+ $\hat{\alpha} = \bar{X} = \frac{1}{n}\sum\limits_{i=1}^nx_i$
+ $\hat{\sigma}^2 = S^2 = \frac{1}{n}\sum\limits_{i=1}^n(x_i - \bar{X})^2$

Ta có:

\begin{equation}
\frac{\partial^2 ln(L(x \mid 
\{\alpha, \sigma \}))}{\partial \alpha^2} = \frac{-n}{\sigma^2} < 0
\end{equation}

Do $\sum\limits_{i = 1}^n(x_i - \alpha)^2 = n(\hat{\sigma})^2$ nên: 

\begin{equation}
\frac{\partial^2 ln(L(x \mid 
\{\alpha, \sigma \}))}{\partial (\sigma^2)^2} \mid_{\sigma^2 = (\hat{\sigma})^2} = \frac{n}{2 (\hat{\sigma})^4} - \frac{\sum\limits_{i = 1}^n(x_i - \alpha)^2}{(\hat{\sigma})^6} = \frac{n}{2 (\hat{\sigma})^4} - \frac{n(\hat{\sigma})^2}{(\hat{\sigma})^6} = -\frac{n}{2 (\hat{\sigma})^4} < 0
\end{equation}

***vậy ước lượng cho các tham số của phân phối chuẩn là:***

+ $\hat{\alpha} = \bar{X} = \frac{1}{n}\sum\limits_{i=1}^nx_i$
+ $\hat{\sigma}^2 = S^2 = \frac{1}{n}\sum\limits_{i=1}^n(x_i - \bar{X})^2$


##### 3.1.3.3 Ước lượng tham số p đối với phân phối Bec-nu-li

Do biến gốc $X$ tuân theo ***phân phối Bec-ni-li*** nên các biến củ thể $X_i, \forall 1 \le i \le n$ cũng tuân theo ***phân phối Bec-ni-li***

Ta có công thức hàm xác suất cho mỗi biến $X_i$:

\begin{equation}
f(x_i \mid p) = p^{x_i}(1-p)^{1-x_i}, 0 < p < 1
\end{equation}

Khi đó:

\begin{equation}
L(x \mid p) = L(X_1=x_1, X_2=x_2, ..., X_n=x_n \mid p) = \prod\limits_{i=1}^nf(x_i \mid p) = (p)^{\sum\limits_{i=1}^nx_i}(1-p)^{n-\sum\limits_{i=1}^nx_i}
\end{equation}

***Do $L(x \mid 
p) > 0$ nên lấy logarit cơ số e hai vế, ta có:***

\begin{equation}
ln(L(x \mid 
p)) = ln((p)^{\sum\limits_{i=1}^nx_i}(1-p)^{\sum\limits_{i=1}^nx_i}) = lnp \times \sum\limits_{i=1}^nx_i + ln(1-p) \times (n-\sum\limits_{i=1}^nx_i)
\end{equation}

Do $ln(L(x \mid 
p))$ khả vi với $0 < p < 1$, lấy đạo hàm hai vế ta được:

\begin{equation}
\frac{\partial ln(L(x \mid 
p))}{\partial p} = \frac{\sum\limits_{i=1}^nx_i}{p} + \frac{n-\sum\limits_{i=1}^nx_i}{p-1}
\end{equation}

Giải phương trình: $\frac{\partial ln(L(x \mid 
p))}{\partial p} = 0$, ta có:

\begin{equation}
\hat{p} = \frac{\sum\limits_{i=1}^nx_i}{n}
\end{equation}

Ta có:

\begin{equation}
\frac{\partial^2 ln(L(x \mid 
p))}{\partial p^2} = -\frac{\sum\limits_{i=1}^nx_i}{p^2} - \frac{n - \sum\limits_{i=1}^nx_i}{(p-1)^2} < 0
\end{equation}

Do tuân theo ***phân phối Bec-nu-li*** hay $x_i = 0$ hoặc $x_1 = 1$ nên $0 \le \sum\limits_{i=1}^nx_i \le n$


***vậy ước lượng cho các tham số của phân phối Bec-nu-li là:*** $\hat{p} = \frac{\sum\limits_{i=1}^nx_i}{n}$

##### 3.1.3.4 Ước lượng vector tham số $p$ đối với phân phối categorical
Các giá trị của các biến củ thể $X_i$ thay vì nhận một trong 2 giá trị ***{0, 1}*** như trong phân phối ***Bec-nu-li*** thì có thể nhận một trong $m$ giá trị khác nhau.

***Vector $p = [p_1, p_2, ..., p_m], 0 < p_i < 1$*** thỏa mãn $p_1 + p_2 + ... + p_m = 1$

Do biến gốc $X$ tuân theo ***phân phối multinoulli*** nên các biến củ thể $X_i, \forall 1 \le i \le n$ cũng tuân theo ***phân phối multinoulli***

Khi đó:
+ $x_i = [x_i^1, x_i^2, ..., x_i^m]$
+ Nếu $x_i$ nhận giá trị thứ $k$ trong $m$ giá trị thì:
    + $x_i^k = 1$
    + $x_i^j = 0, 1 \le j \ne k \le m$

Ta có công thức hàm xác suất cho mỗi biến $X_i$:

\begin{equation}
f(x_i \mid p) = \prod\limits_{k=1}^m(p_k)^{x_i^k}
\end{equation}

Khi đó:

\begin{equation}
L(x \mid p) = L(X_1=x_1, X_2=x_2, ..., X_n=x_n \mid p) = \prod\limits_{i=1}^nf(x_i \mid p) = \prod\limits_{i=1}^n\prod\limits_{k=1}^m(p_k)^{x_i^k} = \prod\limits_{k=1}^m(p_k)^{\sum\limits_{i=1}^nx_i^k}
\end{equation}


***Do $L(x \mid 
p) > 0$ nên lấy logarit cơ số e hai vế, ta có:***

\begin{equation}
ln(L(x \mid 
p)) = ln(\prod\limits_{k=1}^m(p_k)^{\sum\limits_{i=1}^nx_i^k}) = \sum\limits_{k=1}^m[ln(p_k)\sum\limits_{i=1}^nx_i^k]
\end{equation}



Do $ln(L(x \mid 
p))$ khả vi với $0 < p < 1$, lấy đạo hàm hai vế ta được:

Trong n mẫu, giá trị có thể nhận 1 trong m giá trị. Giả sử:
+ số lượng mẫu nhận giá trị thứ 1 trong m giá trị là $n_1$
+ số lượng mẫu nhận giá trị thứ 2 trong m giá trị là $n_2$
+ ***$...$***
+ số lượng mẫu nhận giá trị thứ n trong m giá trị là $n_m$

Khi đó:



\begin{equation}
ln(L(x \mid 
p)) = ln(\prod\limits_{k=1}^m(p_k)^{\sum\limits_{i=1}^nx_i^k}) = \sum\limits_{k=1}^m[ln(p_k)n_k]
\end{equation}

Do $\sum\limits_{k = 1}^mp_k = 1$ nên:


\begin{equation}
p = arg \max\limits_p ln(L(x \mid 
p)), \text{ với } \sum\limits_{k = 1}^mp_k = 1
\end{equation}

***tương đương***

\begin{equation}
p = arg \max\limits_p ln(L(x \mid 
p)) + \lambda (1 - \sum\limits_{k = 1}^mp_k)
\end{equation}

Ta có:
+ Tính đạo hàm theo từng thành phần $p_i$ của vector tham số $p$
\begin{equation}
\frac{\partial [ln(L(x \mid 
p)) + \lambda (1 - \sum\limits_{k = 1}^mp_k)]}{\partial p_i} = \frac{n_i}{p_i} - \lambda, 1 \le i \le m
\end{equation}

+ Tính đạo hàm theo tham số $\lambda$


\begin{equation}
\frac{\partial [ln(L(x \mid 
p)) + \lambda (1 - \sum\limits_{k = 1}^mp_k)]}{\partial \lambda} = 1 - \sum\limits_{k = 1}^mp_k
\end{equation}

***Giải các phương trình đạo hàm = 0:***

$$\lambda = \hat{\lambda} = n_1 + n_2 + ... + n_m$$

$$p_i = \hat{p_i} = \frac{n_i}{n_1 + n_2 + ... + n_m}, 1 \le i \le m$$

Ta có: 

\begin{equation}
\frac{\partial^2 [ln(L(x \mid 
p)) + \lambda (1 - \sum\limits_{k = 1}^mp_k)]}{\partial \lambda^2} = 0
\end{equation}

\begin{equation}
\frac{\partial^2 [ln(L(x \mid 
p)) + \lambda (1 - \sum\limits_{k = 1}^mp_k)]}{\partial p_i^2} = -\frac{n_i}{p_i^2} < 0
\end{equation}


Vậy ước lượng vector tham số p là:

$$p = [\hat{p_1}, \hat{p_2}, ..., \hat{p_m}]$$

$$p_i = \hat{p_i} = \frac{n_i}{n_1 + n_2 + ... + n_m}, 1 \le i \le m$$

### 3.2 Ước lượng khoảng

Như chúng ta đã biết, khi chúng ta ước lượng một thứ gì đó: ***điểm thi***, ***độ cao của người châu Á, ...***. Việc ước lượng điểm đôi khi sẽ mang lại ***giá trị sai lệch lớn nếu kích thước mẫu nhỏ*** và ***khó đánh giá*** vì vậy chúng ta sẽ đi ***ước lượng khoảng*** vì nó sẽ tin cậy hơn và khách quan hơn. 

***Ước lượng khoảng sẽ được xây dựng thông qua ước lượng điểm***

Ta cần ước lượng khoảng tham số $\theta$ với độ tin cậy $1 - \alpha$. Nghĩa là ta sẽ tìm khoảng $(\theta_1, \theta_2)$ sao cho:

$$P(\theta_1 < \theta < \theta_2) = 1 - \alpha$$

Ở trong bài ***[này]()*** mình có nhắc đến khái niệm ***phân vị*** như sau:

Giá trị phân vị $k$% của biến ngẫu nhiên $X$ là 1 giá trị $x_k$ thỏa mãn:

$$P(X < x_k) = \frac{k}{100}$$


Nghĩa là ta sẽ tìm các phân vị $\theta_{\alpha_1}, \theta_{1 - \alpha_2}$ sao cho $\alpha_1 + \alpha_2 = 1$:
+ $P(\theta < \theta_{\alpha_1}) = \alpha_1$
+ $P(\theta < \theta_{1 - \alpha_2}) = 1-\alpha_2$

Khi đó:

$$P(\theta_{\alpha_1} < \theta < \theta_{1 - \alpha_2}) = 1-\alpha_2 - \alpha_1 = 1 - \alpha$$

***Mưa đã có mũ, nắng đã có ô, đọc không hiểu đã có ngay ví dụ***

Mình xét bài toán ước lượng khoảng tin cậy cho kỳ vọng

Ví dụ này xét đến phân phối chuẩn.

Giả sử biến gốc $X$ (biến của tập nền) tuân theo ***phân phối chuẩn*** với kỳ vọng $EX = \mu$, phương sai $VX = \sigma^2$.

Ta sẽ đi ước lượng khoảng tin cậy $\mu$ với độ tin cậy $1 - \alpha$ cho trước.

Giả sử: Đã biết phương sai $\sigma^2 = \sigma_0^2$

Đặt:

$$Z = \frac{\bar{X} - \mu}{\sigma_0}\sqrt{n}$$

Khi đó: $Z$ sẽ tuân theo phân phối chuẩn với ***kỳ vọng = 0*** và ***phương sai = 1***. Hay nói cách khác $Z$ ***tuân theo [phân phối chuẩn rút gọn]()***

Ta sẽ đi tìm $z_{\alpha_1}, z_{1 - \alpha_2}$ sao cho:

+ $\alpha_1 + \alpha_2 = \alpha$
+ $P(Z < z_{\alpha_1}) = \alpha_1$
+ $P(Z < z_{1 - \alpha_2}) = 1 - \alpha_2$

Do $Z$ tuân theo phân phối chuẩn nên:
$$P(Z < -z_{1 - \alpha_1}) = P(Z > z_{1 - \alpha_1}) = 1 - P(Z < z_{1 - \alpha_1}) = 1 - (1 - \alpha_1) = \alpha_1 = P(Z < z_{\alpha_1})$$

Khi đó:

\begin{equation}
P(Z < z_{1 - \alpha_2}) - P(Z < z_{\alpha_1}) = 
P(Z < z_{1 - \alpha_2}) - P(Z < -z_{1 - \alpha_1}) = 1 - (\alpha_1 + \alpha_2) = 1- \alpha
\end{equation}

***Hay:***


\begin{equation}
P(-z_{1 - \alpha_1} < Z < z_{1 - \alpha_2}) = 1- \alpha
\end{equation}

Vậy khoảng tin cậy đối với biến $Z$ là $(-z_{1 - \alpha_1}; z_{1 - \alpha_2})$

Khi đó:

$$\bar{X} - \frac{\sigma_0}{\sqrt{n}}z_{1 - \alpha_1} < \mu < \bar{X} + \frac{\sigma_0}{\sqrt{n}}z_{1 - \alpha_2}$$

Với $1 - \alpha$ cho trước, ta có vô số cặp $(\alpha_1, \alpha_2)$, ta chọn 1 số cặp đặc biệt là:

+ Ta cần tìm khoảng tin cậy là đối xứng:
    Khoảng tin cậy đối xứng là khoảng thỏa mãn: $z_{1 - \alpha_1} = z_{1 - \alpha_2}$ hay $1 - \alpha_1 = 1 - \alpha_2$ hay:
    $$\alpha_1 = \alpha_2 = \frac{\alpha}{2}$$

    Đặt $z_b = z_{1 - \frac{\alpha}{2}}$, khi đó:

    $$\bar{X} - \frac{\sigma_0}{\sqrt{n}}z_b < \mu < \bar{X} + \frac{\sigma_0}{\sqrt{n}}z_b$$


    Đại lượng $\epsilon = \frac{\sigma_0}{\sqrt{n}}z_b$ là ***độ chính xác ước lượng***. Nó thể hiện độ lệch trung bình kì vọng với độ tin cậy $1 - \alpha$

+ Tim tìm khoảng tin cậy phải:
Khoảng tin cậy phải là khoảng thỏa mãn: $\alpha_1 = \alpha, \alpha_2 = 0$

    Đặt $z_b = z_{1 - \alpha}$, khi đó:

    $$\bar{X} - \frac{\sigma_0}{\sqrt{n}}z_b < \mu < +\infty$$

+ Tim tìm khoảng tin cậy trái:
Khoảng tin cậy phải là khoảng thỏa mãn: $\alpha_1 = 0, \alpha_2 = \alpha$

    Đặt $z_b = z_{1 - \alpha}$, khi đó:

    $$ -\infty < \mu < \bar{X} + \frac{\sigma_0}{\sqrt{n}}z_b$$


Ta có: việc tính $z_b$ nghĩa là việc tính điểm phân vị $k$% cho trước của biến $Z$ tuân theo ***phân phối chuẩn rút gọn $\phi$***

Ví dụ: 
+ $z_b = z_{1 - \alpha}$, nghĩa là: $\phi(z_b) = \frac{1}{2} - \alpha$. 
+ $z_b = z_{1 - \frac{\alpha}{2}}$, nghĩa là: $\phi(z_b) = \frac{1 - \alpha}{2}$

***Xem lại [bài trước]() nếu đã quên nhé anh em***


Tra bảng ***Láp-la-xơ*** ta có kết quả.

Bạn tham khảo bảng ***Láp-la-xơ*** tại ***[đây](https://cdn.slidesharecdn.com/ss_thumbnails/banggiatrihamlaplace-150919075726-lva1-app6892-thumbnail-4.jpg?cb=1442649495)***

***Khi đi thi kiểu gì cũng có bảng giắ trị hoặc các giá trị trong Láp-la-xơ cho bạn tra***

Bài toán này mình chỉ giải quyết cho trường hợp đã biết ***phương sai***. Bạn có thể tham khảo 1 số cách ước lượng tin cậy cho các ***tỷ lệ, phương sai*** tại bài giảng chương 4 của ***thầy Lê Xuân Lý - ĐHBKHN*** tại [đây](https://drive.google.com/drive/folders/1d-X3Q3-mjwma8H-KH_DeusLt7CiL2MFR).

## 4.Ví dụ:

Doanh thu của một cửa hàng là biến ngẫu nhiên X(triệu/tháng) có độ lệch chuẩn 2
triệu/tháng. Điều tra ngẫu nhiên doanh thu của 500 cửa hàng có qui mô tương tự nhau
ta tính được doanh thu trung bình là 10 triệu/tháng. Với độ tin cậy 95% hãy ước lượng
khoảng cho doanh thu trung bình của cửa hàng thuộc qui mô đó.

***Bài giải:***

Đặt:

$$Z = \frac{\bar{X} - \mu}{\sigma_0}\sqrt{n}$$

Theo công thức trên ta vừa tình được, khoảng ước lượng tin cậy đối xứng của trung bình $\mu$ là:


$$\bar{X} - \frac{\sigma_0}{\sqrt{n}}z_b < \mu < \bar{X} + \frac{\sigma_0}{\sqrt{n}}z_b$$

Với $z_b = z_{1 - \frac{\alpha}{2}}, 1-\alpha=95\text{%}, \sigma=2, \bar{x} = 10, n = 500$

Khí đó: $\phi(z_b) = \frac{1 - \alpha}{2} = 0.475$ hay $z_b = 1.96$ (tra bảng ***Láp-la-xơ*** tại [đây](https://cdn.slidesharecdn.com/ss_thumbnails/banggiatrihamlaplace-150919075726-lva1-app6892-thumbnail-4.jpg?cb=1442649495))

***Thay số vào ta có:***

$$9.825 < \mu < 10.175$$

## 5. Tài liệu tham khảo

+ ***[Slide bài giảng của thầy Lê Xuân Lý - ĐHBKHN](https://drive.google.com/drive/folders/1d-X3Q3-mjwma8H-KH_DeusLt7CiL2MFR)***

+ ***[Giáo trình xác suất thống kê - Tống đình quy](https://drive.google.com/file/d/1Yw02kvncpFp6WiyP9kZWSEuWyvLL3sYO/view)***
























