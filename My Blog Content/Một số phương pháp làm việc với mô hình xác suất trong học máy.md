# Một số phương pháp làm việc với mô hình xác suất trong học máy

Một trong những câu nói hài hước mà mình rất thích đó là: ***"Vì tao chắc chắn là trên đời này không có gì là chắc chắn..."***.

Bạn đang sống trong một thời đại ***biến thiên vạn hóa***, hay nói cách khác, nhiều thứ trong cuộc sống luôn mang những tính không chắc chắn trong đó.

Việc này cũng liên quan đến rất nhiều vấn đề trong học máy:
+ Mô hình được lựa chọn có phù hợp với dữ liệu. ***Đây là 1 điều không chắc chắn***. Có thể rất tuyệt với nếu dùng các mô hình ***tuyến tính*** để xấp xỉ ***dữ liệu có mang tính tuyến tính***. Hay sẽ thật tệ nếu dùng chính mô hình  ***tuyến tính*** để đi xấp xỉ dữ liệu ***phi tuyến***. 

+ Dữ liệu được thu thập có tin cậy và chắc chắn hay không

+ Chúng ta đã biết đối với một số mô hình chúng ta phải ***dùng một số giả thuyết***. Đây cũng là một điều không chắc chắn.

Thay vì cố tình ***lánh mặt nhau*** sao không đối diện sự thật nhỉ?

Như chúng ta đã biết ***nhắc đến sự không chắc chắn*** thì không thể không nhắc đến 1 lĩnh vực nổi tiếng, đó là ***lình vực xác suất*** với những ***lý thuyết hay ho***. 

Việc áp dụng lý thuyết xác suất vào học máy đã sinh ra các mô hình ***xác suất***

Như các bạn cũng biết nói đến ***lý thuyết xác suất*** thì ta phải nói đến các ***luật phân phối xác suất***. Ứng với từng ***luật phân phối xác suất*** ta có ***hàm xác suất ứng với biến rời rạc*** và ***hàm mật độ ứng với biến liên tục*** và các ***hàm số*** này có các tham số. Ở bài viết ***[Mẫu thống kê và vấn đề ước lượng trong xác suất]()***, mình kí hiệu các tham số này là $\theta$.

Công việc của chúng ta bây giờ là ***học một mô hình xác suất tốt nhất với bộ dữ liệu đã cho trước***. Nghĩa là: ***Bạn cần ước lượng tham số $\theta$*** nào là hợp lý sao cho ***mô hình xác suất khớp nhất với bộ dữ liệu***.

Nếu các bạn đã đọc bài viết ***[Mẫu thống kê và vấn đề ước lượng trong xác suất]()*** của mình, thì đã biết dược một cách ***ước lượng điểm*** cho các tham số này. Đó là cách thức ***ước lượng tham số sao cho xác suất để các điểm trong mẫu xuất hiện là cao nhất***. Phương pháp này trong học máy có tên là ***MLE***. Trong bài viết này mình cũng sẽ đề cập thêm một số phương pháp khác nữa.

## 1. Mô hình xác suất

Như chúng ta đã biết, đối với một số mô hình chúng ta hay đặt ra những giả thuyết ***(assumption)***. Ví dụ: Mô hình tuyến tính thì đặt ra giả thuyết là dữ liệu chúng ta đang làm việc cũng xấp xỉ một hàm tuyền tính nào đó.

Đối với mô hình xác suất cũng không có gì khác. Chúng ta đặt ra giả thuyết rằng dữ liệu tuân theo các phân phối nào đó ***(Phân phối bec-nu-li, phân phối chuẩn, phân phối poisson, ...)*** hay chúng ta đặt ra những giả thuyết về quá trình sinh ra dữ liệu

***Trong mô hình xác suất có các loại biến khác nhau và mỗi loại biến lại có những đặc điểm riêng:***

+ Biến quan sát được ***(Observed variable)*** là biến để mô tả những gì quan sát được ***(Ví dụ: Rating của phim, số lượng lượt xem của phim, ...)***

+ Biến ẩn ***(Hidden variable)*** là biến để mô tả những thuộc tính ẩn của dữ liệu mà chúng ta không thể thu thập được ***(Ví dụ: Bộ phim mang nhiều ý nghĩa nhân văn hay không, bộ phim có thực sự hấp dân hay không, ..)***


Trong mô hình xác suất, chúng ta sẽ đặt ra các giả thuyết các ***biến*** sẽ tuân theo các giả thuyết nhất định. Một mô hình xác suất có thể có nhiều biến cũng đồng nghĩa với việc có thể có nhiều giả thuyết.

Ngoài các biến, khi chúng ta đặt các giả thuyết lên các tham số ***(Ví dụ: tham số của hàm xác suất hoặc hàm mật độ)*** thì chúng ta gọi mô hình này là mô hình ***Bayesian***

***Mình có một ví dụ như sau:***

Giả sử mình có bộ data về điểm thi của sinh viên, củ thể ***số lượng sinh viên mỗi thang điểm*** của môn ***Nhập môn học máy và khai phá dữ liệu*** và môn ***Khoa học dữ liệu*** của sinh viên trường ***X*** như sau:

![](https://i.imgur.com/VM83lnb.png)

Bạn xem đầy đủ bộ dữ liệu tại ***[đây]()***

Mình đặt ra giả thuyết rằng ***điểm ML &DM tuân theo phân phối chuẩn***, ***điểm DS tuân theo phân phối chuẩn***, ***điểm toàn bộ của cả 2 môn tuân theo phân phối chuẩn***

***Như bài viết trước***, mình đã nêu ra cách ***ước lượng điểm tham số*** và củ thể có cả ví dụ cho ***phân phối chuẩn***

***Ước lượng cho các tham số của phân phối chuẩn là:***

+ $\hat{\alpha} = \bar{X} = \frac{1}{n}\sum\limits_{i=1}^nx_i$
+ $\hat{\sigma}^2 = S^2 = \frac{1}{n}\sum\limits_{i=1}^n(x_i - \bar{X})^2$

Đây chính là giá trị khiến cho hàm mật độ khớp với dữ liệu nhất.

Ta tính các giá trị này cho riêng môn ***DM&ML***, ***Data Science*** và cho toàn bộ dữ liệu

Ta có:

+ ***ML & DM:*** $\hat{\alpha} = 5.78, \hat{\sigma} = 1.29$
+ ***Data Science:*** $\hat{\alpha} = 7.07, \hat{\sigma} = 1.26$
+ ***Cả 2 môn:*** $\hat{\alpha} = 6.46, \hat{\sigma} = 1.43$

Từ các giá trị này ta có được mô hình xác suất khớp nhất với dữ liệu

![](https://i.imgur.com/xkDldgz.png)

+ Đường màu xanh là mô hình khớp nhất với dữ liệu đối với môn ***DM&ML***. Hay nói cách khác, đường màu xanh là mô hình phân bố chuẩn được học với dữ liệu đối với môn ***DM&ML***.

+ Đường màu đỏ là mô hình phân bố chuẩn được học khớp nhất với dữ liệu đối với môn ***Data Science***

+ Đường màu tím là mô hình phân bố chuẩn được học khớp nhất với toàn tập dữ liệu (đối với ***cả 2 môn***)

***Vấn đề đặt ra:*** Nếu chỉ dùng duy nhất một mô hình trong 3 mô hình thì ta không thể mô tả đúng hết cả toàn bộ dữ liệu. Đặc biệt ***đường màu tím cũng không thể mô tả đúng cho cả 2 môn được***.

***Sở dĩ đường màu tím*** cũng chỉ cố ép cả 2 loại dữ liệu điểm cho 2 môn về cùng 1 phân phối chuẩn, trong khi đó mỗi môn lại là 1 phân phối chuẩn khác nhau. Vì vậy không thế chỉ dùng ***1 phân phối chuẩn*** để xấp xỉ toàn bộ tập dữ liệu được

Khi đó chúng ta sinh ra mô hình ***GMM - Gaussian Mixture Model***


***Giả sử*** dữ liệu của chúng ta được sinh từ $k$ phân bố chuẩn (Gaussian) và mỗi điểm dữ liệu chỉ được sinh từ một trong $k$ phân bố chuẩn. Và xét mỗi điểm dữ liệu $x$ chỉ có trường thuộc tính.

Ta có: $k$ bộ $\{ (\mu_1, \sigma_1), (\mu_2, \sigma_2), (\mu_3, \sigma_3), ..., (\mu_k, \sigma_k) \}$ của $k$ phân bố chuẩn

Gọi $Z$ là biến tuân theo phân bố đa thức có giá trị là chỉ số của phân bố khi chọn ngẫu nhiên một phân bố trong $k$ phân bố. Các giá trị của $Z$ có thể là: $[1, 2, 3, ..., k]$

Ta có hàm xác suất của $z$ là: 

\begin{equation}
p(z = i) = p_i
\end{equation}

$p_i$ là xác suất khi chọn ngẫu nhiên một điểm dữ liệu thì điểm dữ liệu đó được sinh ra bởi phân bố thứ $i$.


***$p = [p_1, p_2, ..., p_k], p_i \ge 0$ là tham số của phân bố đa thức thỏa mãn $p_1 + p_2 + .. + p_n = 1$***.

Đối với ***ví dụ trên*** của chúng ta tập các giá trị của Z là $[1, 2]$.
Hay $P(z = 1|p) = p_1 = 1 - P(z = 2|p)$

***Hàm mật độ trọng GMM như sau:***

\begin{equation}
p_1f(x|\mu_1, \sigma_1^2) + p_2f(x|\mu_2, \sigma_2^2) + ... + p_nf(x|\mu_n, \sigma_n^2)
\end{equation}

***Trong đó: $f(x|\mu_i, \sigma_i^2)$ là hàm mật độ của phân phối chuẩn quen thuộc:***

\begin{equation}
f(x|\mu_i, \sigma_i^2) = \frac{1}{\sigma_i \sqrt{2\pi}} e ^ {-\frac{(x - \mu_i)^2}{2 \sigma_i^2}}
\end{equation}

Nếu mỗi biến $x$ có nhiều trưỡng dữ liệu, khi đó phương pháp ***GMM*** được gọi là ***GMM đa biến***, khi đó hàm mật độ của ***GMM*** là:

\begin{equation}
p_1f(x|\mu_1, \Sigma_1) + p_2f(x|\mu_2, \Sigma_2) + ... + p_nf(x|\mu_n, \Sigma_n)
\end{equation}

***Trong đó: $f(x|\mu_i, \sigma_i^2)$ là hàm mật độ của phân phối chuẩn cho biến x nhiều chiều:***

\begin{equation}
f(x|\mu_i, \Sigma_i) = \frac{1}{\sqrt{det(2\pi \Sigma_i)}} e ^ {-\frac{1}{2} (x - \mu_i)^T\Sigma_i^{-1}(x-\mu_i)}
\end{equation}


Hai mô hình trên mình vừa giới thiệu có thể được sử dụng để giải quyết ***bài toán phân loại***

Vậy việc huấn luyện đồng nghĩa với việc ta đi học các tham số.

Đối với ***GMM đa biến*** thì ta đi học $k \text { bộ tham số } \{ (\mu_1, \Sigma_1), (\mu_2, \Sigma_2), ..., (\mu_k, \Sigma_k)\}$ và tham số $p = [p_1, p_2, ..., p_k]$. 

***Để biết cách học các tham số này, mời bạn đọc đến với mục tiếp theo***


## 2. Phương pháp MLE

Bạn nên đọc bài viết ***[Mẫu thống kê và vấn đề ước lượng trong xác suất]()*** trước khi tìm hiểu phương pháp ***MLE*** vì phương pháp ***MLE*** có ý tưởng giống 99.99% với cách thước ước lượng điểm tham số.

Ước lượng điểm tham số yêu cầu chúng ta phải biết trước một mẫu dữ liệu $x_1, x_2, ..., x_n$ lấy từ tập nền ***(VD: Lấy mẫu dữ liệu chiều cao của 100 người châu Á)***. Ứng với mẫu dữ liệu biết sẵn này, phương pháp ***MLE*** dùng tập ***training***.

Phương pháp ước lượng điểm ở trong bài viết ***[Mẫu thống kê và vấn đề ước lượng trong xác suất]()*** đi tìm bộ tham số $\theta$ sao cho hàm $L(x \mid \theta)$ đạt giá trị lớn nhất có thể.

Giả sử biến gốc $X$ có hàm xác suất hoặc hàm mật độ là $f(x \mid \theta)$.

\begin{equation}
P(x \mid \theta) = P(X_1=x_1, X_2=x_2, ..., X_n=x_n \mid \theta) = \prod\limits_{i=1}^nf(x_i \mid \theta)
\end{equation}

\begin{equation}
\hat{\theta} = \max\limits_{\theta}\prod\limits_{i=1}^nf(x_i \mid \theta)
\end{equation}

Hàm $P(x \mid \theta)$ còn được gọi là ***likelihood function***

Ta có biểu thức trên là do các biến củ thể trong mẫu dữ liệu ***độc lập với nhau***. Phương pháp ***MLE*** cũng giải sử ***n điểm dữ liệu*** cũng phải độc lập với nhau

Hàm $f(x_i \mid \theta)$ đối với phương pháp ***MLE*** chính là hàm phân phối

Vậy ta tổng kết phương pháp MLE như sau:
+ Phương pháp MLE giả sử các điểm dữ liệu độc lập với nhau
+ Giá trị ước lượng điểm cho các tham số của ***hàm xác suất*** hay ***hàm mật độ*** trong bài viết ***[Mẫu thống kê và vấn đề ước lượng trong xác suất]()*** chính là giá trị tham số tối ưu mà phương pháp ***MLE*** phải học được

Ví dụ:

+ Giá trị ước lượng điểm tốt nhất cho tham số $\lambda$ của phân phối ***Poisson*** là $\hat{\lambda} = \frac{\sum\limits_{i = 1}^nx_i}{n}$ chính là giá trị tham số tối ưu khi phương pháp ***MLE*** học tử dữ liệu từ tập ***training***

+ Giá trị ước lượng điểm  cho các tham số của phân phối chuẩn (các điểm dữ liệu một chiều):
    + $\hat{\alpha} = \bar{X} = \frac{1}{n}\sum\limits_{i=1}^nx_i$
    + $\hat{\sigma}^2 = S^2 = \frac{1}{n}\sum\limits_{i=1}^n(x_i - \bar{X})^2$

    Đây cũng chính là giá trị mà ***MLE*** cần phải học được
    
***Khó khăn gặp phải của phương pháp MLE:***

+ Dữ liệu phức tạp, hàm mật độ phức tạp
+ Chi phí tính toán đạo hàm lớn

Ví dụ với mô hình ***Gaussian đa biến***

+ $\mu = [\mu_1, \mu_2, ..., \mu_k]$
+ $\Sigma = [\Sigma_1, \Sigma_2, ..., \Sigma_k]$
+ $p = [p_1, p_2, ..., p_k]$

Giả sử các mẫu dữ liệu $x$ độc lập với nhau. Khi đó theo phương pháp ***MLE*** ta cần cực đại hóa hàm sau:

\begin{equation}
log(P(x_1, x_2, ..., x_n | \mu, \Sigma, p)) = \sum\limits_{i = 1}^nlog(P(x_i, \mu, \Sigma, p)) = \sum\limits_{i = 1}^n log \sum\limits_{j = 1}^kp_jf(x_j|\mu_j, \Sigma_j)
\end{equation}

Với: 

\begin{equation}
f(x_i|\mu_j, \Sigma_j) = \frac{1}{\sqrt{det(2\pi \Sigma_j)}} e ^ {-\frac{1}{2} (x_i - \mu_j)^T\Sigma_j^{-1}(x_i-\mu_j)}
\end{equation}

Hàm $log(P(x_1, x_2, ..., x_n | \mu, \Sigma, p))$ quá phức tạp.

Để giảm độ phức tạp của mô hình mình xin giới thiệu một phương pháp tên là ***EM***.

## 3. Phương pháp EM - Expectation maximization

### 3.1 Giải thích bản chất của giải thuật

Gọi $Z$ là biến tuân theo phân bố đa thức có giá trị là chỉ số của phân bố khi chọn ngẫu nhiên một phân bố trong $k$ phân bố. Các giá trị của $Z$ có thể là: $[1, 2, 3, ..., k]$


Gọi:
    + $\theta$ là bộ các tham số $(\mu, \Sigma, p)$
    + Tập dữ liệu $D = (x_1, x_2, ..., x_n)$

hay $log(P(x_1, x_2, ..., x_n | \mu, \Sigma, p)) = log(P(D \mid \theta))$

Như chúng ta đã biết hàm $P(D \mid \theta)$ là xác suất xảy các mẫu dữ liệu $D$

Thay vì trực ét tiếp cực đại hóa $log(P(D \mid \theta))$ thì phương pháp $EM$ sẽ đi tìm một cận dưới ***ít phức tạp hơn*** và ***cực đại hóa cận dưới***

Củ thể chúng ta sẽ tìm $h(\theta)$ sao cho $log(P(D \mid \theta)) >= h(\theta)$ và giải bài toán tối ưu cho $h(\theta)$


Khi một điểm dữ liêu đó chỉ có thể thuộc 1 trong $k$ phân phối chuẩn. Hay nói cách khác các sự kiện $Z = 1, Z = 2, .., Z = k$  là một ***nhóm đầy đủ***.

Ta có:

$$log(P(x \mid \theta)) = log\sum\limits_{i = 1}^kP(Z = i, x \mid \theta)$$

Theo quy tắc ***Bayes***, ta có:

$$P(Z = i, x, \theta) = P(Z = i, x \mid \theta).P(\theta)$$

$$P(Z = i, x, \theta) = P(Z = i \mid x, \theta)P(x, \theta) = P(Z = i \mid x, \theta)P(x \mid \theta)P(\theta)$$

Khi đó:

$$P(Z = i, x \mid \theta) = P(Z = i \mid x, \theta)P(x \mid \theta)$$

Hay:

$$log(P(x \mid \theta)) =  log\sum\limits_{i = 1}^kP(Z = i \mid x, \theta)P(x \mid \theta) = log E_{Z \mid x, \theta}P(x \mid \theta)$$

***Ta có bài toán phụ sau:***

$$logE[X] \ge E[log(X)]$$

$$X\text{ chỉ nhận các giá trị không âm}$$

***Chứng minh:***

Ta có:

$$n \times logE[X] = n \times log(\frac{x_1 + x_2 + ... + x_n}{n})$$

$$n \times E[log(X)] = log(x_1) + log(x_2) + ... + log(x_n)$$

Khi đó bài toán phụ tương đương với:

$$log(\frac{x_1 + x_2 + ... + x_n}{n})^n >=  log(x_1) + log(x_2) + ... + log(x_n)$$

$$\iff$$

$$(\frac{x_1 + x_2 + ... + x_n}{n})^n >= e^{(log(x_1) + log(x_2) + ... + log(x_n))}$$

$$\iff$$

$$(\frac{x_1 + x_2 + ... + x_n}{n})^n >= x_1x_2...x_n$$

***Theo bất đẳng thức Cauchy:***

$x_1 + x_2 + ... + x_n \ge n  (^n\sqrt{x_1x_2...x_n})$

$\Rightarrow (\frac{x_1 + x_2 + ... + x_n}{n})^n >= x_1x_2...x_n$

Khi đó: ***ta đã chứng minh xong bài toán phụ***

Áp dụng vào bài toán gốc ta có:

$$log(P(x \mid \theta)) = log E_{Z \mid x, \theta}P(x \mid \theta) >= E_{Z \mid x, \theta}log(P(x \mid \theta)) = \sum\limits_{i = 1}^kP(Z = i \mid x, \theta)logP(x \mid \theta)$$

Khi đó: ***cận dưới của $log(P(D \mid \theta))$*** là $\sum\limits_{i = 1}^kP(Z = i \mid D, \theta)logP(D\mid \theta)$

Áp dụng cho toán bộ tập dữ liệu $D$, $n$ điểm dữ liệu giả sử độc lập với nhau: 

\begin{equation}
log(P(D \mid \theta)) = \sum\limits_{j = 1}^nlog(P(x_j \mid \theta)) \ge  \sum\limits_{j = 1}^n\sum\limits_{i = 1}^kP(Z = i \mid x_j, \theta)logP(x_j \mid \theta)
\end{equation}


Ta có thể xây dụng mô hình ***GMM*** bằng ý tưởng gần giống với mô hình ***K-means***

|  | K-means | GMM |
| -------- | -------- | -------- |
|    Tiền điều kiện  | Số lượng nhóm k xác định trước | Số lượng phân bố chuẩn xác định trước     |
|Tham số cần được học|$k$ tâm của $k$ cụm|$\mu = [\mu_1, \mu_2, ..., \mu_k]$, $\Sigma = [\Sigma_1, \Sigma_2, ..., \Sigma_k]$, $p = [p_1, p_2, ..., p_k]$|
|Quá trình học đối với mỗi điểm dữ liệu $x$|$x$ gán vào cụm $i$ gần nhất|$x$ tương ứng với phân bố thứ $i$ sao cho xác suất xảy ra sự kiện này là cao nhất ***(1)***|
|Sau khi gán các điểm dữ liệu xong|Tính toán lại tâm các cụm|Tính toán lại các tham số của từng phân bố chuẩn ***(2)***|

***(1)*** nghĩa là: Ta tính toán xác suất phân bố thứ $i$ sinh ra điểm dữ liệu $x$ cho trước $P(Z = i|x, \mu, \Sigma, p) = P(Z = i|x, \theta)$





Ta có:

$$P(Z = i \mid x, \theta) = \frac{P(Z = i, x, \theta)}{P(x, \theta)} = \frac{P(Z = i, x, \theta)}{P(x \mid \theta)P(\theta)} = \frac{P(x \mid Z=i,  \theta)P(Z=i | \theta)P(\theta)}{P(x \mid \theta)P(\theta)} = \frac{P(x \mid Z=i,  \theta)P(Z=i | \theta)}{P(x \mid \theta)}$$

Hay:

$$P(Z = i \mid x, \theta) = \frac{f(x \mid \mu_i, \Sigma_i)P(Z=i | \theta)}{P(x \mid \theta)}$$

Ta có:

$$\sum\limits_{i=1}^kP(Z = i \mid x, \theta) = 1$$

$\Rightarrow P(x \mid \theta) = 1 / ({\sum\limits_{i=1}^kf(x \mid \mu_i, \Sigma_i)P(Z=i | \theta)})$ 

***Trong đó, $f(x \mid \mu_i, \Sigma_i)$ là hàm phân bố xác suất thứ $i$***

Ta có: $P(Z=i | \theta) = \int_{-\infty}^{+\infty}P(Z = i, x \mid \theta)dx$

Mặt khác ta đã chứng minh $P(Z = i, x \mid \theta) = P(Z = i \mid x, \theta)P(x \mid \theta)$ ở trên:

$P(Z=i | \theta) = \int_{-\infty}^{+\infty}P(Z = i, x \mid \theta)dx =  \int_{-\infty}^{+\infty}P(Z = i \mid x, \theta)P(x \mid \theta)dx = E_{x}(P(Z = i| x, \theta)) \approx \frac{1}{n}\sum\limits_{j=1}^nP(Z = i|x_j, \theta)$

$P(Z=i|x_j, \theta)$ có nghĩa là: Khi đã có $\theta$ và điểm dữ liệu $x_j$ thì xác suất điểm dữ liệu $x_j$ rơi vào phân bố chuẩn thứ $i$ là bao nhiêu. ***Đây là giá trị mà ta có thể tính được.*** ***($\text{*}$)*** Khi đó ta sẽ tính được $P(Z=i | \theta)$

Mặt khác ta cũng tính được $\sum\limits_{i=1}^kf(x \mid \mu_i, \Sigma_i)$

***Khi đó: $P(x \mid \theta) = 1 / ({\sum\limits_{i=1}^kf(x \mid \mu_i, \Sigma_i)P(Z=i | \theta)})$ xác định*** ***($\text{**}$)***

Như ta đã chứng minh:

\begin{equation}
log(P(D \mid \theta)) = \sum\limits_{j = 1}^nlog(P(x_j \mid \theta)) \ge  \sum\limits_{j = 1}^n\sum\limits_{i = 1}^kP(Z = i \mid x_j, \theta)logP(x_j \mid \theta)
\end{equation}



Do $P(x_j \mid \theta) = f(x_j \mid \theta)$ và tại giá trị $j, i$ nhât định $f(x_j \mid \theta) = f(x_j \mid \theta_i)$, $\theta_i = (\mu_i, \Sigma_i, p_i)$ là bộ tham số thứ $i$. 

Với: 
\begin{equation}
f(x_j|\mu_i, \Sigma_i) = \frac{1}{\sqrt{det(2\pi \Sigma_i)}} e ^ {-\frac{1}{2} (x_j - \mu_i)^T\Sigma_i^{-1}(x_j-\mu_i)}
\end{equation}

Giá trị cận dưới của $log(P(D \mid \theta))$ là:

\begin{equation}
P(Z = i \mid x_j, \theta) \times [-\frac{1}{2} (x_j - \mu_i)^T\Sigma_i^{-1}(x_j-\mu_i) - log \sqrt{det(2\pi \Sigma_i)}]
\end{equation}

***Đây là hàm đơn giản hơn hàm $log(P(D \mid \theta))$ rất nhiều*** và ***việc cực đại hóa đã đơn giản đi rất nhiều vì $(Z = i \mid x_j, \theta)$ tính được do giá trị $\theta$ xác định tại mỗi giai đoạn lặp*** ***(2)***

Khi đó thuật toán ***EM*** gọi bước thứ ***(1)*** là bước ***E***, còn bước thứ ***(2)*** là bước ***M***

Nói cách khác sau khi thực hiện bước ***(1)*** là tính $P(Z = i \mid x_j, \theta)$ thì ta thấy cận dưới của hàm $log(P(D \mid \theta))$ sẽ là một hàm độc lập $k$ bộ tham số $(\mu_i, \Sigma_i)$. Chúng ta có thể tìm được giá trị cuả $k$ bộ $(\mu_i, \Sigma_i)$ sao cho cực đại hóa hàm cận dưới của hàm $log(P(D \mid \theta))$. 


***Các bạn nghĩ mình sẽ không tính chỉ tiết mà chỉ ghi công thức. Nhưng không, mình sé tính từ ***A - gót chân*** cho các bạn***. Cái gì càng phức tạp thì phải rõ ràng đúng không nào ?

Sau khi thực hiện bước ***E*** thì ta có các giá trị $\omega_{ij} = P(Z = i \mid x_j, \theta), 1 \le i \le k, 1 \le j \le n$ đã ***xác định***

Khi đó hàm ***cận dưới*** trở thành:

$$LB(\mu, \Sigma) = \sum\limits_{j = 1}^n\sum\limits_{i = 1}^k\omega_{ij}  [-\frac{1}{2} (x_j - \mu_i)^T\Sigma_i^{-1}(x_j-\mu_i) - log\sqrt{det(2\pi \Sigma_i)}]$$

Việc cực đại hóa hàm ***cận dưới*** ta tính đạo hàm và tìm nghiệm:

Trước khi thực hiện ***đạo hàm*** mình sẽ cho bạn nào chưa biết một số công thức siêu dễ nhé:

***$A$ là ma trận vuông kích thức $d \times d$, $x$ là vector cột $d \times 1$***

+ $\frac{\partial f(x)^Tg(x)}{\partial x} = \frac{[\partial f(x)]g(x) + [\partial g(x)]f(x)}{\partial x}$


+ $\frac{\partial (Ax)}{x} = A^T$


Khi đó:

$$\frac{\partial (x^TAx)}{\partial x} = Ax + \frac{\partial (Ax)}{x} \times x = (A + A^T)x$$

Áp dụng tính:
+ Tính đạo hàm tại các $\mu_i, 1 \le i \le k$: 

$$\frac{\partial LB(\mu, \Sigma)}{\partial \mu_i} = \frac{1}{2} \sum\limits_{j=1}^n\omega_{ij}[\Sigma_i^{-1} + (\Sigma_i^{-1})^T](\mu_i - x_j) = 0$$

$$\Leftrightarrow$$

$$ \sum\limits_{j=1}^n\omega_{ij}(\mu_i - x_j) = 0$$

$$\Leftrightarrow$$

$$\mu_i = \frac{\sum\limits_{j=1}^n\omega_{ij}x_j}{\sum\limits_{j=1}^n\omega_{ij}}$$


+ Tính đạo hàm tại các $\Sigma_i, 1 \le i \le k$:

Để thực hiện việc tính đạo hamf cho môi $\Sigma_i$ thì mình chuẩn bị cho bạn nào chưa biết 2 công thức sau:

+ $\frac{\partial log(det(A))}{\partial A} = (A^{-1})^T$
+ $\frac{\partial x^TA^{-1}x}{\partial A} = -(A^{-1})^Txx^T(A^{-1})^T$

Khi đó:

+ $\frac{\partial((x_j - \mu_i)^T\Sigma_i^{-1}(x_j-\mu_i))}{\partial \Sigma_i} =- (\Sigma_i^{-1})^T(x_j - \mu_i)(x_j - \mu_i)^T(\Sigma_i^{-1})^T$

+ $\frac{\partial(log\sqrt{det(2\pi \Sigma_i)})}{\partial \Sigma_i}= \frac{1}{2}\frac{\partial(log\text{ det}(2\pi \Sigma_i))}{\partial \Sigma_i} = \frac{1}{2} \frac{\partial(log\text{ det}(\Sigma_i))}{\partial \Sigma_i} = \frac{1}{2} (\Sigma_i^{-1})^T$

***Chú ý: $det(2\pi A) = (2\pi)^d det(A)$, $A$ là ma trận kích thức $d \times d$***

$\Rightarrow \frac{\partial LB(\mu, \Sigma)}{\partial \Sigma_i} = \frac{1}{2} \sum\limits_{j=1}^n \omega_{ij} [(\Sigma_i^{-1})^T(x_j - \mu_i)(x_j - \mu_i)^T(\Sigma_i^{-1})^T - (\Sigma_i^{-1})^T]$

Ta có: 

$$\frac{\partial LB(\mu, \Sigma)}{\partial \Sigma_i} = 0$$


$$\Leftrightarrow$$

$$\sum\limits_{j=1}^n \omega_{ij}[(\Sigma_i^{-1})^T(x_j - \mu_i)(x_j - \mu_i)^T(\Sigma_i^{-1})^T - (\Sigma_i^{-1})^T] = 0$$

$$\Leftrightarrow$$

$$\sum\limits_{j=1}^n \omega_{ij}(\Sigma_i^{-1})^T[(x_j - \mu_i)(x_j - \mu_i)^T(\Sigma_i^{-1})^T - I_{d \times d}] = 0$$

$$\Leftrightarrow$$

$$\sum\limits_{j=1}^n \omega_{ij}(x_j - \mu_i)(x_j - \mu_i)^T(\Sigma_i^{-1})^T = \sum\limits_{j=1}^n \omega_{ij}I_{d \times d}$$

$$\Leftrightarrow$$

$$\sum\limits_{j=1}^n \omega_{ij}(x_j - \mu_i)(x_j - \mu_i)^T(\Sigma_i^{-1})^T\Sigma_i^T = \sum\limits_{j=1}^n \omega_{ij}\Sigma_i^T$$

$$\Leftrightarrow$$

$$\sum\limits_{j=1}^n \omega_{ij}(x_j - \mu_i)(x_j - \mu_i)^T(\Sigma_i \Sigma_i^{-1})^T = \sum\limits_{j=1}^n \omega_{ij}\Sigma_i^T$$

$$\Leftrightarrow$$

$$\sum\limits_{j=1}^n \omega_{ij}(x_j - \mu_i)(x_j - \mu_i)^T = \sum\limits_{j=1}^n \omega_{ij}\Sigma_i^T$$

$$\Leftrightarrow$$

$$\Sigma_i^T = \frac{\sum\limits_{j=1}^n \omega_{ij}[(x_j - \mu_i)(x_j - \mu_i)^T]}{\sum\limits_{j=1}^n \omega_{ij}}$$





***Chú ý: Ma trận hiệp phương sai $\Sigma_i$ là ma trận đối xứng và ta không lấy nghiệm $\Sigma_i^{-1} = 0$***

Vậy tại bước ***M*** của giải thuật ***EM*** thì ta cập nhật k bộ tham số $[{(\mu_1, \Sigma_1), (\mu_2, \Sigma_2), ..., (\mu_i, \Sigma_i), ..., (\mu_k, \Sigma_k)}]$ theo công thức:

+ $\mu_i = \frac{\sum\limits_{j=1}^n\omega_{ij}x_j}{\sum\limits_{j=1}^n\omega_{ij}}$

+ $\Sigma_i = \frac{\sum\limits_{j=1}^n \omega_{ij}[(x_j - \mu_i)(x_j - \mu_i)^T]}{\sum\limits_{j=1}^n \omega_{ij}}$



### 3.2 Chi tiết các bước cài đặt giải thuật EM


#### 3.2.1 Các giả thuyết để có thể áp dụng được giải thuật EM

+ Các điểm dữ liệu độc lập với nhau
+ Mỗi điểm dữ liệu chỉ được sinh ra bởi duy nhất một phân phối chuẩn
+ Có $k$ phân phối chuẩn dùng để sinh ra bộ dữ liệu 


#### 3.2.2 Đầu vào của giải thuât

+ Tập huấn luyên $D = (x_1, x_2, ..., x_n)$
+ Số nguyên dương $k$

#### 3.2.3 Đầu ra của giải thuật:

Mô hình ***GMM*** với bộ $k$ tham số được ***học***, $\theta = [{(p_1, \mu_1, \Sigma_1), (p_2, \mu_2, \Sigma_2), ..., (p_i, \mu_i, \Sigma_i), ..., (p_k, \mu_k, \Sigma_k)}]$

#### 3.2.4 Quá trình lặp của giải thuật GM

+ Bước khởi tạo ngẫu nhiên tham số $\theta$ thỏa mãn:
    + $\sum\limits_{i=1}^kp_i = 1, p_i \ge 0, 1  \le i \le k$
    + Ma trận $\Sigma_i$ đối xứng ($1 \le i \le k$)
    
+ Tại bước lặp thứ $t$:
    + Tại bước ***E - Expectation(Tính xác suất điểm dữ liệu rơi vào tất cả các phân phối thứ $1 \le i \le k$)***: 
        + Đã có $p_i^{(t)} = P(Z=i | \theta^{(t)})$
        + Đã có $P(x \mid \theta^{(t)}) = 1 / ({\sum\limits_{i=1}^kf(x \mid \mu_i^{(t)}, \Sigma_i^{(t)})P(Z=i | \theta^{(t)})})$
        + Đã có $f(x \mid \mu_i^{(t)}, \Sigma_i^{(t)}) = \frac{1}{\sqrt{det(2\pi \Sigma_i^{(t)})}} e ^ {-\frac{1}{2} (x_j - \mu_i^{(t)})^T(\Sigma_i^{(t)})^{-1}(x_j-\mu_i^{(t)})}$
        + Đã có $\omega_{ij} = P(Z = i|x_j, \theta^{(t)}) = \frac{f(x \mid \mu_i^{(t)}, \Sigma_i^{(t)})P(Z=i | \theta^{(t)})}{P(x \mid \theta^{(t)})}$
         + Tính (cập nhật) $p_i^{(t + 1)} = P(Z=i | \theta^{(t + 1)})) = \frac{1}{n}\sum\limits_{j=1}^nP(Z = i|x_j, \theta^{(t)})$



    + Tại bước ***M - Maximization(Cực đại hóa cận dưới)***:
        + Đã có $\omega_{ij} = P(Z = i|x_j, \theta^{(t)}) = \frac{f(x \mid \mu_i^{(t)}, \Sigma_i^{(t)})P(Z=i | \theta^{(t)})}{P(x \mid \theta^{(t)})}$
        + Tính (cập nhật) $\mu_i^{(t + 1)} = \frac{\sum\limits_{j=1}^n\omega_{ij}x_j}{\sum\limits_{j=1}^n\omega_{ij}}$
        + Tính (cập nhật) $\Sigma_i^{(t + 1)} = \frac{\sum\limits_{j=1}^n \omega_{ij}[(x_j - \mu_i^{(t + 1)})(x_j - \mu_i^{(t + 1)})^T]}{\sum\limits_{j=1}^n \omega_{ij}}$
        
+ Điều kiện lặp: Tiếp tục ***lặp bước tiếp theo $t + 1$*** nếu các tham số chưa hội tụ

![image alt](https://slidetodoc.com/presentation_image/39f9b8124a0152e5f7c0bd3bfbbf6718/image-41.jpg)

***Nhìn vào hình vẽ trên ta sẽ có cách nhìn tường mình và dễ hiểu hơn về giải thuật EM***

+ Bước lặp thứ $t$:
    + Tại thời điểm $t$ với giá trị bộ tham số $\theta^{(t)}$ xác định, ta xét cận dưới ***(đường màu xanh lá cây)*** của hàm log-likehood ***(đường màu xanh nước biển)*** 

    + Tại bước ***M-Maximization*** của giải thuật, cực đại hóa hàm cận dưới và cập nhật giá trị tối ưu $\theta^{(t + 1)}$ cho $\theta$`

+ Tiếp tục bước lặp thứ $t + 1$ với bộ tham số tối ưu $\theta^{(t + 1)}$ đã biết ở bước thứ $t$ cho tới khi hội tụ
## 4.Thực hành

### 4.1 Cài đặt mô hình GMM bằng thuật toán EM

#### 4.1 Khởi tạo bộ các tham số

+ ***mu_arr*** là bộ k tham số kỳ vọng của k phân bố chuẩn
+ ***Sigma_arr*** là bộ k ma trận hiệp phương sai của k phân bố chuẩn
+ ***p_arr*** là bộ k tham số với $p_i$ được định nghĩa như trong lý thuyết mình đã nêu
+ ***omega_matrix*** chính là ma trận với $\omega_{ij}$ được định nghĩa như trong lý thuyết mình đã nêu

```jsx=
def _intitialize_parameters(self):
        self.mu_arr = np.asmatrix(np.random.random((self.k, self.n)) + np.mean(self.data))
        self.Sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
        self.p_arr = np.ones(self.k) / self.k
        self.omega_matrix = np.asmatrix(np.empty((self.m, self.k)), dtype=float)
```

#### 4.2 Bước E - Expectation Step

***Các công thức trong code bạn xem trên phần lý thuyết nhé***

```jsx=
def _expectation_step(self):
        for j in range(self.m):
            sum_density = 0
            for i in range(self.k):
                multi_normal_density = st.multivariate_normal.pdf(self.data[j, :], self.mu_arr[i].A1,
                                                                  self.Sigma_arr[i]) * self.p_arr[i]
                sum_density += multi_normal_density
                # print(multi_normal_density)
                self.omega_matrix[j, i] = multi_normal_density

            const_p_x_theta = 1 / sum_density

            self.omega_matrix[j, :] = self.omega_matrix[j, :] * const_p_x_theta
            assert self.omega_matrix[j, :].sum() - 1 < 1e-4

```

#### 4.3 Bước M - Maximization Step
```jsx=
def _maximaization_step(self):
        for i in range(self.k):
            sum_omega_for_ith_norm = self.omega_matrix[:, i].sum()
            self.p_arr[i] = 1 / self.m * sum_omega_for_ith_norm
            mu = np.zeros(self.n)
            Sigma = np.zeros((self.n, self.n))

            for j in range(self.m):
                mu += self.omega_matrix[j, i] * self.data[j, :]
                dis = (self.data[j, :] - self.mu_arr[i, :]).T * (self.data[j, :] - self.mu_arr[i, :])
                Sigma += self.omega_matrix[j, i] * dis

            self.mu_arr[i] = mu / sum_omega_for_ith_norm
            self.Sigma_arr[i] = Sigma / sum_omega_for_ith_norm
```

***Phần source code về huấn luyện mô hình bạn xem ở [đây]()*** 

### 4.2 Áp dụng vào ví dụ củ thể

***Ở mục này mình sẽ chỉ nêu môtj ví dụ thôi nhé. Các ví dụ khác cá bạn tham khảo đâỳ đủ source code tại [đây]()***

Ở phần thực hành này mình muốn sử dụng lại ví dụ về ***điểm của 2 môn học MLDM và Data Science***

##### 4.2.1.2 Khai báo một số thư viện cần sử dụng

+ pandas: thư viện xử lý dữ liệu dạng bảng
+ numpy: thư viện cho đại số tuyến tính dùng để biến đổi ma trận, làm việc với vector
+ GMM: gói chứa mô hình GMM và chứa các phương thức để vẽ biểu đồ


```jsx=
import numpy as np
import pandas as pd
from GMM.GMM import *
from GMM import plot

```

##### 4.2.1.3 Đọc dữ liệu

```jsx=
mark = pd.read_csv('./ml_ds_mark.csv', sep=',', encoding='utf-8')
print("Kích thước bộ dữ liệu:", mark.shape)
mark
```

![](https://i.imgur.com/4Yy8suF.png)


##### 4.2.1.4 TIền xử lý dữ liệu

+ Chuyển thang điểm bằng cách lấy trung bình
```jsx=
def convert_scale_2_num(scale):
    a = float(scale.split("-")[0])
    b = float(scale.split("-")[1])
    return (a + b) / 2
```

###### 4.2.1.5 Huấn luyện mô hình:

```jsx=
def markClassification(k=2):
    x = np.array(mark_all).reshape((len(mark_all), 1))
    col_name="mark"
    legend = ["ML & DM", "Data Science", "All"]
    gmm = GMM(x, k)
    gmm.fit()
    plot.plot_1D(gmm, x, legend, col_name)

markClassification()
```

![](https://i.imgur.com/OzBeZ2r.png)


Như chúng ta đã thấy có vẻ mô hình của chúng ta đang khá là khớp với dữ liệu đối với cả 2 môn ***(Đường màu đỏ)*** vì nó thay vì chỉ dùng 1 phân phối chuẩn để xấp xỉ cho dữ liệu của 2 môn thì mô hình đã học ra 1 mô hình là hợp của 2 mô hình phân phối chuẩn theo 1 tỷ lệ nào đó ***(cũng đã được học)***


## 5 Bàn luận về giải thuật EM và giải thuật K-Means

+ Đối với K-means: đối với mỗi điểm sẽ được thực hiện phép gán cứng ***(Hard assignment***, nghĩa là tại mỗi thời điểm thì mỗi điểm chỉ được gán cho một cụm mà thôi
+ Đối với giải thuật EM: đối với mỗi điểm dữ liệu sẽ được thực hiện phép gán mềm ***(Soft assignment)***, nghĩa là thay thì mỗi điểm sẽ được tính xác suất nó thuộc vào cả $k$ phân bố chuẩn

![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/ClusterAnalysis_Mouse.svg/1000px-ClusterAnalysis_Mouse.svg.png)

***Hình vẽ được tham khảo từ [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)***

Nhìn vào hình vẽ ta thấy ***cả 2 phương pháp*** đều tìm được ***tâm cụm*** giống như dữ liệu ban đầu tuy nhiên:

+ Phương pháp ***K-means*** không tìm được đúng hình dạng của các cụm
+ Phương pháp ***EM*** mô hình hóa được hình dạng của các cụm dữ liệu tốt hơn


***Vì sao lại như vậy nhỉ?***

Cùng nhau xem lại công thức hàm mật độ nhé

\begin{equation}
f(x_j|\mu_i, \Sigma_i) = \frac{1}{\sqrt{det(2\pi \Sigma_i)}} e ^ {-\frac{1}{2} (x_j - \mu_i)^T\Sigma_i^{-1}(x_j-\mu_i)}
\end{equation}

Nếu các bạn đã đọc bài ***[Ma trận hiệp phương sai và những điều thú vị có thể bạn chưa biết]()*** tại chính blog của mình thì đã biết được rằng: ***ma trận hiệp phương sai có ảnh hưởng và liên quan chặt chẽ đến hình dạng của dữ liệu***. Do có điều quan trọng này nên đối với từng điểm dữ liệu thay vì hay tính các khoảng cách ***Euclid*** như trong phương pháp ***K-means*** thì phương pháp ***EM*** còn học thêm được ***k ma trận hiệp phương sai*** cho ***k phân bố chuẩn***. Vì vậy mà hình dáng của ***k cụm*** được học từ phương pháp ***EM*** sẽ mô tả tốt dữ liệu hơn phương pháp ***K-means***.

Khi phương pháp ***K-means sử dụng khoảng cách Euclid*** có nghĩa là đang sử dụng ***ma trận hiệp phương sai là ma trận đơn vị***. Bài viết ***[Ma trận hiệp phương sai và những điều thú vị có thể bạn chưa biết]()*** đã cho chúng ta biết rằng khi ***ma trận hiệp phương sai là ma trận đơn vị*** thì độ trải ra ở các chiều là giống nhau, do đó mà hình dáng dữ liệu ở các cụm nếu được huận luyện bằng phương pháp ***K-means*** sẽ có xu hướng tương tự nhau vì hình dáng.


## 6. Tài liệu tham khảo

+ [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)
+ [Probabilistic Modeling - Trần Quang Khoát - ĐHBKHN](https://users.soict.hust.edu.vn/khoattq/ml-dm-course/L11.1-prob-models.pdf)

+ [Appendix D - Matrix Calculus](https://ccrma.stanford.edu/~dattorro/matrixcalc.pdf)