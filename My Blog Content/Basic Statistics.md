# Xác suất thống kê

Một trong những câu nói hài hước mà mình rất thích đó là: ***"Vì tao chắc chắn là trên đời này không có gì là chắc chắn..."***.

Bạn đang sống trong một thời đại ***biến thiên vạn hóa***, hay nói cách khác, nhiều thứ trong cuộc sống luôn mang những tính không chắc chắn trong đó.

Việc này cũng liên quan đến rất nhiều vấn đề trong học máy:
+ Mô hình được lựa chọn có phù hợp với dữ liệu. ***Đây là 1 điều không chắc chắn***. Có thể rất tuyệt với nếu dùng các mô hình ***tuyến tính*** để xấp xỉ ***dữ liệu có mang tính tuyến tính***. Hay sẽ thật tệ nếu dùng chính mô hình  ***tuyến tính*** để đi xấp xỉ dữ liệu ***phi tuyến***. 

+ Dữ liệu được thu thập có tin cậy và chắc chắn hay không

+ Chúng ta đã biết đối với một số mô hình chúng ta phải ***dùng một số giả thuyết***. Đây cũng là một điều không chắc chắn.

Thay vì cố tính ***lánh mặt nhau*** sao không đối diện sự thật nhỉ?

Như chúng ta đã biết ***nhắc đến sự không chắc chắn*** thì không thể không nhắc đến 1 lĩnh vực nổi tiếng, đó là ***lình vực xác suất*** với những ***lý thuyết hay ho***. 

Việc áp dụng lý thuyết xác suất vào học máy đã sinh ra các mô hình ***xác suất***

## 1. Kiến thức xác suất cơ bản

### 1.1 Thuật ngữ phải nắm chắc

Sau đây là 1 chút lý thuyết xác suất giúp những ai chưa biết có thể kiếm được ***8 điểm đề thi giữa kì*** và ***6 điểm đề thi cuối kì*** môn ***xác suất thống kê*** ở bách khoa nhé. Còn những ai biết rồi hoặc không học ở "trường bách khoa yêu dấu" ***(trong dấu ngoặc kép nhé)*** thì cũng đọc lại cho ***tui*** đỡ buốn nhé.

+ Sự kiện: sự việc, hiện tượng nào đó tong cuộc sống
    + Sự kiện được gọi là tất yếu nếu nó chắc chắn xảy ra
    + Sự kiện được gọi là bất khả nếu nó không thể xảy ra
    + Sự kiện ngẫu nhiên là có thể xảy ra hoặc không

+ Phép thử ngẫu nhiên: là một loạt quan sát và thực hiện với một thí nghiệm nào đó mà kết quả chúng ta không thể đoán được
+ Tập hợp các kết cục có thể có của một phép thử là không gian mẫu, kí hiệu $\Omega$

Ví dụ: khi gieo một con xúc xắc

+ gieo một con xúc xắc: phép thử
+ Xuất hiện bảy chấm là sự kiện ***bất khả***, số xuất hiện trên xúc xắc $\ge 1$ là sự kiện ***tất yếu***, số xuất hiện trên xúc xắc là 3 là sự kiên ***ngẫu nhiên***
+ $A_i$ là sự kiến xuất hiện mặt $i$ trên xúc xắc 
$\Omega = \{ A_1, A_2, A_3, A_4, A_5, A_6 \}$ là ***không gian mẫu***

+ Bạn muốn biết sự kiên $A_i$ xuất hiên với tần xuất như thế nào, bạn đang nói đến ***xác suất***

### 1.1.2 Phép toán và quan hệ của các sự kiện

+ $A+B$ là sự kiện xuất hiện khi xuất hiện ít nhất một trong 2 sự kiên
+ $AB$ là sự kiện xuất hiện khi đổng thời 2 sự kiện đều xuất hiện
+ $\bar{A}$ là sự kiện không xuất hiện $A$
+ Xung khắc: $A_1, A_2, ..., A_n$ được gọi là xung khắc nêu $A_iA_j$ là sự kiện không thể xảy ra, $\forall i \neq j$
+ $A - B$ là sự kiện xuất hiện $A$ nhưng không xuất hiện $B$, $A - B = A\bar{B}$
+ Kéo theo: $A \Rightarrow B$, chỉ nếu xuất hiện $A$ thì xuất hiện $B$
+ Đầy đủ: $A_1, A_2, ..., A_n$ được gọi là đầy đủ nếu $A_1 + A_2 + ... + A_n = \Omega$

***Một số công thức:***
$U$ là tập vũ trụ, $V$ là tập rỗng ($\oslash$), $A, B$ là hai sự kiện con của $U$

+ $A+B = B+A, AB = BA, \text{ (giao hoán)}$
+ $A + (B + C) = (A + B) + C, A(BC) = (AB)C, \text{ (kết hợp)}$
+ $A(B + C) = AB + AC, \text{ (phân phối)}$
+ $A+U = U, A+V = a, A+A = A$
+ $AU = A, AV = V, AA = A$

Công thức mở rộng:

Xét n sự kiện: $A_1, A_2, ..., A_n$

\begin{equation}
P(\sum\limits_{i = 1}^nA_i) = \sum\limits_{i = 1}^nP(A_i) - \sum\limits_{i < j}P(A_iA_j) + \sum\limits_{i < j < k}A_iA_jA_k - ... + (-1)^{n - 1}P(A_1A_2...A_n)
\end{equation}

Nếu $A_i \text{ và }A_j$ xung khắc $\forall i, j$ thì:

\begin{equation}
P(\sum\limits_{i = 1}^nA_i) = \sum\limits_{i = 1}^nP(A_i)
\end{equation}


### 1.1.3 Xác suất là gì
Xét phép thử với n kết cục đồng khả năng. Tập n kết cục đó là không gian mẫu $\Omega = \{ A_1, A_2, ..., A_n \}$ Có m kết cục mà sự kiện $A_i$ thỏa mãn

$P(A_i) = \frac{m}{n}$

Tính chất:
+ $0 \le P(A) \le 1$
+ $P(A_1 +A_2 + ... + A_n) = 1$
+ $P(V) = 0$
+ $P(A + B) = P(A) + P(B) - P(AB)$
+ $A, B$ xung khắc hay $P(AB) = 0$ nên $P(A + B) = P(A) + P(B)$
+ $P(\bar{A}) = 1 - P(A)$
+ Nếu $A \Rightarrow B$ thì $P(A) \le P(B)$

***Xác suất có điều kiện***

Giả sử trong 1 phép thử ta có $P(B) > 0$. 
Xác suất xuất hiện có điều kiện của sự kiện $A$ khi đã có sự kiên $B$ là:
\begin{equation}
    P(A|B) = \frac{P(AB)}{P(B)}
\end{equation}

Khi đó, ta có:

\begin{equation}
P(AB) = P(A|B)P(B) = P(B|A)P(A)
\end{equation}

$A, B$ được gọi là ***độc lập***, nghĩa là sự xuất hiện của sự kiện $B$ xảy ra không liên quan gì đến sự xuất hiện của sự kiện $A$ cả và ngược lại:

\begin{equation}
P(A|B) = P(A) \text{ hoặc } P(B|A) = P(B)
\end{equation}

Khi đó:
\begin{equation}
P(AB) = P(A)P(B)
\end{equation}

***Lý thuyết cũng chán, sang ví dụ tý cho mới mẻ***

***Ví dụ 1:*** Rút ***lần lượt 2 con bài*** từ bộ bài tú lơ khơ ***52 con***. Tính:
+ Xác suất ***con thứ 1 là át***.
+ Xác suất ***con thứ 2 là át***, biết ***con thứ 1 đã là át***.

Gọi $A_i$ là sự kiện con thứ $i$ là át

+ $P(A_1) = \frac{4}{52} = \frac{1}{13}$

+ $P(A_2|A_1) = \frac{3}{51}$

***Ví dụ 2:*** Ba bạn học sinh bách khoa đi tán gái với xác suất tỏ tình thành công với bạn ***H.H*** cửa từng người tương ứng là 0.7; 0.8 và 0.9. Giả sử bạn ***H.H*** có thể ***"bắt cá nhiều tay"***

Tính xác suất:
+ a. Có đúng 2 bạn tỏ tình thành công với bạn ***H.H***
+ b. Có ít nhất 1 bạn bị bạn ***H.H*** từ chối

Gọi $A_i$ là xác suất bạn thứ $i$ tỏ tình thành công với bạn ***H.H***

$P(A_1) = 0.7, P(A_2) = 0.8, P(A_3) = 0.9$

a. Gọi A là sự kiện có hai bạn tỏ tình thành công với bạn ***H.H***, khi đó:

\begin{equation}
A = A_1A_2\bar{A_3} + A_1\bar{A_2}A_3 + \bar{A_1}A_2A_3
\end{equation}

Ta có: do $A_i$ và $\bar{A_i}$ xung khắc $\forall 1 \le i \le 3$ nên:

$A_1A_2\bar{A_3}$ và $A_1\bar{A_2}A_3$ và  $\bar{A_1}A_2A_3$ cũng xung khắc với nhau:

Theo công thức mở rộng tính tổng trên ta có:

$P(A) = P(A_1A_2\bar{A_3} + A_1\bar{A_2}A_3 + \bar{A_1}A_2A_3) = P(A_1A_2\bar{A_3}) + P(A_1\bar{A_2}A_3) + P(\bar{A_1}A_2A_3)$

Do $A_1, A_2, A_3, \bar{A_1}, \bar{A_2}, \bar{A_3}$ đôi một độc lập với nhau nên:

+ $P(A_1A_2\bar{A_3}) = 0.7 \times 0.8 \times (1-0.9)$

+ $P(A_1\bar{A_2}A_3) = 0.7 \times (1-0.8) \times 0.9$

+ $P(\bar{A_1}A_2A_3) = (1-0.7) \times 0.8 \times 0.9$

$\Rightarrow P(A) = 0.398$

### 1.1.4 Công thức Bayes
#### 1.1.4.1 Nhóm đầy đủ

Nhóm các sự kiện $A_1, A_2,..., A_n$, $n \ge 2$ tạo thành một nhóm đầy đủ nếu:

+ $A_i$ và $A_j$ xung khắc, $\forall i, j$
+ $A_1 + A_2 + ... + A_n = U$

Ví dụ: 
+ Xét phép thử khi gieo 1 con xúc xắc. $A_i$ là sự kiện xuất hiện mặt thứ $i, 1 \le n \le 6$. Nhóm $\{A_1, A_2,..., A_6\}$ là nhóm đầy đủ
+ $\{A, \bar{A} \}$ cũng là nhóm đầy đủ

Giả sử xét sự kiện $H$ nào đó.

Do nhóm $\{A_1, A_2,..., A_n\}$ là nhóm đầy đủ nên:

\begin{equation}
A_1 + A_2 + ... + A_n = U
\end{equation}

Khi đó:

\begin{equation}
H = HU = H(A_1 + A_2 + ... + A_n) = HA_1 + HA_2 + .. + HA_n
\end{equation}

\begin{equation}
P(H) = P(HA_1 + HA_2 + .. + HA_n)
\end{equation}

Do $A_i$ và $A_j$ xung khắc, $\forall i, j$ nên:

\begin{equation}
P(H) = P(H|A_1)P(A_1) + P(H|A_1)P(A_1) + ... + P(H|A_1)P(A_1) = \sum\limits_{i = 1}^n P(H|A_i)P(A_i)
\end{equation}


#### 1.1.4.2 Công thức Bayes

Xét nhóm đầy đủ $\{A_1, A_2,..., A_n\}$. Ta muốn xác định xác suất $P(A_i|H), 1 \le i \le n$

Ta có:

\begin{equation}
P(A_iH) = P(A_i)P(H|A_i) = P(A_i|H)P(H)
\end{equation}

Khi đó:

\begin{equation}
P(A_i|H) = \frac{P(H|A_i)P(A_i)}{P(H)} = \frac{P(H|A_i)P(A_i)}{\sum\limits_{i = 1}^n P(H|A_i)P(A_i)}
\end{equation}

***Công thức trên là công thức Bayes***

Công thức ***Bayes*** cho 2 sự kiện $A, B$

\begin{equation}
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
\end{equation}

***Có một số thuật ngữ khá đặc biệt sau:***
+ $P(A)$ còn được gọi là ***Prior*** (Xác suất tiên nghiệm)
+ $P(A)$ còn được gọi là ***Evidence***
+ $P(B|A)$ còn được gọi là ***Likelihood***
+ $P(A|B)$ còn được gọi là ***Posterior*** (Xác suất hậu nghiệm)

## 1.2. Biến ngẫu nhiên và một số phân bố xác suất cơ bản
### 1.2.1 Biến ngẫu nhiên

Biến ngẫu nhiên là khái niệm phụ thuộc vào kết cục của 1 phép thử ngẫu nhiên nào đó, được sinh ra giúp dễ dàng làm việc vơi các sự kiện ngẫu nhiên

Ví dụ: Gieo một con xúc xắc thì ***số chấm xuất hiện*** là một biến ngẫu nhiên

+ Biến ngẫu nhiên được gọi là rời rạc ***(discrete)*** nếu tập giá trị của nó hữu hạn hoặc là tập giá trị vô hạn nhưng ***đếm được số lượng phần tử***. Ví du: Số chấm trên xúc xắc, số cuộc gọi đến tổng đài trong 1 đơn vị thời gian. Miền giá tị của biến ***rời rạc*** là một dãy số $x_1, x_2, ..., x_n, ...$

+ Biến ngẫu nhiên được gọi là liên tục ***(continuos)*** nếu miien giá trị của nó là một đoạn $[a, b] \subset R$ hoặc $(-\infty, +\infty)$. Ví dụ: Chiều cao, cân nặng

### 1.2.2 Luật phân phối xác suất

Đối với biến rời rạc $X$ thì mỗi giá trị của nó tương ứng với với xác suất đặ trưng cho khả năng biến ngẫu nhiên nhận giá trị đó $p_i = P(X=x_i)$

***Ví dụ:*** Xét biến ngẫu nhiên $X$ là số chấm xuất hiện trên con xúc xắc. Ta có bảng phân phối xác suất cho biến $X$ là:


| $X = x$ |  1 |  2 | 3 |  4 | 5 | 6 |
| -------- | -------- | -------- |-------- |-------- |-------- |-------- |
| $p(X=x)$      | $\frac{1}{6}$     | $\frac{1}{6}$     |$\frac{1}{6}$     |$\frac{1}{6}$     |$\frac{1}{6}$     |$\frac{1}{6}$     |

***Chú ý:*** $p(x) = 0, \forall x \notin \{1, 2, 3, 4, 5, 6\}$


Hàm $p(x) = P(X = x), x \in$ tập giá trị của $X$ được gọi là hàm xác suất của biến $X$:
+ $p(x) \ge 0, \forall x$
+ $\sum\limits_{\text{mọi }x}p(x) = 1$

Bạn có thể lập được một bảng phân phối cho một biễn rời rạc có quan hệ đã biết với các biến rời rạc khác. ***Ví dụ:***

Cho 2 biến rời rạc $X$ và $Y$ có bảng phân phối

| $X=x$ | -1 | 0 | 1 |
| -------- | -------- | -------- |-------- |
| $p(X=x)$     | 0.3     | 0.4     |0.3|

| $Y=y$ | 1 | 2 |
| -------- | -------- | -------- |
| $p(Y=y)$     | 0.3     | 0.7     |

Lập bảng phân phới cho biến: $Z = X+Y$

Ta có: $Z = X+Y$ nên $Z$ có thể nhận các giá trị sau: 0, 1, 2, 3.

Ta có:
\begin{equation}
P(Z=z_i) = P(X+Y=z_i) = \sum\limits_{x_m + y_n = z_i}P(X=x_m; Y=y_n)
\end{equation}

| $X=x$ | 0 | 1 | 2 | 3 |
| -------- | -------- | -------- |-------- |-------- |
| $p(X=x)$     | 0.09     | 0.33     |0.37|0.21|

***Hàm phân phối xác suất $F(X)$*** của $X$ được định nghĩa là:

\begin{equation}
F(x) = P(X < x) = \sum\limits_{x_i < x} P(x_i), x \in R
\end{equation}

***$F(x)$ mô tả tính tập trung xác suất về phía bên trái của số thực $x$***. Đây là điều mà ***bảng phân phối xác suất*** không làm được. 

Một số tính chất chất của hàm:

+ $0 \le F(x) \le 1$
+ Nếu $x < y$ thì $F(x) \le F(y)$
+ $F(x_1 \le X \le x_2) = F(x_2) - F(x_1)$
+ $F(+\infty) = 1; F(-\infty) = 0$

***Bảng phân phối xác suất*** không thể biết rõ phân phối xác suất tại 1 lân cận củ thể nào đó, vì vậy đối với biến ngẫu nhiên liên tục và $F(x)$ khả vi khái niêm hàm ***mật độ xác suất $f(x)$*** của $X$ là:

\begin{equation}
f(x) = F'(x)
\end{equation}

Hay:

\begin{equation}
F(x) = \int_{-\infty}^xf(t)dt
\end{equation}

Ta có: $P(x_1 \le X < x_2) = \int_{x_1}^{x_2}f(x)dx$

+ $f(x) \ge 0$
+ Do $F(-\infty) = 0$ và $F(+\infty) = 0$ nên: $\int_{-\infty}^{+\infty}f(x) = 1$

### 1.2.3 Một số đặc trưng của biến ngẫu nhiên không biết là không được

+ ***Kì vọng của 1 biến ngẫu nhiên X, kí hiệu là $EX$***. Kỳ vọng còn được gọi là ***trị trung bình*** của $X$. Khác với ***giá trị trung bình*** là trung bình cộng của các giá trị. Trong thực tế chúng ta đo $X$ nhiều lần và lấy giá trị trung bình cộng, khi số lần đó càng lớn thì giá trị trung bình càng gần $EX$
    + $X$ là biến rời rạc với hàm xác suất $p(x)$:
    $EX = \sum\limits_{x_i}x_ip(X=x_i)$ 

    + $X$ là biến liên tục với hàm mật độ $f(x)$: 
    $EX = \int_{-\infty}^{+\infty}xf(x)dx$

    + Một số tính chất:
        + $E(c) = c$, $c$ là hằng số
        + $E(cX) = c \times EX$, $c$ là hằng số
        + $E(X + Y) = EX + EY$
        + Nếu $X, Y$ là hai biến độc lập thì $E(XY) = EX \times EY$
        + Nếu $Y = g(X)$ thì:
            + $EY = \sum\limits_{x_i}g(x_i)p(X=x_i)$, nếu $X$ là biên rời rạc
            + $EY = \int_{-\infty}^{+\infty}g(x)f(x)dx$, nếu $X$ là biến liên tục

+ ***Phương sai của 1 biến ngẫu nhiên X, kí hiệu là $VX$. Phương sai đặc trưng cho độ phân tán của dữ liệu*** . $VX = E[(X - EX)^2] = E(X^2) - (EX)^2$
    + $X$ là biến rời rạc với hàm xác suất $p(x)$:
    $VX = \sum\limits_{x_i}(x_i - EX)^2p(X=x_i)$ 

    + $X$ là biến liên tục với hàm mật độ $f(x)$: 
    $EX = \int_{-\infty}^{+\infty}(x - EX)^2f(x)dx$
    + Một số tính chất:
        + $Vc = 0$, $c$ là hằng số
        + $V(cX) = c^2VX$
        + Nếu $X, Y$ độc lập: $V(X + Y) = VX + VY$

+ ***Độ lệch chuẩn***: $\sigma(X) = \sqrt{VX}$

+ ***Phân vị k% của $X$:*** là một là giá trị t của $X$ sao cho $F(t) = k/100$
+ ***Trung vị*** là một trường hợp đặc biệt của ***phân vị*** với ***k = 50***, kí hiệu là $medX$

## 1.3 Một số phân phối bạn hay gặp

