# Ma trận hiệp phương sai và những điều thú vị có thể bạn chưa biết

Trong bài viết ***[Xác suất thống kê]()***, mình đã nhắc tới khái niệm ***phương sai*** và đại lượng này đặc trưng cho độ phân tán của dữ liệu.

Ở trong bài viết này mình sẽ đề cập tới một khái niệm cực kí quan trọng và có liên quan mật thiết tới ***phương sai***. Đó chính là ***ma trận hiệp phương sai***

## 1. Thế nào là ma trận hiệp phương sai

Xét ***vector $X = [X_1, X_2, ..., X_n]$***

Với các ***vector thành phần $X_i, X_j$*** thì ***hiệp phương sai của $X_i, X_j$*** được định nghĩa là:

$\Sigma_{ij} = cov_{ij} = \sigma_{ij} =  E[(X_i - \mu_i)(X_j - \mu_j)]$ với $\mu_i = EX_i$



\begin{equation}
\Sigma = 
\begin{pmatrix}
\Sigma_{1,1} & \Sigma_{1,2} & \cdots & \Sigma_{1,n} \\
\Sigma_{2,1} & \Sigma_{2,2} & \cdots & \Sigma_{2,n} \\
\vdots  & \vdots  & \ddots & \vdots  \\
\Sigma_{n,1} & \Sigma_{n,2} & \cdots & \Sigma_{n,n} 
\end{pmatrix}
\end{equation}

## 2. Mối quan hệ giữa ma trận hiệp phương và hình dáng của dữ liệu

+ Như chúng ta đã biết ý nghĩa của phương sai là đặc trưng cho độ phân tán cuả dữ liệu.

Như chúng ta đã thấy trên thì chúng ta có thể biết được độ phân tán của bộ dữ liệu xung quanh kỳ vọng. 

+ Ý nghĩa của phương sai chỉ có thể là đặc trưng của độ phân tán dữ liệu trên các chiều song song với các trục tọa độ

Xét tập dữ liệu trông đó mỗi điểm dữ liệu có số chiều là hai tương ứng với hai trục x, y. Hình vẽ trên thể hiện một cách tổng thể dữ liệu được phân bố như thế nào trong không gian hai chiều. Nếu chiếu tất cả các điểm dữ liệu lên hai trục x, y thì dộ phân tán trên hai trục này có thể được suy ra từ giá trị phương sai cho biến X và biến Y.


### 2.1 Mối tương quan giữa hai biến

+ Mối tương quan giữa hai biến được đặc trưng bởi hệ số tương quan và hệ số tương quan nằm trong khoảng từ -1 đến 1:
    + 1: Có mối tương quan dương giữa các biến, khi biến này tăng thì biến kia cũng tăng
    + -1: Có mối tương quan âm giữa các biến, khi biến này tăng các biến kia giảm
    + 0: Không có bất cứ mỗi tương quan nào giữa các biến




+ Nhìn hình vẽ ta trên ta có thể thấy được mỗi quan hệ giưa hai biên này. Khi X tăng thì Y tăng, khi X giảm thì Y giảm. ý nghĩa này thì phương sai hoàn toàn không thể diên giải được trong khi đó ma trận hiệp phương sai có thể đặc trưng được điều này.

    + Theo công thức tính ma trân hiệp phương sai thì ta có ma trận hiệp phương sai cho tập dữ liệu hai chiều như sau:

    \begin{equation}
    \Sigma = 
    \begin{pmatrix}
    \sigma_{x,x} & \sigma_{x,y} \\
    \sigma_{y, x} & \sigma_{y,y} \\
    \end{pmatrix}
    \end{equation}

 
    + Nếu X tương quan dương với Y thì Y cũng tương quan dương với X. Nếu X tương quan âm với Y thù Y cũng tương quan âm với X. Nói cách khác thì \sigma_{x,y} = \sigma_{y, x}. Điều này có nghĩa ma trận hiệp phương sai là ma trận đối xứng. 


+ Tất cả những tính chất trên đúng với tập dữ liệu trong đó mỗi điểm dữ liệu có thể là hai chiều, ba chiều hay d chiêu.



### 2.2 Sự liên quan giữa ma trận hiệp phương sai và hình dạng của dữ liệu

Nhìn vào 4 hình vẽ trên các bạn sẽ thấy rõ rằng:

+ Các giá trị phương sai cho $X, Y$ sẽ cho chúng ta độ phân tán của dữ liệu khi chiểu trên lần lượt 2 trục $X, Y$

+ $VX$ lớn thì độ phân tán dữ liệu đối với trục $X$ lớn và $Y$ cũng tương tự

+ Hình ở góc trái trên cùng: Giá trị $\sigma_{x, y} = \sigma_{y, x} = 4 > 0$: Hai biến $X, Y$ có mối tương quan dương, khi $X$ tăng thì $Y$ tăng, khi $X$ giảm thì $Y$ giảm

+ Hình ở góc phải trên cùng: Giá trị $\sigma_{x, y} = \sigma_{y, x} = -4 <  0$: Hai biến $X, Y$ có mối tương quan âm, khi $X$ tăng thì $Y$ giảm, khi $X$ giảm thì $Y$ tăng
+ 2 hình ở phía dưới: Giá trị $\sigma_{x, y} = \sigma_{y, x} = 0$: hai biến $X, Y$ không có mối tương quan gì, $X$ tăng thì $Y có thể tăng hoặc giảm và ngược lại

Ta rút ra 1 kết luận rằng:

+ Các giá trị phương sai đặc trưng cho độ phân tán của dữ liệu trên các trục
+ Ma trận hiệp phương sai còn đặc trưng cho mối tương quan giữa các biến

### 2.3 Trị riêng - eigenvalue và vector riêng - eigenvector

Xét ma trận vuông cấp $n$

+ Số thực $\lambda$ được gọi là trị riêng của $A$ nếu phương trình $Ax = \lambda x, x \in R^n$ có nghiệm $(x_1, x_2, ..., x_n) \neq (0, 0, ..., 0)$
    + Ta có: $Ax = \lambda x \Leftrightarrow (A - \lambda I)x = 0$
    + Điều kiện cần và đủ để phương trình không có nghiệm tầm thường $x = (x_1, x_2, ..., x_n) \neq (0, 0, ..., 0)$ là $det(A - \lambda I) = 0$ 
    + $det(A - \lambda I)$ được gọi là đa thức đặc trưng cuả của $A$

+ Vector $x$ ứng với trị riêng $\lambda$ được gọi là vector riêng của ma trận $A$ ứng với trị riêng $\lambda$

## 2.4 Mối quan hệ của trị riêng và vector riêng lên hướng phân tán của dữ liệu

Các vector riêng của ma trận hiệp phương sai sẽ xác định đúng hướng phân tán dữ liệu

***Ví dụ 1:***

Xét ma trận hiệp phương sai: 

\begin{equation}
    \Sigma = 
    \begin{pmatrix}
    5 & 0 \\
    0 & 1 \\
    \end{pmatrix}
    \end{equation}

Xét đa thức đặc trưng của $\Sigma$ là $det(\Sigma - \lambda I) = (5 - \lambda)(1 - \lambda)$


Ta có: $det(\Sigma - \lambda I) = 0 \Leftrightarrow \lambda = 5 \vee \lambda = 1$

+ Với $\lambda = 5$ ta có $(\Sigma - 5I)u = 0$ với: $u = [u_1, u_2]^T$
Hay $u_2 = 0, \forall u_1 \in R$. Vậy chọn $u = [1, 0]^T$

+ Với $\lambda = 1$ ta có $(\Sigma - I)v = 0$ với: $v = [v_1, v_2]^T$
Hay $v_1 = 0, \forall v_2 \in R$. Vậy chọn $v = [0, 1]^T$

Chú ý: Ta có thể thấy được hướng phân tán của dữ liệu theo vector $u, v$

***Ví dụ 2:***

Xét ma trận hiệp phương sai: 

\begin{equation}
    \Sigma = 
    \begin{pmatrix}
    3 & 2 \\
    2 & 3 \\
    \end{pmatrix}
    \end{equation}

Xét đa thức đặc trưng của $\Sigma$ là $det(\Sigma - \lambda I) = (3 - \lambda)^2 - 4 = 0$


Ta có: $det(\Sigma - \lambda I) = 0 \Leftrightarrow \lambda = 5 \vee \lambda = 1$

+ Với $\lambda = 5$ ta có $(\Sigma - 5I)u = 0$ với: $u = [u_1, u_2]^T$
Hay $u_2 = u_1, \forall u_1 \in R$. Vậy chọn $u = [1, 1]^T$

+ Với $\lambda = 1$ ta có $(\Sigma - I)v = 0$ với: $v = [v_1, v_2]^T$
Hay $v_1 = -v_2, \forall v_2 \in R$. Vậy chọn $v = [1, -1]^T$

Chú ý: Ta có thể thấy được hướng phân tán của dữ liệu theo vector $u, v$. 
 
## 3. Tài liệu tham khảo

+ [A geometric interpretation of the covariance matrix](http://www.cs.utah.edu/~tch/CS6640F2020/resources/A%20geometric%20interpretation%20of%20the%20covariance%20matrix.pdf)

+ [Bài giảng đại số - Bùi Xuân Diệu - ĐHBKHN](https://drive.google.com/file/d/1mA2gpN_T_udNNMheAVfkkE8IeUQbJaw7/view)