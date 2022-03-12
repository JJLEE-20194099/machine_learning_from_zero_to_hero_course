# Decision Tree và thuật toán ID3

+ "Nếu kì này tớ được học bổng, tớ sẽ quyết định tỏ tình với cậu"

+ "Nếu tớ tỏ tình với bạn Hà thành công, tớ sẽ cố gắng giàu thật sớm để bạn Hà không phải vất vả làm việc nữa"

+ "Nếu hôm nay 8/3 mà tớ không chúc ngưới người yêu của tớ thì tớ chắc chắn sẽ bị cô ấy giận dỗi 1 tuần"

Chắc hẳn trong cuộc sống của chúng ta đều phải gặp phải những câu nói ***nếu như thế này, nếu như thế nọ*** và vô tình chúng ta áp đặt 1 số quy luật nào đó vào các tình huống gặp phải:

+ "Nếu trời mưa thì tiết thể dục sẽ bị trì hoãn"
+ "Nếu trời nằng thì chắc chắn sẽ chiều nay sẽ đá bóng"

Những quy luật như vậy khiến chúng ta đưa ra quyết định ứng với từng tình huống củ thể. Và trong học máy cũng như vậy, có một mô hình đưa ra các quyết định dựa trên các điểu kiện (hay các câu hỏi) cho trước.

Cây quyết định ***(Decision Tree)*** là một mô hình như vậy

## 1. Giới thiệu

Đối với các thuật toán chúng ta học từ trước, chúng ta hay giả sử dữ liệu của chúng ta phải xấp xỉ theo một lớp hàm số cho trước, ví dụ bài toán hồi quy tuyến tính ***Linear Regression***, hàm số chúng ta giả sử có dạng là tuyến tính. 

Vậy việc giả sử như vậy chúng ta gặp phải vấn đề gì?

Một trong những vấn đề dễ thấy đó là dữ liệu quá phức tạp nhưng chúng ta chỉ xấp xỉ bằng 1 mô hình quá đơn giản. Khi đó lỗi rất cao, hay chúng ta gặp phải trường hợp ***underfitting***

Vì thế trong học máy, thay vì chúng ta xấp xỉ dữ liệu bằng 1 mô hình hàm số toán học thì chúng ta xấp xỉ dữ liệu của chúng ta bằng cấu trúc cây ***(tree)***

Kiến trúc ***cây***: bao gồm các ***node***, mỗi ***mode*** có các nốt con ứng với các nhánh.

Ứng với mỗi cây quyết định, chúng ta có thể nhìn ra ngay được các tập luật trong đó và có thể biết được rằng ***Nếu xảy ra X, thì có Y***

## 2. Ví dụ về cây quyết định

![](https://i.imgur.com/oz06pzN.png)

***Tham khảo hình chi tiết tại [đây](https://towardsdatascience.com/all-about-decision-tree-algorithm-443731371170)***
Hình trên mô tả một cây quyết định, trong đó có các yếu tố ***Outlook***, ***Windy***,... ảnh hưởng tới quyết định có nên đi chơi ***golf*** hay không.

Cây quyết định như chúng ta ta thấy ở hình vẽ trên bao gồm:

+ Nút gốc: ***Outlook***
+ Nút con: nút gốc ***Outlook*** có 3 nút con là ***Sunny***, ***Overcast***, ***Rainy***
+ Nhánh: nút gốc ***Outlook*** có 3 nút con ứng với 3 nhánh ***(Branch)***
+ Nút lá: là nút không có con, ứng với nút ***Yes*** và nút ***No***

Theo hình vẽ trên, từ cây quyết định ta có thể có được các tập luật như sau:

+ Nếu trời nắng (Outlook: Sunny) và không có gió(Windy: False) thì mình sẽ đi chơi golf
***Path: Outlook->Sunny->Windy->False->Yes***
+ Nếu trời u ám (Outlook: Overcast) thì không đi chơi golf đâu
***Path: Outlook->Overcast->No***
+ Nếu trời mưa (Outlook: Rainy) và độ ẩm bình thường (Humidity: Normal) thì vẫn đi chơi golf
***Path: Outlook->Rainy->Humidity->High->Yes***
.... 

## 3. Cây quyết định - Decision Tree được xây dựng theo những quy tắc gì như thế nào

Cây quyết định là bài toán học có giám sát ***(Supervised Learning)***. Nói cách khác, dựa trên bộ dữ liệu huấn luyện, chúng ta học ra được các tập luật bao gồm quyết định ứng với tập luật đó.

Như chúng ta thấy, dữ liệu trong thực tế rất phức tạp đồng nghĩa 1 mẫu dữ liệu quan sát trong thực tế có rất nhiều thuộc tính. Vậy nên chọn thứ tự các thuộc tính nào để thiết lập nên các tập luật cho cây quyết định. Như mình để cập ở trên ứng với mỗi ***Path*** chúng ta đã vô tình thiết lập thứ tự cho nó.

***Ví dụ:*** Path: Outlook->rainy->Humidity->High->Yes. Chúng ta chọn xét thuộc tính ***Outlook*** trước

Hơn thế nữa như chúng ta thấy, đối với thuật toán ***hồi quy*** hay ***K láng giềng gần nhất KNN***, khi thực hiện huấn luyện mô hình, bằng 1 cách nào đó ta cần phải đưa tất cả trường (thuộc tính) có giá trị không phải số về giá trị số. Đặc biệt với 1 số thuộc tính có có miền ***giá trị lớn*** ta nên chuẩn hóa lại dữ liệu.

Tuy nhiên, đối với mô hình cây quyết định ***(Decision Tree)***, nó có thể làm việc với các trường dữ liệu rời rạc (Ví dụ: ***loại*** học sinh: Giỏi, Khá, Trung Bình, Yếu) hoặc dữ liệu không có thứ tự (Ví du: ***tên***: Hà Hoàng, Long, Cường). Cây quyết định có thể làm việc với các dữ liệu có ***miền giá trị liên tục*** bằng ***phương pháp rời rạc hóa***.

Để biểu diễn hay xây dựng cây quyết định thì ta tuân theo các nguyên tắc sau:

+ Sử dụng đỉnh trong biểu diễn duy nhất 1 thuộc tính để giúp chúng ta kiểm tra dữ liệu chúng ta cần phải đoán ***(Đỉnh trong - Internal Node là đỉnh không phải nút lá)***

+ Mỗi nhánh của 1 ***node*** sẽ ứng với từng giá trị của thuộc tính được biểu diễn bởi ***node*** đó .

+ Những đỉnh không có con ***(Leaf Node)*** sẽ biểu diễn kết quả nhãn lớp của mỗi quan sát dữ liệu

+ Việc phán đoán một mẫu dữ liệu mới bằng cách sử dụng các thuộc tính của mẫu dữ liệu đó  và thược hiện các quá trình duyệt trên cây cho tới khi gặp nút lá. Khi đó giá trị nhãn tại nút lá sẽ mang ra phán đoán

Cứ 1 đường đi trên cây từ nút gốc ***(root node)*** đến nút lá ***(leaf node)***  là tập hợp của các biểu thức (Ví dụ: ***[(Outlook is Sunny) AND (Windy is False)]***), và 1 cây quyết định có nhiều đường đi.

Vấn đề chúng ta gặp phải ở đây: Chúng ta nên sử dụng ***thuộc tính*** nào ở ***node*** nào. Vấn đề này nghĩa là chúng ta cần học ra 1 cây quyết định như thế nào để có thứ tự của các ***thuộc tính***. Trong học máy, có 1 giải thuật tên là ***ID3 (Iterative Dichotomiser 3)*** giúp chúng ta có thể làm được điểu này.

## 4. Xây dựng cây quyết định bằng giải thuật ID3

Việc xây dựng cây quyết định đồng nghĩa với việc học 1 mô hình từ dữ liệu tập huấn luyện

Giải thuật ID3 giúp chúng ta lựa chọn ra ***thuộc tính*** giúp chúng ta có thể phân loại tốt được tập dữ liệu. Và tất nhiên sau khi chọn xong thuộc tính thì ta sẽ xây dựng được các nhánh con ứng với mỗi ***giá trị*** của thuộc tính đó.

Việc xây dựng cây đồng nghĩa với việc  tìm ra ***mỗi thuộc tính cho mỗi node trong cây*** và mỗi ***đường đi*** trên cây chỉ chứa các ***thuộc tính không được lặp lại***

Vậy điều kiện dừng cho thuật toán của ID3 là gì?

+ Tất cả những thuộc tính của bộ dữ liệu ***huấn luyện*** đã được sử dụng
+ Chúng ta đã xây dựng được cây quyết định mà có khả năng phân loại được hoàn toán dữ liệu trên tập luấn luyện 

***ID3*** xây dựng cây phải tuân thủ quy tắc: Khi mà chúng ta đã tính toán được ***thuộc tính*** nào nằm tại ***nút*** nào thì ***nút*** đó sẽ cố định và không được thay đổi trong quá trình học của cây

Mục tiêu tại mỗi nút, thuật toán ***ID3*** tìm ra thuộc tính giúp cây quyết định có thể phân loại tốt nhất bộ dữ liệu. Những thuộc tính như vâỵ gọi là thuộc tính có tính tác biệt ***(discriminative)***

### 4.1 Thế nào là thuộc tính có tính tách biệt

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

### 4.2 Information Gain

***Infomation Gain*** là độ đo giúp chúng ta tính toán xem thế nào là 1 ***thuộc tính có tính tách biệt*** và nó giúp ta so sánh được trong ***2 thuộc tính***, thuộc tính nào có ***độ tách biệt cao hơn***

#### 4.2.1 Entropy
Mình sẽ ôn lại một chút khái niệm về ***Entropy:***

+ Entropy mô tả mức độ hỗn loạn của 1 tập dữ liệu (lý thuyết thông tin)
+ Entropy có thể mô tả mức thông tin trong 1 đoạn văn bản, ...

+ Xét một tập ***S*** gồm các ***mẫu dữ liệu*** được phân loại vào ***c lớp***
Gọi $p_i$ là xác suất khi chọn 1 mẫu dữ liệu rơi vào lớp $i, 1 \le i \le c$. Khi đó $p_1 + p_2 + ... + p_c = 1$.
\begin{equation}
Entropy(S) = - \sum\limits_{i = 1}^cp_i \times log_2p_i
\end{equation}
***Chú ý:*** Quy định $0 \times log_20 = 0$
Entropy(S) sẽ mô tả lượng thông tin trung bình trong tập ***S***

Lấy vị dụ ở hình ảnh này mình đã đề cập ở trên:

![](https://i.imgur.com/fLp0AnH.png)
***(Hình ảnh lấy tại [bài giảng ML and DM](https://users.soict.hust.edu.vn/khoattq/ml-dm-course/L7-Random-forests.pdf) của thấy Trần Quang Khoát - ĐH BKHN)***

Khi chọn thuộc tính ***A1*** và đi theo nhánh ***v11*** dữ liệu của chúng ta bây giỡ sẽ còn lại ***30*** mẫu dữ liệu với ***21 mẫu thuộc lớp c1*** và ***9 mẫu thuộc lớp c2***

Ta có:
+ $p_{c_1}$ là xác suất khi chọn 1 mẫu dữ liệu thì rơi vào lớp $c_1$, nghĩa là: $p_{c_1} = \frac{21}{30} = 0.7$
+ Tương tư: $p_{c_2} = \frac{9}{30} = 0.3$
+ Theo ***công thức tính Entropy***, ta có:
Entropy(S) = $-p_{c_1} \times log_2(p_{c_1}) -p_{c_2} \times log_2(p_{c_2}) = -0.7 \times log_2(0.7) - 0.3 \times log_2(0.3) = 0.88$

![](https://i.imgur.com/4YYxkbc.png)

Đối với ***tập S mà các mẫu dữ liệu chỉ có thể phân loại vào 2 lớp*** thì ta có đồ thị như hình trên

+ p = 0 hoặc p = 1 đồng nghĩa là trong tập S các mẫu dữ liệu  chỉ thuộc vào 1 lớp. Hay entropy = 0 nghĩa là tập ***S*** không biểu hiện hỗn loạn của 2 lớp
+ p = 0.5 đồng nghĩa entropy lớn nhất. Mức độ hỗn loạn của tập ***S*** cao nhất, ***thông tin của tập S*** lớn nhất và chúng ta rất khó phán đoán được mẫu dữ liệu sẽ thuộc lớp nào 

Từ 2 nhận xét trên ta thấy rằng:

+ Để giá trị Entropy lớn nhất, ta sẽ làm cho tập S có ***nhiều thông tin nhất có thể***, nghĩa là xác suất 1 mẫu dữ liệu rơi vào tất cả ***c lớp*** đểu bằng nhau (Tập S hỗn loạn nhất có thể):

\begin{equation}
    p_1 = p_2 = ... = p_c
\end{equation}
  
+ Để giá trị Entropy bé nhất, ta sẽ làm cho tập S ***ít thông tin nhất có thể***, nghĩa là tất cả các mẫu dữ liệu của tập S đều ***cùng 1 phân lớp***

\begin{equation}
    \exists \text{ } 1 \le i \le c: p_i = 1
\end{equation}

#### 4.2.2 Infomation Gain là gì?

***Infomation Gain*** của 1 ***tập S*** mô tả rằng nếu chia ***tập S*** theo ***một thuộc tính*** nào đó thì ***Entropy(S) sẽ bị giảm như thế nào***

Giả sử thuộc tính chúng ta lựa chọn là $A$. Do thuộc tính $A$ sẽ có nhiều giá trị nên sẽ được phân ra thành nhiều nhánh. Ta giải sử là $K$ nhánh, ứng với các ***node con là*** $S_v, v \in Values(A), Values(A) \text{ là tập các giá trị của thuộc tính } A$

Khi đó ***Infomation Gain*** của thuộc tính $A$ là:

\begin{equation}
   G(S, A) = Entropy(S) - \sum\limits_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)
\end{equation}

#### 4.2.3 Thuật toán ID3

Thuật toán ***ID3*** khi quyết định chọn thuộc tính bằng cách cực đại hóa $G(S, A)$, nghĩa là thuộc tính nào khiến cho ***lượng Entropy bị mất nhiều nhất*** có thể, hay ***lượng Entropy còn lại ít nhất*** có thể, thì đó là thuộc tính $A^*$ tối ưu mà ***ID3*** lựa chọn:

\begin{equation}
  A^* = arg \max\limits_{A}G(S, A) = arg \min\limits_{A} \sum\limits_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)
\end{equation}

#### 4.2.4 Hiểu thuật toán ID3 một cách dễ nhất có thể

##### 4.2.4.1 Một chút xàm xí

Nếu bạn đọc đang là sinh viên và đã/đang học môn liên quan đến ***Nhập môn học máy*** thì khả năng cao ***đề thi cuối kì*** khả năng cao sẽ có bài tập liên quan đến ***thuật toán ID3*** để lựa chọn thuộc tính đặt vào các nút

Với tư cách là sinh viên ***HUST*** và đã từng học  và đạt điểm A cho môn học và đạt điểm 10 trong bảo vệ bài tập lớn môn học do thầy ***Trần Quang Khoát*** đảm nhiệm, mình tin rằng sẽ giúp các bạn hiểu rõ giải thuật này 1 cách nhanh nhất có thể!!!

##### 4.2.4.2 Ví dụ:

Giả sử ta có dữ liệu sau để ***dự đoán*** 1 người bạn trai ***có nên đi dạo hồ gươm*** với cô bạn gái vào ngày chủ nhật hay không. 



| Ngày | Outlook | Temperature | Humidity | Wind |  Đi dạo hồ gươm |
| -------- | -------- | -------- | -------- | -------- | -------- |
| N1     | Nắng     | Nóng     | Cao     | Yếu  | Không     |
| N2     | Nắng     | Nóng     | Cao     | Mạnh  | Không     |
| N3     | Âm u     | Nóng     | Cao     | Yếu  | Có|
| N4     | Mưa     | Bình thường     | Cao     | Yếu  | Có|
| N5    | Mưa     | Mát mẻ     | Bình thường     | Yếu  | Có|
| N6    | Mưa     | Mát mẻ     | Bình thường     | Mạnh  | Không|
| N7 | Âm u     | Mát mẻ     | Bình thường     | Mạnh  | Có |
| N8 | Nắng     | Bình thường     | Cao     | Yếu  | Không|
| N9 | Nắng     | Mát mẻ     | Bình thường     | Yếu  | Có|
| N10 | Mưa     | Bình thường     | Bình thường     | Yếu  | Có|
| N11 | Nắng     | Bình thường     | Bình thường     | Mạnh  | Có|
| N12 | Âm u     | Bình thường     | Cao     | Mạnh  | Có|
| N13 | Âm u     | Nóng     | Bình thường     | Yếu  | Có|
| N14 | Mưa     | Bình thường     | Cao     | Mạnh  | Không|

Chúng ta sẽ tìm thứ tự các thuộc tính và đặt vào các ***node*** bằng ***giải thuật ID3*** chúng ta vừa học ở trên như sau:

Tập dữ liệu có 14 mẫu, có 9 mẫu thuộc lớp ***có*** và 5 mẫu thuộc lớp ***không***

Khi đó:
+ $p_1$ là xác xuất khi chọn một trong 14 mẫu dữ liệu thì rơi là lớp ***có***, $p_1 = \frac{9}{14}$

+ $p_2$ là xác xuất khi chọn một trong 14 mẫu dữ liệu thì rơi là lớp ***không***, $p_1 = \frac{5}{14}$

Khi đó Entropy của tập dữ liệu ban đầu là:

$Entropy(S) = -p_1 \times log_2(p_1) - p_2 \times log_2(p_2) = -\frac{9}{14}log_2(\frac{9}{14}) -\frac{5}{14}log_2(\frac{5}{14}) \approx 0.94$

Ta thấy tập dữ liệu có ***4 thuộc tính***, đó là: ***Outlook***, ***Temperature***, ***Humidity***, ***Wind***.

Vì vậy để lựa chọn thuộc tính nào sẽ đặt vào ***node*** tiếp theo ta phải tính ***Information Gain*** của 4 thuộc tính này.

Mình tính mẫu cho thuộc tính ***Outlook***

Ta thấy thuộc tính ***Outlook*** có 3 giá trị và $Values(Outlook)  = \text{{Nắng, Âm u, Mưa}}$

Ứng với ***Outlook: Nắng***, chúng ta có 5 mẫu dữ liệu thỏa mãn điểu đó

| Ngày | Outlook | Temperature | Humidity | Wind |  Đi dạo hồ gươm |
| -------- | -------- | -------- | -------- | -------- | -------- |
| N1     | Nắng     | Nóng     | Cao     | Yếu  | Không     |
| N2     | Nắng     | Nóng     | Cao     | Mạnh  | Không     |
| N8 | Nắng     | Bình thường     | Cao     | Yếu  | Không|
| N9 | Nắng     | Mát mẻ     | Bình thường     | Yếu  | Có|
| N11 | Nắng     | Bình thường     | Bình thường     | Mạnh  | Có|

Ta tính ***Entropy của tập trên:***

$Entropy(S_{Nắng}) = -\frac{2}{5}log_2(\frac{2}{5}) -\frac{3}{5}log_2(\frac{3}{5}) \approx 0.97$

Ứng với ***Outlook: Âm u***, chúng ta có 4 mẫu dữ liệu thỏa mãn điểu đó

| Ngày | Outlook | Temperature | Humidity | Wind |  Đi dạo hồ gươm |
| -------- | -------- | -------- | -------- | -------- | -------- |
| N3     | Âm u     | Nóng     | Cao     | Yếu  | Có|
| N7 | Âm u     | Mát mẻ     | Bình thường     | Mạnh  | Có |
| N12 | Âm u     | Bình thường     | Cao     | Mạnh  | Có|
| N13 | Âm u     | Nóng     | Bình thường     | Yếu  | Có|

Ta tính ***Entropy của tập trên:***

$Entropy(S_{Âm \text{ u}}) = -\frac{4}{4}log_2(\frac{4}{4}) -\frac{0}{4}log_2(\frac{0}{4}) = 0$

Ứng với ***Outlook: Mưa***, chúng ta có 5 mẫu dữ liệu thỏa mãn điểu đó

| Ngày | Outlook | Temperature | Humidity | Wind |  Đi dạo hồ gươm |
| -------- | -------- | -------- | -------- | -------- | -------- |
| N4     | Mưa     | Bình thường     | Cao     | Yếu  | Có|
| N5    | Mưa     | Mát mẻ     | Bình thường     | Yếu  | Có|
| N6    | Mưa     | Mát mẻ     | Bình thường     | Mạnh  | Không|
| N10 | Mưa     | Bình thường     | Bình thường     | Yếu  | Có|
| N14 | Mưa     | Bình thường     | Cao     | Mạnh  | Không|

Ta tính ***Entropy của tập trên:***

$Entropy(S_{Mưa}) = -\frac{3}{5}log_2(\frac{3}{5}) -\frac{2}{5}log_2(\frac{2}{5}) \approx 0.97$

Khi đó:

***Trung bình Entropy còn lại nếu chọn thuộc tính Outlook là***

\begin{equation}\sum\limits_{v \in Values(Outlook)} \frac{|S_v|}{|S|} Entropy(S_v)
= \frac{5}{14}Entropy(S_{Nắng}) + \frac{4}{14}Entropy(S_{Âm \text{ u}}) + \frac{5}{14}Entropy(S_{Mưa})= \frac{5}{14} \times 0.97 + \frac{4}{14} \times 0+ \frac{5}{14} \times 0.97 \approx 0.62
\end{equation}


Vậy: 
$G(S, Outlook) = Entropy(S) - \sum\limits_{v \in Values(Outlook)} \frac{|S_v|}{|S|} Entropy(S_v) = 0.94 - 0.62 = 0.32$

***Tương tự:***

$G(S, Temperature) = 0.03$
$G(S, Humidity) = 0.15$
$G(S, Wind) = 0.04$
 

Ta thấy giá trị $G(S, Outlook)$ lớn nhất vì vậy ta sẽ đặt thuộc tính ***Outlook*** làm đỉnh gốc. KHi đó có 3 nhánh con là: ***Outlook: Nắng***, ***Outlook: Âm u***, ***Outlook: Mưa***


![](https://i.imgur.com/GhcfoX9.png)

Nhìn hình vẽ trên ta tiếp tính thứ tự của các thuộc tính khác ở các nhánh con của ***node cha Outlook***

Ví dụ ở nhánh con đầu tiên ứng với ***Outlook: Sunny*** thì ta  có bộ dữ liệu $S_{Nắng}$


| Ngày | Outlook | Temperature | Humidity | Wind |  Đi dạo hồ gươm |
| -------- | -------- | -------- | -------- | -------- | -------- |
| N1     | Nắng     | Nóng     | Cao     | Yếu  | Không     |
| N2     | Nắng     | Nóng     | Cao     | Mạnh  | Không     |
| N8 | Nắng     | Bình thường     | Cao     | Yếu  | Không|
| N9 | Nắng     | Mát mẻ     | Bình thường     | Yếu  | Có|
| N11 | Nắng     | Bình thường     | Bình thường     | Mạnh  | Có|

Như mình đã nói ở trên, mỗi ***PATH*** của cây quyết định chỉ chứa các thuộc tính riêng biệt, vì vậy node con ở nhánh ***Outlook: Sunny*** không thế chứa thuộc tính ***Outlook***

Ta tiếp tục tính các giá trị $G(S_{Nắng}, Temperature)$, $G(S_{Nắng}, Humidity)$, $G(S_{Nắng}, Wind)$

Ta có:
\begin{equation}
   G(S_{Nắng}, A) = Entropy(S_{Nắng}) - \sum\limits_{v \in Values(A)} \frac{|S_v|}{|S_{Nắng}|} Entropy(S_v)
\end{equation}

Dễ có:

$Entropy(S_{Nắng}) = -\frac{2}{5}log_2(\frac{2}{5}) -\frac{3}{5}log_2(\frac{3}{5}) \approx 0.97$

Thuộc tính ***Temperature*** có 3 giá trị và và $Values(Temperature)  = \text{{Nóng, Bình thường, Mát mẻ}}$ và:

+ $Entropy(S_{Nóng}) = 0$

| Ngày | Outlook | Temperature | Humidity | Wind |  Đi dạo hồ gươm |
| -------- | -------- | -------- | -------- | -------- | -------- |
| N1     | Nắng     | Nóng     | Cao     | Yếu  | Không     |
| N2     | Nắng     | Nóng     | Cao     | Mạnh  | Không     |

+ $Entropy(S_{\text{Bình thường}}) = 1$

| Ngày | Outlook | Temperature | Humidity | Wind |  Đi dạo hồ gươm |
| -------- | -------- | -------- | -------- | -------- | -------- |
| N8 | Nắng     | Bình thường     | Cao     | Yếu  | Không|
| N11 | Nắng     | Bình thường     | Bình thường     | Mạnh  | Có|

+ $Entropy(S_{\text{Mát mẻ}}) = 0$

| Ngày | Outlook | Temperature | Humidity | Wind |  Đi dạo hồ gươm |
| -------- | -------- | -------- | -------- | -------- | -------- |
| N9 | Nắng     | Mát mẻ     | Bình thường     | Yếu  | Có|

***Khi đó:***

\begin{equation}
   G(S_{Nắng}, Temperature) = Entropy(S_{Nắng}) - \sum\limits_{v \in Values(Temperature)} \frac{|S_v|}{|S_{Nắng}|} Entropy(S_v) = 0.97 - (\frac{2}{5} \times 0 + \frac{2}{5} \times 1 + \frac{1}{5} \times 0) = 0.57
\end{equation}

***Tương tự:***

$G(S_{Nắng}, Humidity) = 0.97$
$G(S_{Nắng}, Wind) = 0.02$

Ta thấy giá trị $G(S_{Nắng}, Humidity)$ lớn nhất vì vậy ta sẽ đặt thuộc tính ***Humidity*** tại node con của ***Oulook*** ứng với nhánh ***Outlook: Nắng***

(Việc tính toán các node còn lại các bạn tính toán lại cho quen tay nhé!!!)

#### 4.2.5 Vấn đề đối với mô hình cây quyết định

Nếu chúng ta quá ***tập trung vào việc phát triển cây***, và thực tế tập dữ liệu có ***số chiều khá lớn***, điều này đồng nghĩa với việc cây quyết định chúng ta tạo ra cũng ***khá lớn***. 

Nếu đánh giá 1 cách khách quan hơn, so sánh với mô hinh***Linear Regression***, do chúng ta đã quá tập trung ***tối ưu hóa hàm lỗi thực nghiệm*** dẫn đến mô hình bị ***overfit*** với dữ liệu. Điều này cũng xảy ra tương tự với cây quyết định khi chúng ta xây dụng 1 cây quyết định đầy đủ đến mức mà có thể ***fit 100%*** với tập huấn luyện , điều kiện ***trong tập dữ liệu không xác định các mẫu dữ liệu không nhất quán ví dụ: cùng giá trị các thuộc tính, nhưng mà giá trị cột quyết định lại khác nhau***

Vì vấn đề quá rõ ràng trên thì trong học máy cũng sẽ tìm cách để ***regularization*** cho cây quyết định để tránh hiện tượng ***overfitting***.

***Cắt tỉa (Prunning)*** là một tỏng những kỹ thuật như vậy

##### 4.2.5.1 Phương pháp cắt tỉa - Prunning

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

##### 4.2.5.2 Một số kỹ thuật khác

Bạn cũng có thể không phát triển cây tại 1 ***node*** nếu thỏa mãn điều kiện sau đây:

+ Nếu node có quá ít mẫu dữ liệu ta sẽ không phân tách thành các nhánh nữa và sẽ xem đó là ***node lá - leaf node*** với nhãn là lớp chiếm phần đa trong mẫu dữ liệu còn lại đó.
+ Nếu ***entropy = 0***, nghĩa là ***độ hỗn loạn của mẫu dữ liệu thấp nhất*** hay ***mẫu dữ liệu chứa rất ít thông tin*** hay ***chỉ có duy nhất 1 lớp trong mẫu dữ liệu***, ta cũng xem đây là ***node lá***
+ Khi độ giảm ***Information Gain*** không quá nhiều, ta cũng không phân chia thành các nhánh

## 5. Tổng kết ưu và nhược điểm của mô hình cây quyết định

## 5.1 Ưu điểm:

+ Dữ liệu trước khi được đưa vào mô hình không cần chuẩn hóa dữ liệu hay ***scaling*** lại dữ liệu

+ Do mô hình cây quyết định là tập hợp tất cả các luật, nên rất dễ dàng để hiểu được và giải thích được

## 5.2 Nhược điểm

+ Thời gian huấn luyện hay thời gian xây dựng cây thường khá cao đối với dữ liệu phức tạp nhiều chiều trong thực tế
+ Mô hình cây quyết định tuy làm việc tốt với các trường dữ liệu ***categorical** nhưng vẫn gặp khó khăn đối với các ***trường dữ liệu liên tục***

## 6. Tài liệu tham khảo

+ [Decison Tree - WIKI](https://vi.wikipedia.org/wiki/C%C3%A2y_quy%E1%BA%BFt_%C4%91%E1%BB%8Bnh)
+ [Decision Tree Slide- Thầy Trần Quang Khoát - ĐHBKHN](https://users.soict.hust.edu.vn/khoattq/ml-dm-course/L7-Random-forests.pdf)
+ 


