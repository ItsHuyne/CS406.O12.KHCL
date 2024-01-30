
<h3 align="center" font-size= 14px;><b>Trường Đại Học Công Nghệ Thông Tin - ĐHQH TPHCM</b></h3>
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>
<h1 align="center"><b>Đồ án cuối kỳ môn Máy học - CS114.O11.KHCL</b></h1>
<h2 align="center"><b>BÀI TOÁN IMAGE MATTING SỬ DỤNG MODEL U^2-NET <br>THÀNH PHỐ HỒ CHÍ MINH
 </br></h2>


### Giảng viên hướng dẫn

Họ tên | Email
--- | --- 
PGS.TS. Lê Đình Duy | duyld@uit.edu.vn
Ths. Phạm Nguyễn Trường An | truonganpn@uit.edu.vn

### Tên nhóm: LHH
### Các thành viên của nhóm
Họ tên | MSSV | Email | GitHub
--- | --- | -- | --
Hoàng Gia Huy | 19521607 | 19521607@gm.uit.edu.vn | https://github.com/ItsHuyne
</p>




# **BẢNG MỤC LỤC**
1. [Giải Trình Sau Vấn Đáp](#giaitrinh)
2. [Tổng Quan Về Đồ Án](#tongquan)
3. [Xây Dựng Bộ Dữ Liệu](#dulieu)
4. [Phương pháp, mô hình sử dụng](#method)
5. [Training Và Đánh Giá Model](#training)
6. [Hướng Phát Triển Và Cải Tiến](#ungdung)
7. [Demo mô hình](#demo)
8. [Nguồn Tham Khảo](#thamkhao)

<a name="giaitrinh"></a>
# **1. Giải Trình Sau Vấn Đáp**

[Cách thu thập dữ liệu.](#thuthap)

[Đánh giá mô hình.](#danhgia)



<a name="tongquan"></a>
# **2. Tổng Quan Về Đồ Án**

## **2.1. Ngữ cảnh ứng dụng**
  * Trong bối cảnh hiện tại, việc loại bỏ background được ứng dụng trong rất nhiều lĩnh vực như: chỉnh sửa ảnh hay video, thực tế ảo, tăng cường thực tế,… Từ những phim trường nổi tiếng, trên những bộ máy tính có hiệu suất lớn chỉ nhằm phục vụ việc tách đối tượng ra khỏi nền, đến những chiếc điện thoại smartphone trên tay chúng ta cũng đã có thể tách đối tượng ra khỏi nền của nó trong cùng một tấm ảnh. Để giải quyết vấn đề này, các phương pháp truyền thống thường đòi hỏi sự tương tác của người dùng, như vẽ một hộp giới hạn hay cung cấp “trimaps”. Các thách thức của bài toán bao gồm sự mơ hồ về màu sắc giữa phần đối tượng và phần nền, ranh giới đối tượng phức tạp và đối tượng trong suốt. Trong bài báo cáo này, nhóm sẽ giới thiệu một mô hình học máy có thể giải đáp những thách thức của bài toán Image Matting. 
  
## **2.2. INPUT và OUTPUT Bài toán**
  * INPUT: 
    * Một tấm ảnh chụp chân dung của 1 hoặc 2 người ở thế giới thực với đa dạng background, kích thước tấm ảnh đầu vào là 512x512.
 ![](https://github.com/ItsHuyne/CS114.O11.KHCL/blob/main/Image_in_Report/Input_pic.png)

  * OUTPUT: 
    * Là một ảnh mask có cùng kích thước với input, có:
      * các pixel sẽ có các giá trị từ 0-1 đại diện cho xác suất các pixel đó thuộc về foreground hay background.
      * 0 là background, 1 là foreground, các pixel có giá trị nằm trong khoảng (0; 1) sẽ mang các đặc trưng của cả foreground và background. .

      ![](https://github.com/ItsHuyne/CS114.O11.KHCL/blob/main/Image_in_Report/output_pic.png)
      


<a name="dulieu"></a>
# **3.Xây Dựng Bộ Dữ Liệu**
## **3.1. Giới thiệu về bộ dữ liệu**
* Nhóm sẽ sử dụng bộ dữ liệu bộ dữ liệu P3M-10k(Li, et al., 2021), đây là bộ dữ liệu chuẩn lớn đầu tiên được ẩn danh cho bài toán Privacy-Preserving Portrait Matting. P3M-10k bao gồm 10.000 hình ảnh chân dung mờ chất lượng cao kèm theo alpha mattes chất lượng cao
* Những hình ảnh trong bộ dữ liệu sẽ có từ một hoặc tối đa là hai người trong cùng một tấm ảnh với phần bối cảnh đa dạng. Ngoài ra, những tấm ảnh có thể được chụp từ chính diện hoặc từ đằng sau lưng.
 ![]( https://github.com/ItsHuyne/CS114.O11.KHCL/blob/main/Image_in_Report/data_1.jpg)
 ![](https://github.com/ItsHuyne/CS114.O11.KHCL/blob/main/Image_in_Report/data_2.jpg)
 ![](https://github.com/ItsHuyne/CS114.O11.KHCL/blob/main/Image_in_Report/data_3.jpg)

### **3.1.1. Thông tin thu thập dữ liệu** <a name= "thuthap"></a>


*	Bộ dữ liệu được thu thập, chọn lọc và ghi chú với tầm 10000 tấm ảnh từ nguồn Internet với bản quyền tự do.

*	Bộ dữ liệu sẽ bao gồm những tấm ảnh được làm mờ khuôn mặt do cần phải đáp ứng được yêu cầu của bài toán Privacy-Preserving Portrait Matting. Do đó, với những tấm ảnh chụp chân dung của người không phải là người nổi tiếng sẽ được làm mờ khuôn mặt. 


### **3.1.2. Cách làm mờ khuôn mặt**
  * Để làm mờ khuôn mặt, tác giả sử dụng các thuật toán phát hiện điểm mốc khuôn mặt (facial landmark detection) để tạo ra những tấm ảnh với khuôn mặt được làm mờ, cụ thể: : 
    *  i. Sử dụng thuật toán điểm mốc để lấy các điểm mốc trên khuôn mặt ( mắt, mũi, miệng,…) 
    *  ii. Tự động tạo ra mask dựa trên các điểm mốc, bao gồm toàn bộ khuôn mặt.
    *  iii. Loại bỏ phần chuyển tiếp giữa khuôn mặt và nền để chỉ giữ phần khuôn mặt rõ ràng.
    *  iv. Sử dụng Gaussian blur để làm mờ phần trong mask.
	![](https://github.com/ItsHuyne/CS114.O11.KHCL/blob/main/Image_in_Report/thu_thap.png)
  
### **3.1.3. Kết quả thu thập dữ liệu**
Sau 2 lần thu thập dữ liệu, nhóm thu thập được thêm hơn 1000 ảnh, chủ yếu là ảnh của người nổi tiếng vì nó khá phổ biến và không yêu cầu bản quyền. Tổng cộng là hơn 11,500 tấm ảnh được thu thập.

## **3.2. Xử lý dữ liệu**

### **3.2.1. Chia tập train/val**

  * Tiến hành chia bộ dữ liệu cho 2 tập train/val theo tỉ lệ 10000 cho train và 1500 cho val :
    * Train: 10000 tấm ảnh. Có đặc điểm sau: 
      *	đối với những ảnh chụp chân dung của người không phải người nổi tiếng thì sẽ được làm mờ khuôn mặt để đảm bảo tính bảo mật.
      
    * Val: 1500 ảnh.
      *	Nhóm sẽ chọn những tấm ảnh của người nổi tiếng.
      *	phần foreground sẽ chiếm khoảng 50% diện tích tấm ảnh.
      

   
###  **3.2.2. Tăng cường dữ liệu:**
  * Tiến hành tăng cường dữ liệu trên tập train. Quá trình tăng cường được thực hiện bằng thử viện Albumentation, với các kĩ thuật:

Tên kĩ thuật tăng cường | Lý do áp dụng
--- | --- 
HorizontalFlip | Các biến đổi bao gồm việc lật ngang hình ảnh
RandomBrightnessContrast | thay đổi độ sáng và độ tương phản ngẫu nhiên 
ElasticTransform | biến đổi đàn hồi
CLAHE | sử dụng phương pháp CLAHE để cân bằng histogram

<a name="model"></a>
# **4. Phương pháp và mô hình huấn luyện**
## **4.1 Giới thiệu về mô hình**
  * Chúng tôi chọn mô hình U2-Net là mô hình chính cho bài toán Image Matting. Mô hình U2-Net có cấu trúc U lồng 2 cấp độ, được công bố vào năm 2020 bởi nhóm nghiên cứu dẫn dắt bởi Xuebin Qin. Mô hình đã đạt được giải thưởng “2020 Pattern Regconition BEST PAPER AWARD” với đề tài: “U2-Net: Going deeper with nested U-structure for salient object dectection.” Mô hình U2-Net sẽ không sử dụng bất kì pretrained backbones từ những bài toán phân loại hình ảnh từ trước. Cấu trúc chữ U lồng 2 cấp độ: ở cấp độ thấp nhất, là một khối ReSidual U-Block(RSU), nó có thể trích xuất đặc trưng đa tỷ lệ mà không làm giảm độ phân giải của feature map. Ở cấp độ cao nhất, giống như cấu trúc của U-Net, mỗi giai đoạn được lấp đầy bởi khối RSU.
## **4.2 ReSidual U-Blocks(RSU)**  
  * Được lấy cảm hứng từ cấu trúc của mô hình U-Net, ReSidual U-block (RSU) để trích xuất những đặc trưng đa tỉ lệ trong cùng một giai đoạn mà không giảm chất lượng bản đồ đặc trưng. Cấu trúc của RSU-L(Cin,M,Cout ) với L là số lượng lớp của phần giải mã; Cin, Cout là kênh đầu vào và đầu ra; M là số lượng kênh đầu ra của lớp bên trong là một lớp tích chập (convolutional layers) trong khối RSU. Tổng hợp sẽ có 3 thành phần như sau:
    * Là một input convolution layer để biến cái input feature map x có (HxWxCin) thành intermediate map F1(x) với kênh của Cout. Đây là lớp tích chập đơn giản để trích xuất các features cục bộ của ảnh.
    * Là một  dạng như U-net sử dụng đối xứng cấu trúc encoder và decoder với nhau có chiều dài L là 7 mà nó sẽ lấy Intermediate feature map F1(x) là input và học để trích xuất và mã hóa nó thành muti-scale feature ( nó có thể học được cái đặc trưng từ nhiều tỷ lệ khác nhau). Các muti-scale feature thì được trích xuất xuống dần các feature maps và sẽ được mã hóa lại thành các feature maps có độ phân giải cao bằng cách lấy mẫu tăng dần, ghép nối  và tích chập với nhau. Lớp convolutional cuối cùng trong chuỗi có tốc độ giãn nở là 2, được biểu thị bằng ( d=2 ), cho phép mạng có trường tiếp nhận rộng hơn và thu được nhiều thông tin theo ngữ cảnh hơn mà không cần tăng số lượng tham số.
    * Cuối cùng là sự kết hợp của đặc trưng cục bộ được lấy từ ban đầu với đặc trưng đa tỉ lệ qua bước U-block: F1(x)+U(F1(x)).
    <p align ="middle">   
  <img src="https://github.com/ItsHuyne/CS114.O11.KHCL/blob/main/Image_in_Report/RSU.jpg" alt="drawing" width="400" height='300'/>
  <img src="https://github.com/ItsHuyne/CS114.O11.KHCL/blob/main/Image_in_Report/U_block.png" alt="drawing" width="400" height='300'/>
</p>
	##**4.3 U^2-Net structure**
 * Cấu trúc: 
 <img src="https://github.com/ItsHuyne/CS114.O11.KHCL/blob/main/Image_in_Report/U2_architect.jpg" alt="drawing" width="400" height='300'/>
 *  Cấu trúc của U2-Net có 3 phần chính : (1) 6 giai đoạn mã hóa, (2) 5 giai đoạn giải mã và (3) một module tổng hợp bản đồ saliency được gắn với các giai đoạn giải mã và giai đoạn mã hóa cuối cùng: 
    * Trong các giai đoạn mã hóa En 1, En 2, En 3 và En 4, chúng tôi sử dụng các khối U còn lại lần lượt là RSU-7, RSU-6, RSU-5 và RSU-4. Như đã đề cập trước đó, “7”, “6”, “5” và “4” biểu thị chiều cao (L) của khối RSU. Đối với các feature maps có chiều cao và chiều rộng lớn, chúng tôi sử dụng L lớn hơn để thu được nhiều thông tin tỷ lệ lớn hơn. Do đó, trong cả hai giai đoạn En 5 và En 6, RSU-4F là RSU là một phiên bản giãn nở, trong đó  thay thế các hoạt động gộp và lấy mẫu lại bằng các convolutions  giãn nở.
    * Các giai đoạn giải mã có cấu trúc tương tự như các giai đoạn mã hóa đối xứng của chúng. Trong De 5, chúng tôi cũng sử dụng phần dư U-block RSU-4F phiên bản giãn nở tương tự như phiên bản được sử dụng trong các giai đoạn mã hóa En 5 và En 6. Mỗi giai đoạn giải mã lấy sự kết hợp của các bản đồ tính năng được lấy mẫu từ giai đoạn trước đó và các bản đồ từ giai đoạn bộ mã hóa đối xứng của nó làm đầu vào.
    * Phần cuối cùng là mô-đun tổng hợp saliency map fusion module  được sử dụng để tạo ra các bản đồ xác suất nổi. trước tiên tạo ra sáu xác suất saliency  maps đầu ra bên từ các giai đoạn En 6, De 5, De 4, De 3, De 2 và De 1 bằng lớp chập 3 × 3 và hàm sigmoid. Sau đó, nó lấy mẫu các logit (đầu ra tích chập trước các hàm sigmoid) của ánh xạ độ mặn đầu ra bên với kích thước hình ảnh đầu vào và kết hợp chúng với thao tác nối, sau đó là lớp chập 1×1 và hàm sigmoid để tạo ra bản đồ xác suất độ mặn cuối cùng Sfuse.


<a name="training"></a>
# **5. Training Và Đánh Giá Model**

## **5.3. Các bước tiến hành train**
### **5.3.1. Môi trường train và đánh giá**
  * Môi trường train và đánh giá:
    * Tiến hành train trên Kaggle, Kaggle là một nền tảng trực tuyến cho cộng đồng Machine Learning (ML) và Khoa học dữ liệu. Kaggle cho phép người dùng chia sẻ, tìm kiếm các bộ dữ liệu; tìm hiểu và xây dựng models, tương tác với những nhà khoa học và kỹ sư ML trên toàn thế giới; tham gia các cuộc thi để có cơ hội chiến thắng những giải thưởng giá trị. Người dùng Kaggle sẽ được hỗ trợ Graphic Processing Unit (GPU) và gần đây có thêm Tensor Processing Unit (TPU) để tăng tốc độ tính toán trong quá trình training cũng như inference.

  ![](https://github.com/ItsHuyne/CS114.O11.KHCL/blob/main/Image_in_Report/kaggle.png)
  
   * Kaggle có một điểm tiện lợi hơn so với Google Colab là ở chỗ ta có thể save lại các version, nên ta có thể lưu trữ các file weight, cũng như những file log. Tuy nhiên, do có giới hạn về thời gian, cụ thể là 30 giờ/tuần, sẽ cập nhật lại vào thứ 7 hằng tuần nên quá trình train diễn ra thường bị ngắt quãng.

### **5.3.2. U^2-Net**
* Mô hình U2-Net có 2 version: mô hình U2-Net, và mô hình U2-Net-lite. Để tiết kiệm thời gian cũng như tài nguyên, nhóm em sẽ sử dụng phiên bản lite của mô hình U2-Net. 
  
* Quá trình training model:
  * Upload dataset lên kaggle, define mô hình U2-Net bản lite
  * Chuẩn bị lại các file cần thiết và tài nguyên để chuẩn bị cho việc training: 
    * Cài đặt các thư viện cần thiết.
    * Load dataset, file train sẽ chứa các ảnh trong tập train, file validation sẽ chứa các ảnh trong tập validate.
    * Fine-tune hàm loss, sử dụng tổ hợp 3 hàm loss: alpha_loss(hàm tính toán mean squared error giữa y_true và y_predict của alpha matte, Hàm loss này thường được sử dụng trong các tác vụ làm mờ hình ảnh trong đó mục tiêu là ước tính độ trong suốt của từng pixel trong hình ảnh), hàm ssim_loss(Hàm mất này thường được sử dụng trong các tác vụ làm mờ hình ảnh trong đó mục tiêu là ước tính độ trong suốt của từng pixel trong hình ảnh), và cả binary cross entropy loss. Lí do sử dụng tổ hợp này là vì nó giúp mô hình có thể bắt được những chi tiết nhỏ tốt hơn so với phiên bản gốc được sử dụng.
    * Tinh chỉnh các thông số của quá trình train:
        * width, height=512,512.
        * Batch=4: Xử lý 4 ảnh trong 1 vòng lặp.
        * Learning_rate = 1e-7.
        * Num_epochs=500.
        * Lưu lại kết quả tốt nhất sau 5 epochs.
        * File weights được lưu trong kaggle/output/ sau mỗi 5 iters.
        * Train.
        
Notebook:    
https://www.kaggle.com/code/huyhonggia/u2net.


## **5.4. Đánh giá mô hình** <a name="danhgia"></a>
Sau khi thực hiện quá trình training model, để xác định model của chúng ta có đủ tốt hay không cũng như đảm bảo khả năng nhận diện trong tương lai, ta cần có một phương pháp đánh giá với tiêu chí cụ thể. Đối với bài toán Image Matting, các phương pháp đánh giá thường được sử dụng là MAD, MSE, SAD, BCE,… Trong bài toán này, nhóm quyết định sử dụng 3 thang đo gồm: MAD,MSE, SAD để đánh giá độ chính xác của model. Cũng như có thể so sánh với các model khác khi huấn luyện trên dataset P3M-10k.
### **5.4.1.	Khái niệm**
#### **5.4.1.1.	MSE(Mean Squared Error).**
MSE là thang đo cho phép xác định độ lệch không khí của giá trị dự đoán và giá trị thực tế. Để tính MSE, ta tính tổng các giá trị lệch không khí của các giá trị dự đoán và giá trị thực tế, sau đó chia cho số lượng các giá trị dự đoán và giá trị thực tế.  

#### **5.4.1.2. MAD(Mean Absolute Difference) **
MAD là thang đo cho phép xác định độ lệch trung bình của giá trị dự đoán và giá trị thực tế. Để tính MAD, ta tính tổng các giá trị lệch trung bình của các giá trị dự đoán và giá trị thực tế, sau đó chia cho số lượng các giá trị dự đoán và giá trị thực tế.

#### **SAD (Sum of Absolute Differences)**
SAD là thang đo cho phép xác định độ lệch trung bình của giá trị dự đoán và giá trị thực tế. Để tính SAD, ta tính tổng các giá trị lệch trung bình của các giá trị dự đoán và giá trị thực tế.

#### **5.4.2 Đánh giá**
Sau khi huấn luyện mô hình trên 450 epochs, mỗi epoch tốn khoảng 1 tiếng hơn, giá trị các hàm loss trên tập validation không thay đổi đáng kể, còn hàm loss thì có xu hướng giảm nhẹ xung quanh 0.985. Cho rằng mô hình đã hội tụ nên nhóm dừng train, chọn ra best.weights để tiến hành đánh giá.
Trong đó:

![](https://github.com/ItsHuyne/CS114.O11.KHCL/blob/main/Image_in_Report/compare_model.png) 

Từ kết quả, ta đúc kết được rằng, mô hình chưa thật sự hiệu quả do thiếu tài nguyên huấn luyện cũng như phương pháp tiếp cận chưa tốt. Cần phải huấn luyện thêm để có thể đạt được kết quả tiệm cận với với những mô hình khác, thậm chí là tốt hơn vì đây là một mô hình đủ tốt để có thể phát triển trong tương lai


<a name="ungdung"></a>
# **6. Ứng Dụng và Hướng Phát Triển**
## **6.1. Cải tiến**

Để có thể đưa mô hình này vào ứng dụng rộng rãi, cần phải cải tiến về một số khía cạnh:

**Về data:**
*	Thu thập thêm những hình ảnh để có thể tăng cường dữ liệu. Có thể tăng cường bằng cách thu thập hình ảnh được chụp của các diễn viên, nghệ sĩ người nổi tiếng.
*	Sử dụng các thư viện tăng cường data mạnh mẽ như Albumentation để có thể thay đổi.
*	Tìm hiểu thêm các kĩ thuật tăng cường data. Việc áp dụng không hiệu quả các kĩ thuật tăng cường dữ liệu trong đề tài này càng cho thấy tuy số lượng dữ liệu quan trọng nhưng chất lượng dữ liệu cũng là một yếu tố ảnh hưởng mạnh mẽ tới độ chính xác model.
*	Huấn luyện bộ dữ liệu bằng model U^2-Net phiên bản đầy đủ và với số lượng epochs lớn hơn.
*	Thay đổi phương pháp tiếp cận cũng như thay đổi cách fine-tune mô hình để có thể cải thiện mô hình và kết quả dự đoán.

## **6.2. Hướng phát triển trong tương lai**

*	-	Cải tiến mô hình để có thể inference Real-time.
*	-	Huấn luyện mô hình với dữ liệu được tăng cường bằng cách thay đổi background
*	-	Ứng dụng nhiều hơn và các ứng dụng photoshop như là một công cụ hỗ trợ.


<a name="demo"></a>
# **7. Demo mô hình**

 [Demo.](./Demo/)

<a name="thamkhao"></a>
# **8. Tài liệu tham khảo**

[1]https://arxiv.org/abs/2005.09007

[2]https://paperswithcode.com/dataset/p3m-10k

[3]https://arxiv.org/abs/2104.14222

**Mẫu bài báo cáo**

[1]https://github.com/lphuong304/CS114.L21/blob/main/FINAL_PROJECT/Final_Report.md

