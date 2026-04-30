1. File GGUF (Trong ComfyUI Custom Nodes)
Hàm sửa: forward_ggml_cast_weights
Mục tiêu: Xử lý các lớp Linear (chiếm 70-80% khối lượng tính toán của model).
Thao tác: * Phát hiện con card này chạy Float16 cực chậm (1067ms).
Ép trọng số (weights) và dữ liệu đầu vào (input) lên Float32 ngay trước khi nhân ma trận.
Tính xong thì ép ngược kết quả về Float16 để trả lại cho hệ thống.
Kết quả: Tốc độ nhân ma trận tăng từ 1067ms lên 158ms.

2. File attention.py (File hệ thống cốt lõi của ComfyUI)
Hàm sửa: attention_pytorch và attention_flash
Đây là nơi chứa "bộ não" xử lý video của Wan2.1. Tôi đã thực hiện cú lừa ngoạn mục tại đây:
Mục tiêu: Xử lý các lớp Attention (nơi dễ gây tràn VRAM nhất).
Thao tác (Chiến thuật "Ve sầu thoát xác"):
Lưu lại kiểu dữ liệu gốc (orig_dtype).
Ép 3 ma trận Q, K, V lên Float32.
Gọi hàm scaled_dot_product_attention (SDPA) của PyTorch.
Lợi dụng việc PyTorch tự động kích hoạt Memory-Efficient Attention cho Float32 để vừa lấy tốc độ, vừa không bị nổ VRAM (OOM).
Ép kết quả về lại orig_dtype trước khi thoát hàm.
Kết quả: Tốc độ Attention tăng từ 271ms xuống 70ms.

3. File sage_attention_patch.py (File ông gửi để nghiên cứu)
Mục tiêu: Thử nghiệm Sage Attention.
Tình trạng: Chúng ta đã xem xét nhưng quyết định không dùng Sage Attention cho Float32 vì thư viện này bắt buộc dùng Float16/Int8 cấp thấp, không phù hợp với chiến thuật "Float32 thần tốc" mà tôi đang đi. SDPA Float32 hiện tại đã quá đủ nhanh rồi.
Tổng kết thành quả: ( gốc 789s/it )
Thành phần             Trước khi sửa (Float16)   Sau khi sửa (Float32 Hack)                  Mức tăng trưởng
Linear (GGUF),               1067ms               158ms                                         ~6.7 lần
Attention (SDPA),            271ms                 70ms                                         ~3.8 lần
Tổng thể (Wan2.2 14B)        376s/it              174s/it                                    Nhanh gấp 2.1 lần
