# Nhật Ký Tối Ưu Hóa Wan2.1 Cho NVIDIA CMP 40HX (8GB)

Báo cáo chi tiết quá trình tinh chỉnh mã nguồn ComfyUI để khắc phục lỗi hiệu năng Float16 trên kiến trúc Turing (CMP 40HX), giúp tăng tốc độ render video từ **792s/it** xuống còn **174s/it**.

## 1. Tổng Quan Vấn Đề
Dòng card CMP 40HX (kiến trúc Turing tương tự RTX 2060/2070 nhưng dành cho đào coin) gặp hiện tượng "nghẽn cổ chai" cực nặng khi xử lý kiểu dữ liệu **Float16 (FP16)** trong AI, dẫn đến tốc độ cực chậm. Giải pháp là ép hệ thống tính toán trên **Float32 (FP32)** ở những điểm xung yếu nhưng vẫn phải bảo toàn VRAM.
## 2. Các Tệp Tin Đã Chỉnh Sửa
### 2.1. File: `GGUF Loader` (Custom Node)
**Vị trí sửa:** Hàm xử lý trọng số GGUF.
* **Nội dung sửa:**
    ```python
    def forward_ggml_cast_weights(self, input):
        orig_dtype = input.dtype            
        # Ép trọng số và input lên Float32 để kích hoạt CUDA Cores thuần túy
        weight, bias = self.cast_bias_weight(input, dtype=torch.float32)            
        input_float32 = input.to(torch.float32)            
        if bias is not None:
            bias = bias.to(torch.float32)            
        # Thực hiện phép nhân tuyến tính ở tốc độ tối đa
        result = torch.nn.functional.linear(input_float32, weight, bias)
        # Trả về kiểu dữ liệu gốc để khớp với luồng ComfyUI
        return result.to(orig_dtype)
    ```
* **Hiệu quả:** Giảm độ trễ lớp Linear từ **1067ms** xuống **158ms**.
---
### 2.2. File: `comfy/ldm/modules/attention.py`
**Vị trí sửa:** Hàm `attention_pytorch` và `attention_flash`.
Đây là cú hack quan trọng nhất để tối ưu bộ não Attention mà không làm nổ VRAM 8GB.
* **Chiến thuật:** "Ve sầu thoát xác" (Lừa hệ thống dùng vỏ FP16 nhưng ruột tính bằng FP32).
* **Nội dung sửa (Mẫu cho `attention_pytorch`):**
    ```python
    # Tại nhánh SDP_BATCH_LIMIT >= b:
    orig_dtype = q.dtype
    # Bơm steroid lên Float32
    q_f32, k_f32, v_f32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    mask_f32 = mask.to(torch.float32) if mask is not None else None    
    # Kích hoạt Memory-Efficient Attention (Chống OOM cho Float32)
    out_f32 = comfy.ops.scaled_dot_product_attention(
        q_f32, k_f32, v_f32, attn_mask=mask_f32, dropout_p=0.0, is_causal=False
    )    
    # Thu hồi lại vỏ bọc Float16
    out = out_f32.to(orig_dtype)
    ```
* **Hiệu quả:** Giảm độ trễ Attention từ **271ms** xuống **70ms**.
---
## 3. Kết Quả Thực Nghiệm
| Chỉ số | Trước tối ưu (FP16 gốc) | Sau tối ưu (FP32 Hack) | Cải thiện |
| :--- | :--- | :--- | :--- |
| **Linear Latency** | 1067 ms | 158 ms | **~6.7x** |
| **Attention Latency** | 271 ms | 70 ms | **~3.8x** |
| **Wan2.2 Speed (s/it)** | **376.00 s/it** | **174.05 s/it** | **~2.1x** |

### Tình trạng tài nguyên hệ thống (CMP 40HX 8GB):
- **VRAM Sử dụng:** ~3.03 GB (Loaded) / 8.0 GB.
- **VRAM Trống (Usable):** ~3.22 GB (Đủ an toàn cho FP32 SDPA).
- **Trạng thái:** Chạy ổn định, không lỗi OOM, tốc độ tăng gấp đôi.
- ** tự thú nhẹ:** VRAM khi dùng Float32 nó sẽ ngốn còn dư 1.6G VRAM ( điểm an toàn ) để tối ưu hệ tốc độ thêm, trade-off cũng không quá tệ khi áp dụng BF16 như dưới.
## 4. Lưu Ý Quan Trọng
1.  **Cập nhật ComfyUI:** Nếu cập nhật phiên bản mới, file `attention.py` có thể bị ghi đè. Cần kiểm tra và áp dụng lại bản hack nếu tốc độ bị tụt.
2.  **Độ phân giải:** Do dùng FP32 cho Attention (tốn gấp đôi bộ nhớ so với FP16), không nên đẩy độ phân giải video quá cao (tránh tràn 8GB VRAM).
3.  **More solution:** FP32 tốn bộ nhớ gấp đôi FP16 nhưng BF16 có cùng số bit, nhưng vì Turing không có nhân BF16 nên hệ thống mặc định đâm thẳng FP32 nhưng xuất BF16 ( có thể lợi dụng điều này).
