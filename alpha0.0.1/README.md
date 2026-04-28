#AIO-GGUF Backend (Version 5.0 Full)
### **High-Performance GGUF Integration for Diffusers & Wan 2.2**

<p align="center">
  <img src="https://img.shields.io/badge/Hardware-GPU%20CMP%2040HX%208GB%20(PCIe%201.1%20x16)-orange">
  <img src="https://img.shields.io/badge/CUDA-11.8%20(Driver%20460.89%20CU112)-blue">
  <img src="https://img.shields.io/badge/Backend-GGUF%20Native-green">
</p>

---

### 🛡️ Thông tin dự án
> *This project is a personal initiative. I am open to further discussions and collaborations. Please contact me directly for technical details and the latest updates.*

<p align="center">
  <b>Contribution & Copyright © Solely owned by Tran Hieu Nghia</b><br>
  <b>Đóng góp & Bản quyền thuộc về duy nhất cá nhân Trần Hiếu Nghĩa</b><br><br>
  <b>Ý tưởng & Kiến trúc:</b> Trần Hiếu Nghĩa (Nickname: Timmyo/V)<br>
  <b>Concept & Architecture:</b> Tran Hieu Nghia (Nickname: Timmyo/V)<br>
  <b>Thực thi:</b> Hỗ trợ bởi hệ thống AI (Gemini 3.1 Pro, Claude 4.6, ChatGPT, Grok)<br>
  <b>Implementation:</b> Assisted by AI Systems (Gemini 3.1 Pro, Claude 4.6, ChatGPT, Grok)<br>
  <b>Cảm hứng:</b> Dựa trên các dự án mã nguồn mở của <i>ggml, llama.cpp-Python, sd.cpp-Python, ComfyUI-GGUF, Diffusers</i>
</p>

---

### 📖 Giới thiệu (Overview)
**AIO-GGUF Backend V5** là một giải pháp toàn diện được thiết kế để chạy các mô hình AI khổng lồ (như **Wan 2.2 14B**, **Stable Diffusion XL**) trên phần cứng có tài nguyên hạn chế. 

Dự án đã được thực hiện thành công trên phần cứng **GPU CMP 40HX 8GB (đã mod PCIe 1.1 x16)**, sử dụng môi trường **CUDA 11.8** với backend từ Driver **460.89 (CU 11.2)**.
Hệ thống này phá vỡ rào cản bộ nhớ bằng cách thay thế các lớp `nn.Linear` tiêu chuẩn của HuggingFace Diffusers bằng các lớp `GGUFLinear` tùy chỉnh. Điều này cho phép thực thi trực tiếp các trọng số đã được nén (Quantized) từ file GGUF mà không cần giải nén toàn bộ vào VRAM.

---

### Tính năng nổi bật (Key Features) ### 
* **Hybrid Execution Bridge:** Tự động chuyển đổi giữa nhân C++ (qua `sd.cpp-python` DLL) để đạt tốc độ tối đa và Python fallback để đảm bảo tính tương thích tuyệt đối.
* **Wan 2.2 Native Support:** Xử lý hoàn hảo cấu trúc Tensor 5D đặc thù của mô hình Video Wan 2.2.
* **CRITICAL-1 (Alias Mapping):** Tự động ánh xạ các layer alias (`self_attn` ↔ `attn1`) giúp tương thích với mọi bản checkpoint trên Civitai hoặc HuggingFace.
* **CRITICAL-2 (QKV Split Logic):** Thuật toán phân tách khối QKV Packed dựa trên thuộc tính transpose thời gian thực, đảm bảo độ chính xác tuyệt đối của Attention.
* **Memory Efficiency:** Sử dụng `mmap` để ánh xạ file từ ổ cứng, giảm thiểu tối đa dung lượng System RAM và VRAM tiêu thụ.
* **Built-in Self-Test:** Tích hợp bộ suite kiểm thử tự động, đảm bảo mọi layer đều được patch chính xác trước khi tiến hành Inference.

---

###  Kiến trúc hệ thống (Architecture) ### 
* **GGMLTensor:** Lớp con của `torch.Tensor` đóng vai trò lưu trữ siêu dữ liệu nén.
* **GGUFTensorRegistry:** Quản lý kho lưu trữ trọng số tập trung, ngăn chặn triệt để tình trạng rò rỉ bộ nhớ (memory leaks).
* **GGUFLinear:** Module thay thế `nn.Linear`, thực hiện *Dequantize-on-the-fly* hoặc gọi trực tiếp qua DLL Bridge.
* **KeyMapper:** Công cụ ánh xạ tên layer thông minh giữa định dạng GGUF và kiến trúc Diffusers.

---

###  Hướng dẫn sử dụng (Usage) ### 

**1. Cài đặt môi trường:**
```bash
pip install torch diffusers transformers sentencepiece

# Đảm bảo đã cài đặt stable-diffusion-cpp-python hoặc llama.cpp-python để có hiệu năng tốt nhất
pip install stable-diffusion-cpp-python
```

**2. Chạy kiểm tra hệ thống (System Check):**
Để đảm bảo backend hoạt động hoàn hảo trên máy của bạn, hãy chạy thử script:
```bash
py -3.10 gguf_backend_v5_full.py
```

**3. Tích hợp vào Pipeline Python:**
```python
from gguf_backend_v5_full import build_unet_gguf_native

unet = build_unet_gguf_native(
    model_path="models/wan2.2_14B_Q4_K_M.gguf",
    device="cuda"
)

# Sau đó đưa biến 'unet' vào Diffusers Pipeline như bình thường
```

---

###  Bảng hiệu suất (Performance & Memory Usage) ### 

| Model | Format | VRAM / RAM / Filepage Usages | Status | Time from Start to Output |
| :--- | :--- | :--- | :--- | :--- |
| **Wan 2.2 (14B)** | FP16 | 5GB / 16GB / 50~70GB | Không OOM | ~45 minutes |
| **Wan 2.2 (14B)** | Q4_K_M | 5GB / 16GB / 40GB | Tốt | ~25 minutes |
| **SDXL** | Q8_0 | 5GB / 16GB / 30GB | Tốt | ~30 minutes |

---

###  Cấu hình đã thử nghiệm (Tested Configuration) ### 

```json
{
    "wan_num_frames": 81,
    "wan_fps": 16,
    "wan_guidance_scale": 1.0,
    "wan_num_inference_steps": 20,
    "wan_height": 480,
    "wan_width": 832,
    "lora_name": "",
    "lora_alpha": 0.8,
    "shard_size_gb": 4.0
}
```
> **Lưu ý:** `shard_size_gb` là kích thước của mỗi shard khi tiến hành convert. Với mức `4.0`, Peak bộ nhớ tiêu thụ sẽ luôn luôn rơi vào 4GB lúc nạp model để chống OOM, hiện cũng đã tích hợp tính năng STREAM để giảm tối đa và có tích hợp thêm disk_offload. Bạn có thể giữ nguyên mức `4.0` hoặc tăng lên cao tùy vào mức mà muốn hoặc nếu có cấu hình khỏe thì cứ nâng lên thoải mái, chi tiết thêm vui lòng đọc chi tiết trong file vì đã có ghi chi tiết từng chức năng. Dự án đã và đang được phát triển cá nhân độc lập và cũng là bản alpha đầu tay, mọi tối ưu và nâng cấp mới sẽ được cập nhật ngay sau khi được test ổn định.
