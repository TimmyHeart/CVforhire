# 🧠 AIO-GGUF Backend (Version 5.0 Full)
### **High-Performance GGUF Integration for Diffusers & Wan 2.2**

<p align="center">
  <img src="https://img.shields.io/badge/Hardware-GPU%2040HX%208GB%20(PCIe%201.1%20x16)-orange">
  <img src="https://img.shields.io/badge/CUDA-11.8%20(Driver%20460.89)-blue">
  <img src="https://img.shields.io/badge/Backend-GGUF%20Native-green">
</p>

---

### 🛡️ Thông tin dự án
> *This project is a personal initiative. I am open to further discussions and collaborations. Please contact me directly for technical details and the latest updates.*

<p align="center">
  <b>Đóng góp & Bản quyền thuộc về duy nhất cá nhân Trần Hiếu Nghĩa</b><br>
  <b>Ý tưởng & Kiến trúc:</b> Trần Hiếu Nghĩa (Nickname: Timmyo/V)<br>
  <b>Thực thi:</b> Hỗ trợ bởi hệ thống AI (Gemini 1.5 Pro, Claude 3.5, ChatGPT, Grok)<br>
  <b>Cảm hứng:</b> Dựa trên các dự án mã nguồn mở của <i>ggml, llama.cpp-Python, sd.cpp-Python, ComfyUI-GGUF, Diffusers</i>
</p>

---

### 📖 Giới thiệu (Overview)
**AIO-GGUF Backend V5** là một giải pháp toàn diện được thiết kế để chạy các mô hình AI khổng lồ (như **Wan 2.2 14B**, **Stable Diffusion XL**) trên phần cứng có tài nguyên hạn chế. 

Dự án đã thực hiện thành công trên phần cứng **GPU 40HX 8GB (Mod PCIe 1.1 x16)**, sử dụng môi trường **CUDA 11.8** với backend từ Driver **460.89**.

### 🛠️ Cơ chế hoạt động
Hệ thống này phá vỡ rào cản bộ nhớ bằng cách:
* **Thay thế** các lớp `nn.Linear` tiêu chuẩn của HuggingFace Diffusers bằng các lớp **`GGUFLinear`** tùy chỉnh.
* **Thực thi trực tiếp** các trọng số đã được nén (Quantized) từ file GGUF mà không cần giải nén toàn bộ vào VRAM.
* **Tối ưu hóa** luồng dữ liệu cho các chuẩn PCIe băng thông thấp nhưng vẫn đảm bảo tốc độ Inference ổn định.

---


