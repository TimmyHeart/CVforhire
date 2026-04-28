### 🛡️ Thông tin dự án
> *This project is a personal initiative. I am open to further discussions and collaborations. Please contact me directly for technical details and the latest updates.*

<p align="center">
  <b>Contribution & Copyright © Solely owned by Tran Hieu Nghia</b><br>
  <b>Đóng góp & Bản quyền thuộc về duy nhất cá nhân Trần Hiếu Nghĩa</b><br>
  <b>Ý tưởng & Kiến trúc:</b> Trần Hiếu Nghĩa (Nickname: Timmyo/V)<br>
  <b>Concept & Architecture:</b> Tran Hieu Nghia (Nickname: Timmyo/V)<br>
  <b>Thực thi:</b> Hỗ trợ bởi hệ thống AI (Gemini 1.5 Pro, Claude 3.5, ChatGPT, Grok)<br>
  <b>Implementation:</b> Assisted by AI Systems (Gemini 1.5 Pro, Claude 3.5, ChatGPT, Grok)<br>
  <b>Cảm hứng:</b> Dựa trên các dự án mã nguồn mở của <i>ggml, llama.cpp-Python, sd.cpp-Python, ComfyUI-GGUF, Diffusers</i>
</p>

---

### 📖 Giới thiệu
**AIO-GGUF Backend V5** là một giải pháp toàn diện được thiết kế để chạy các mô hình AI khổng lồ (như **Wan 2.2 14B**, **Stable Diffusion XL**) trên phần cứng có tài nguyên hạn chế. 

Dự án đã thực hiện thành công trên phần cứng **GPU CMP 40HX 8GB (Mod PCIe 1.1 x16)**, sử dụng môi trường **CUDA 11.8** với backend từ Driver **460.89** là CU112

### 📖 Overview
**AIO-GGUF Backend V5** is a comprehensive solution designed to run massive AI models (such as **Wan 2.2 14B** and **Stable Diffusion XL**) on resource-constrained hardware.

The project has been successfully deployed on **GPU 40HX 8GB (Modded PCIe 1.1 x16)** hardware, utilizing a **CUDA 11.8** environment with backends powered by **Driver 460.89 (CU112)**.
---
# 🧠 AIO-GGUF Backend (Alpha V0.0.1)
### **I’m building a high-performance GGUF-to-Diffusers backend that leverages (llama.cpp/sd.cpp).dll to load and run quantized models directly through Diffusers. It's designed to be much lighter and faster than the native Safetensors pipeline, specifically addressing the current optimization gaps in Diffusers' GGUF implementation.**

<p align="center">
  <img src="https://img.shields.io/badge/Hardware-GPU%20CMP%2040HX%208GB%20(PCIe%201.1%20x16)-orange">
  <img src="https://img.shields.io/badge/CUDA-11.2%20(Driver%20460.89)-blue">
  <img src="https://img.shields.io/badge/Backend-GGUF%20Native-green">
</p>

---

