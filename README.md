# AIO-GGUF Backend V5 — Low-level AI Inference Backend (NOT a Website)

> ⚠️ This repository is **NOT a CV website or frontend project**.  
> It contains a **low-level inference backend system** for running large AI models using GGUF and Diffusers on constrained hardware.

---

### 📖 Overview

**AIO-GGUF Backend V5** is a comprehensive **AI inference backend** designed to run massive models (such as **Wan 2.2 14B** and **Stable Diffusion XL**) on resource-constrained hardware.

The system focuses on:
- Running **quantized GGUF models**
- Integrating with **Diffusers pipelines**
- Optimizing execution under **severe hardware limitations**

The project has been successfully deployed on:
- **GPU CMP 40HX 8GB (Modded PCIe 1.1 x16)**
- **CUDA 11.8 environment**
- **Driver 460.89 (CU112 backend)**

---

### 📖 Giới thiệu

**AIO-GGUF Backend V5** là một **hệ thống backend AI cấp thấp** được thiết kế để chạy các mô hình lớn (như **Wan 2.2 14B**, **Stable Diffusion XL**) trên phần cứng có tài nguyên hạn chế.

Dự án đã triển khai thành công trên:
- **GPU CMP 40HX 8GB (Mod PCIe 1.1 x16)**
- **CUDA 11.8**
- **Driver 460.89 (CU112 backend)**

---

# 🧠 Backend Architecture (Alpha V0.0.1)

This project implements a **high-performance GGUF-to-Diffusers execution backend** that:

- Leverages **(llama.cpp / sd.cpp).dll**
- Loads and runs **quantized models directly inside Diffusers**
- Bypasses the standard **Safetensors-heavy pipeline**
- Provides a **lighter and faster inference path**
- Targets current **optimization gaps in Diffusers' GGUF support**

---

### ⚙️ Core Direction

- GGUF-native execution backend  
- Quantized inference (GGML / GGUF)  
- Direct integration with Diffusers runtime  
- Optimized for constrained GPU environments  
- Designed for performance, compatibility, and reduced memory overhead  

---

### 🛡️ Project Information

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

<p align="center">
  <img src="https://img.shields.io/badge/Hardware-GPU%20CMP%2040HX%208GB%20(PCIe%201.1%20x16)-orange">
  <img src="https://img.shields.io/badge/CUDA-11.2%20(Driver%20460.89)-blue">
  <img src="https://img.shields.io/badge/Backend-GGUF%20Native-green">
</p>

