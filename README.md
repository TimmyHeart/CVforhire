https://github.com/TimmyHeart/Multi-framework-inference-runtime
### 🛡️ Thông tin dự án
> *This project is a personal initiative. I am open to further discussions and collaborations. Please contact me directly for technical details and the latest updates.*

<p align="center">
  <b>Contribution & Copyright © Solely owned by Tran Hieu Nghia</b><br>
  <b>Đóng góp & Bản quyền thuộc về duy nhất cá nhân Trần Hiếu Nghĩa</b><br>
  <b>Ý tưởng & Kiến trúc:</b> Trần Hiếu Nghĩa (Nickname: Timmyo/V)<br>
  <b>Concept & Architecture:</b> Tran Hieu Nghia (Nickname: Timmyo/V)<br>
  <b>Thực thi:</b> Hỗ trợ bởi hệ thống AI (Gemini, Claude, ChatGPT, Grok)<br>
  <b>Implementation:</b> Assisted by AI Systems (Gemini, Claude, ChatGPT, Grok)<br>
  <b>Cảm hứng:</b> Dựa trên các dự án mã nguồn mở của <i>ggml, llama.cpp-Python, sd.cpp-Python, ComfyUI-GGUF, Diffusers</i>
</p>

---

### 📖 Giới thiệu
**AIO-GGUF Backend V5** là một giải pháp toàn diện được thiết kế để chạy các mô hình AI (như **Wan 2.2 14B**, **Stable Diffusion XL**) trên phần cứng có tài nguyên hạn chế. 

Dự án đã thực hiện thành công trên phần cứng **GPU CMP 40HX 8GB (Mod PCIe 1.1 x16)**, sử dụng môi trường **CUDA 11.8** với backend từ Driver **460.89** là CU112

### 🚀 Hành trình Tối ưu hóa: Từ Crash/1000s+ xuống 153s/it
Dự án này là minh chứng cho việc bứt phá giới hạn vật lý của phần cứng cũ. Với một trong những mô hình AI mới nhất - Wan2.2 14B, băng thông chật hẹp của PCIe 1.1 x16 ban đầu đã gây ra tình trạng thắt cổ chai trầm trọng (1000s/it) và crash hệ thống. Các kỹ thuật lõi đã được áp dụng để giải quyết:

* **Vượt rào Băng thông với Pinned Memory (DMA):** Viết lại luồng quản lý bộ nhớ ở cấp độ thấp, sử dụng bộ đệm cứng `pin_memory=True`. Kỹ thuật này ép hệ điều hành cấp phát bộ nhớ vật lý độc quyền, cho phép GPU rút dữ liệu trực tiếp (Direct Memory Access) từ RAM hệ thống cực nhanh mà không qua xử lý trung gian của CPU, đồng thời khóa đồng bộ để chống kẹt luồng khe PCIe.
* **Tối ưu Lõi Attention cho Kiến trúc Turing:** Tái cấu trúc hàm Scaled Dot-Product Attention (SDPA). Đánh lừa những giới hạn về tập lệnh phần cứng của GPU CMP 40HX bằng cách ép xử lý cục bộ, ép Tensor Cores phải hoạt động hết công suất.
* **Kiểm soát luồng dữ liệu & Batch Size tối đa:** Loại bỏ các phép tính dư thừa bằng cách can thiệp sâu vào thông số CFG và KSampler, giữ Batch Size ở mức tối giản nhất để giảm một nửa khối lượng dữ liệu khổng lồ phải lưu thông qua khe cắm đồ họa.

**🔥 Kết quả:** Từ việc không thể chạy nổi, hệ thống đã render thành công Video bằng mô hình Wan 2.2 14B (81 frames, độ phân giải 480x480, FPS 32) với tốc độ **153s/it**. Đây là một cột mốc đột phá trên chuẩn PCIe 1.1 x16, nhưng mới chỉ là giới hạn tạm thời — quá trình tối ưu hóa vẫn đang được tiếp tục để vắt kiệt từng giọt hiệu năng cuối cùng!

### 📖 Overview
**AIO-GGUF Backend V5** is a comprehensive solution designed to run massive AI models (such as **Wan 2.2 14B** and **Stable Diffusion XL**) on resource-constrained hardware.

The project has been successfully deployed on **GPU 40HX 8GB (Modded PCIe 1.1 x16)** hardware, utilizing a **CUDA 11.8** environment with backends powered by **Driver 460.89 (CU112)**.

### 🚀 Optimization Journey: From Crash/1000s+ down to 153s/it
This project proves that physical hardware limitations can be bypassed. With one of the newest AI models - Wan2.2 14B, the narrow bandwidth of the PCIe 1.1 x16 slot initially caused severe bottlenecks (1000s/it) and system crashes. The following core techniques were implemented:

* **Bypassing Bandwidth Walls with Pinned Memory (DMA):** Rewrote low-level memory management using a `pin_memory=True` buffer. This forces the OS to allocate page-locked physical memory, allowing the GPU to perform Direct Memory Access (DMA) to fetch data from system RAM instantaneously without CPU overhead, completely solving PCIe traffic jams.
* **Core Attention Tuning for Turing Architecture:** Reconstructed the Scaled Dot-Product Attention (SDPA) mechanism. Bypassed the native instruction set limitations of the CMP 40HX GPU by forcing local processing, pushing the Tensor Cores to their absolute maximum efficiency.
* **Data Flow & Batch Size Control:** Eliminated redundant computations by deep-diving into CFG and KSampler parameters, keeping the Batch Size minimal to halve the massive data load forced through the GPU slot.

**🔥 The Result:** From total system crashes, the environment successfully generated Video using the massive Wan 2.2 14B model (81 frames, 480x480, 32 FPS) at a stable **153s/it**. This marks a massive breakthrough on the legacy PCIe 1.1 x16 interface, but it serves only as a current milestone — ongoing optimizations will continue to push these bounds and squeeze out every last drop of performance!

---

# 🧠 AIO-GGUF Backend (Alpha V0.0.1)
### **I’m building a high-performance GGUF-to-Diffusers backend that leverages (llama.cpp/sd.cpp).dll to load and run quantized models directly through Diffusers. It's designed to be much lighter and faster than the native Safetensors pipeline, specifically addressing the current optimization gaps in Diffusers' GGUF implementation.**

<p align="center">
  <img src="https://img.shields.io/badge/Hardware-GPU%20CMP%2040HX%208GB%20(PCIe%201.1%20x16)-orange">
  <img src="https://img.shields.io/badge/CUDA-11.2%20(Driver%20460.89)-blue">
  <img src="https://img.shields.io/badge/Backend-GGUF%20Native-green">
</p>

---
