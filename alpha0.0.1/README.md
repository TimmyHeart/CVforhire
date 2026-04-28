



                 Đóng góp & Bản quyền
Ý tưởng & Kiến trúc: [ Tên : Trần Hiếu Nghĩa Nickname: Timmyo/V]
Thực thi: Hỗ trợ bởi các AI Gemini 3.1 Pro, Claude 4.6, Chatgpt, Grok
Cảm hứng: Dựa trên các dự án mã nguồn mở của ggml, llama.cpp-Python và sd.cpp-Python, comfyUI-GGUF, Diffusers


                                             AIO-GGUF Backend (Version 5.0 Full)High-Performance GGUF Integration for Diffusers & Wan 2.2

                      Giới thiệu (Overview)AIO-GGUF Backend V5 là một giải pháp toàn diện được thiết kế để chạy các mô hình AI khổng lồ (như Wan 2.2 14B, Stable Diffusion XL) 

                    Trên phần cứng có tài nguyên hạn chế, đã được thực hiện trên phần cứng GPU 40HX 8GB đã mod pcie 1.1 x16, sử dụng CU118 với backend từ driver 460.89 với CU112
                            Hệ thống này phá vỡ rào cản bộ nhớ bằng cách thay thế các lớp nn.Linear tiêu chuẩn của HuggingFace Diffusers bằng các lớp GGUFLinear tùy chỉnh, 
                                             Cho phép thực thi trực tiếp các trọng số đã được nén (Quantized) từ file GGUF mà không cần giải nén toàn bộ vào VRAM.

##############################################                  Tính năng nổi bật (Key Features):           #####################################################################################
#              Hybrid Execution Bridge: Tự động chuyển đổi giữa nhân C++ (qua sd.cpp-python DLL) để đạt tốc độ tối đa và Python fallback để đảm bảo tính tương thích tuyệt đối.                 #
#                             #Wan 2.2 Native Support: Xử lý hoàn hảo cấu trúc Tensor 5D đặc thù của mô hình Video Wan 2.2.                                                                     #
#       Critical Patches (v4/v5):CRITICAL-1 (Alias Mapping): Tự động ánh xạ các layer alias (self_attn ↔ attn1) giúp tương thích với mọi bản checkpoint trên Civitai/HuggingFace.               #
#       CRITICAL-2 (QKV Split Logic): Thuật toán phân tách khối QKV Packed dựa trên thuộc tính transpose thời gian thực, đảm bảo độ chính xác của Attention.                                    #
#                     Memory Efficiency: Sử dụng mmap để ánh xạ file từ ổ cứng, giảm thiểu tối đa dung lượng System RAM và VRAM tiêu thụ.                                                       #
#                  Built-in Self-Test: Tích hợp bộ suite kiểm thử tự động, đảm bảo mọi layer đều được patch đúng trước khi tiến hành Inference.                                                 #
#                            Kiến trúc hệ thống (Architecture)GGMLTensor: Lớp con của torch.Tensor lưu trữ siêu dữ liệu nén.                                                                    #
#                                  GGUFTensorRegistry: Quản lý kho lưu trữ trọng số tập trung, ngăn chặn rò rỉ bộ nhớ.                                                                          #
#                          GGUFLinear: Module thay thế nn.Linear, thực hiện Dequantize-on-the-fly hoặc gọi trực tiếp qua DLL Bridge.                                                            #
#                                  KeyMapper: Công cụ ánh xạ tên layer thông minh giữa định dạng GGUF và kiến trúc Diffusers.                                                                   #
#########################################################      Hướng dẫn sử dụng (Usage)         ################################################################################################
#                1. Cài đặt môi trường Bashpip install torch diffusers transformers sentencepiece                                                                                               #
#                   Đảm bảo đã cài đặt stable-diffusion-cpp-python/llama.cpp-python để có hiệu năng tốt nhất                                                                                    #
#                   pip install stable-diffusion-cpp-python                                                                                                                                     #
#                2. Chạy kiểm tra hệ thốngĐể đảm bảo backend hoạt động hoàn hảo trên máy của bạn: py -3.10 gguf_backend_v5_full.py                                                              #
#                3. Tích hợp vào Pipeline Python :                                                                                                                                              #
#                  from gguf_backend_v5_full import build_unet_gguf_native                                                                                                                      #                                                                                                         #
#                        unet = build_unet_gguf_native(                                                                                                                                         #
#                               model_path="models/wan2.2_14B_Q4_K_M.gguf",                                                                                                                     #
#                               device="cuda"                                                                                                                                                   #  
#                               )                                                                                                                                                               #
#                                                                                                                                                                                               #
#                 Sau đó đưa vào Diffusers Pipeline như bình thường                                                                                                                             #
#################################################################################################################################################################################################
#                                                                                                                                                                                               #
#Model                 Format           VRAM/RAM/filepage Usages                Status            Time from start to output                                                                     #
#Wan 2.2 (14B)         FP16               5GB/16GB/50~70GB                    không OOM                  45 minutes                                                                             #
#Wan 2.2 (14B)        Q4_K_M              5GB/16GB/40GB                         Tốt                     25 minutes                                                                              #
#SDXL                  Q8_0               5GB/16GB/30GB                         Tốt                      30 minutes                                                                             #
#                                                                                                                                                                                               #
#                                       CONFIG đã test :                                                                                                                                        #
#    "wan_num_frames":          81,                                                                                                                                                             #
#    "wan_fps":                 16,                                                                                                                                                             #
#    "wan_guidance_scale":      1.0,                                                                                                                                                            #
#    "wan_num_inference_steps": 20,                                                                                                                                                             #
#    "wan_height":              480,                                                                                                                                                            #
#    "wan_width":               832,                                                                                                                                                            #
#    "lora_name":               "",                                                                                                                                                             #
#    "lora_alpha":              0.8,                                                                                                                                                            #
#    # Kích thước mỗi shard khi convert. 4GB = peak RAM ~4GB lúc convert.                                                                                                                       #
#    # Giảm xuống 2 nếu máy hay OOM lúc convert.                                                                                                                                                #
#    "shard_size_gb":           4.0,                                                                                                                                                            #
#                                                                                                                                                                                               #
#                                                                                                                                                                                               #
#                                                                                                                                                                                               #
#################################################################################################################################################################################################
