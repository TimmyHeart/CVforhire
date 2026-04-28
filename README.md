This project is a personal initiative. I am open to further discussions and collaborations. Please contact me directly for technical details and the latest updates.
                                                Đóng góp & Bản quyền thuộc về duy nhất cá nhân Trần Hiếu Nghĩa.
                                                Ý tưởng & Kiến trúc: [ Tên : Trần Hiếu Nghĩa Nickname: Timmyo/V].
                                               Thực thi: Hỗ trợ bởi các AI Gemini 3.1 Pro, Claude 4.6, Chatgpt, Grok.
                             Cảm hứng: Dựa trên các dự án mã nguồn mở của ggml, llama.cpp-Python và sd.cpp-Python, comfyUI-GGUF, Diffusers.
                                             AIO-GGUF Backend (Version 5.0 Full)High-Performance GGUF Integration for Diffusers & Wan 2.2.
                      Giới thiệu (Overview)AIO-GGUF Backend V5 là một giải pháp toàn diện được thiết kế để chạy các mô hình AI khổng lồ (như Wan 2.2 14B, Stable Diffusion XL).
                    Trên phần cứng có tài nguyên hạn chế, đã được thực hiện trên phần cứng GPU 40HX 8GB đã mod pcie 1.1 x16, sử dụng CU118 với backend từ driver 460.89 với CU112.
                            Hệ thống này phá vỡ rào cản bộ nhớ bằng cách thay thế các lớp nn.Linear tiêu chuẩn của HuggingFace Diffusers bằng các lớp GGUFLinear tùy chỉnh, 
                                             Cho phép thực thi trực tiếp các trọng số đã được nén (Quantized) từ file GGUF mà không cần giải nén toàn bộ vào VRAM.


