import gc
import torch
import nodes
import comfy.model_management as mm

class ZeroCacheWanBF16Flexible:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "start_image": ("IMAGE", ),
                "length": ("INT", {"default": 81, "min": 1, "max": 9999}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 9999}),
            },
            "optional": {
                # Trả clip_vision về đúng vị trí optional, không ép buộc nữa!
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "process"
    CATEGORY = "Custom/Zero Cache Worker"

    def process(self, positive, negative, vae, start_image, length, batch_size, clip_vision_output=None, width=None, height=None):
        print(f"\n[🚀] Khởi động Wrapper BF16 (Chế độ Linh hoạt)...")
        
        # 1. Tìm node gốc
        TargetClass = nodes.NODE_CLASS_MAPPINGS.get("WanImageToVideo")
        if TargetClass is None:
            raise RuntimeError("❌ Không tìm thấy node WanImageToVideo gốc!")
            
        node_instance = TargetClass()
        func_name = getattr(TargetClass, "FUNCTION", "process") 
        original_function = getattr(node_instance, func_name)
        
        # 2. Chuẩn bị dữ liệu và ép kiểu BF16 cho các tensor nặng
        # Ép start_image về BF16 để truyền tải cho "mượt"
        start_image_bf16 = start_image.to(torch.bfloat16)
        
        kwargs = {
            "positive": positive,
            "negative": negative,
            "vae": vae,
            "start_image": start_image_bf16,
            "length": length,
            "batch_size": batch_size,
            "clip_vision_output": clip_vision_output # Có thể là None, node gốc sẽ tự xử
        }
        
        if width is not None: kwargs["width"] = width
        if height is not None: kwargs["height"] = height
        
        # 3. Thực thi với BF16 Autocast (Ép PyTorch dùng luồng xử lý thông minh nhất)
        # Tắt fp16 autocast, chỉ cho phép bf16 hoặc float32
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            print("[⚙️] Đang tính toán qua backend thông minh (BF16/FP32 combo)...")
            result = original_function(**kwargs)
        
        # Tách kết quả
        out_positive, out_negative, out_latent = result

        # 4. Ép cục Latent đầu ra về BFloat16 để "đánh lừa" băng thông PCIe
        if isinstance(out_latent, dict) and "samples" in out_latent:
            out_latent["samples"] = out_latent["samples"].to(torch.bfloat16)

        print("[✅] Xong! Tiến hành tự sát dọn rác...")

        # 5. DỌN DẸP CHIẾN TRƯỜNG (Sạch sẽ tuyệt đối)
        del node_instance, kwargs, start_image_bf16, result
        
        # Thu hồi mọi biến tạm trong scope này
        gc.collect()
        
        # Vắt kiệt VRAM
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        print("[♻️] VRAM đã được giải phóng hoàn toàn.\n")
        
        return (out_positive, out_negative, out_latent)

NODE_CLASS_MAPPINGS = {
    "WanImageToVideoBF16Flex": ZeroCacheWanBF16Flexible
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanImageToVideoBF16Flex": "WanImageToVideo (Zero-Cache BF16 Flexible)"
}