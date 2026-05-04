import torch
import folder_paths
import comfy.sd
import comfy.model_management as mm
import gc

class VAELoaderBFloat16:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "Custom/Loaders"

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        print(f"[🔥] Đang nạp VAE từ: {vae_path}")
        
        # 1. Nạp file VAE từ ổ cứng
        sd = comfy.utils.load_torch_file(vae_path)
        
        # 2. Khởi tạo đối tượng VAE (ComfyUI sẽ mặc định nạp fp16 trên 40HX)
        vae = comfy.sd.VAE(sd=sd)
        
        # 3. PHẪU THUẬT ÉP KIỂU: Tóm cổ model bên trong và ép sang BFloat16
        if hasattr(vae, "first_stage_model"):
            print("[⚙️] Cảnh báo: Đang cưỡng ép VAE sang chuẩn BFloat16...")
            vae.first_stage_model.to(torch.bfloat16)
            
            # Kiểm tra xem có lớp nào bị sót không
            for module in vae.first_stage_model.modules():
                module.to(torch.bfloat16)
        
        # 4. Dọn dẹp rác thừa phát sinh trong quá trình nạp
        del sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"[✅] VAE đã sẵn sàng với chuẩn BF16 (Backend: {vae.first_stage_model.dtype})")
        return (vae,)

NODE_CLASS_MAPPINGS = {
    "VAELoaderBFloat16": VAELoaderBFloat16
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAELoaderBFloat16": "VAE Loader (Force BF16)"
}