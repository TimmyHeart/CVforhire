import os
import sys
import json
import torch
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
comfy_root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if comfy_root_dir not in sys.path:
    sys.path.insert(0, comfy_root_dir)

import folder_paths
try:
    import nodes
    AVAILABLE_CLIP_TYPES = nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0]
except Exception:
    AVAILABLE_CLIP_TYPES = ["stable_diffusion", "sdxl", "sd3", "flux", "wan", "hunyuan_video", "mochi", "ltxv"]

def isolated_clip_worker(clip_name, clip_type, prompt, negative_prompt, output_path):
    import torch
    import nodes 
    import gc
    
    clip_loader = nodes.CLIPLoader()
    clip = clip_loader.load_clip(clip_name=clip_name, type=clip_type)[0]
    
    tokens_pos = clip.tokenize(prompt)
    cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
    
    tokens_neg = clip.tokenize(negative_prompt)
    cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
    
    torch.save({
        "pos_cond": cond_pos.cpu() if cond_pos is not None else None,
        "pos_pooled": pooled_pos.cpu() if pooled_pos is not None else None,
        "neg_cond": cond_neg.cpu() if cond_neg is not None else None,
        "neg_pooled": pooled_neg.cpu() if pooled_neg is not None else None
    }, output_path)

    del clip, cond_pos, pooled_pos, cond_neg, pooled_neg, clip_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class ZeroCacheUniversalCLIP:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip_name": (folder_paths.get_filename_list("clip"), ),
            "clip_type": (AVAILABLE_CLIP_TYPES, {"default": "wan"}), 
            "device": (["vga + cpu", "vga", "cpu"], {"default": "vga + cpu"}), 
            "prompt": ("STRING", {"multiline": True, "default": "masterpiece, best quality, cinematic"}),
            "negative_prompt": ("STRING", {"multiline": True, "default": "lowres, bad anatomy, worst quality"}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "Custom/Zero Cache Worker"

    def encode(self, clip_name, clip_type, device, prompt, negative_prompt):
        temp_dir = folder_paths.get_temp_directory()
        import uuid
        unique_id = uuid.uuid4().hex
        
        output_path = os.path.join(temp_dir, f"isolated_embeds_{unique_id}.pt")
        payload_path = os.path.join(temp_dir, f"isolated_payload_{unique_id}.json")
        error_log_path = os.path.join(temp_dir, f"isolated_error_{unique_id}.txt")
        
        payload = {
            "clip_name": clip_name,
            "clip_type": clip_type,
            "device": device,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "output_path": output_path,
            "error_log_path": error_log_path
        }
        with open(payload_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)

        print(f"\n[⚡] Active Zero-Cache Worker: [{clip_name}] | Type: [{clip_type.upper()}] | Device: [{device.upper()}]")
        
        script_path = os.path.abspath(__file__)        
        result = subprocess.run(
            [sys.executable, script_path, payload_path], 
            capture_output=True, 
            text=True
        )
        
        if os.path.exists(error_log_path):
            with open(error_log_path, "r", encoding="utf-8") as f:
                err_msg = f.read()
            try: os.remove(error_log_path)
            except: pass
            raise RuntimeError(f"❌ LUỒNG PHỤ BỊ LỖI:\n\n{err_msg}")
            
        if result.returncode != 0 or not os.path.exists(output_path):
            raise RuntimeError(f"Subprocess Crash! Code: {result.returncode}\n{result.stderr}\n{result.stdout}")
            
        print("[✅] Encode xong! Đang đưa Condition vào luồng chính...")
        
        data = torch.load(output_path, map_location="cpu")
        pos_out = [[data["pos_cond"], {"pooled_output": data["pos_pooled"]}]]
        neg_out = [[data["neg_cond"], {"pooled_output": data["neg_pooled"]}]]
        
        try: os.remove(output_path); os.remove(payload_path)
        except: pass
            
        return (pos_out, neg_out)

NODE_CLASS_MAPPINGS = {
    "ZeroCacheUniversalCLIP": ZeroCacheUniversalCLIP
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZeroCacheUniversalCLIP": "Zero-Cache Universal Text Encode (Worker)"
}

if __name__ == "__main__":
    import sys
    import json
    import traceback
    
    payload_path = sys.argv[1]
    
    try:
        with open(payload_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        device_choice = data.get("device", "vga + cpu")
        if device_choice == "cpu":
            sys.argv = [sys.argv[0], "--cpu"]
        elif device_choice == "vga":
            sys.argv = [sys.argv[0], "--normalvram"] 
        else:
            sys.argv = [sys.argv[0], "--lowvram"] 
            
        error_log_path = data.get("error_log_path", payload_path + ".error.txt")
        
        isolated_clip_worker(
            clip_name=data["clip_name"],
            clip_type=data["clip_type"],
            prompt=data["prompt"],
            negative_prompt=data["negative_prompt"],
            output_path=data["output_path"]
        )
    except Exception as e: 
        with open(data.get("error_log_path", payload_path + ".error.txt"), "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        sys.exit(1)