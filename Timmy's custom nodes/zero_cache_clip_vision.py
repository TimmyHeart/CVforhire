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
except Exception:
    pass

def isolated_clip_vision_worker(clip_name, input_image_path, output_path):
    import torch
    import nodes 
    import gc
    clip_loader = nodes.CLIPVisionLoader()
    clip_vision = clip_loader.load_clip(clip_name=clip_name)[0]
    image = torch.load(input_image_path, map_location="cpu", weights_only=True)
    clip_vision_output = clip_vision.encode_image(image)
    torch.save(clip_vision_output, output_path)
    
class ZeroCacheCLIPVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip_name": (folder_paths.get_filename_list("clip_vision"), ),
            "image": ("IMAGE", ),
            "device": (["vga + cpu", "vga", "cpu"], {"default": "vga + cpu"}), 
        }}
    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    RETURN_NAMES = ("clip_vision_output",)
    FUNCTION = "encode"
    CATEGORY = "Custom/Zero Cache Worker"
    def encode(self, clip_name, image, device):
        temp_dir = folder_paths.get_temp_directory()
        import uuid
        unique_id = uuid.uuid4().hex
        payload_path = os.path.join(temp_dir, f"cv_payload_{unique_id}.json")
        input_image_path = os.path.join(temp_dir, f"cv_in_img_{unique_id}.pt")
        output_path = os.path.join(temp_dir, f"cv_out_{unique_id}.pt")
        error_log_path = os.path.join(temp_dir, f"cv_err_{unique_id}.txt")
        torch.save(image.cpu(), input_image_path)
        payload = {
            "clip_name": clip_name,
            "input_image_path": input_image_path,
            "output_path": output_path,
            "error_log_path": error_log_path,
            "device": device
        }
        with open(payload_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        print(f"\n Active Zero-Cache Vision Worker: [{clip_name}] | Device: [{device.upper()}]")
        script_path = os.path.abspath(__file__)        
        result = subprocess.run(
            [sys.executable, script_path, payload_path], 
            capture_output=True, 
            text=True
        )
        if os.path.exists(error_log_path):
            with open(error_log_path, "r", encoding="utf-8") as f:
                err_msg = f.read()
            try: 
                os.remove(error_log_path)
                os.remove(input_image_path)
                os.remove(payload_path)
            except: pass
            raise RuntimeError(f"LUỒNG PHỤ CLIP VISION BỊ LỖI:\n\n{err_msg}")
        if result.returncode != 0 or not os.path.exists(output_path):
            raise RuntimeError(f"Subprocess Crash! Code: {result.returncode}\n{result.stderr}\n{result.stdout}")
        print(" Vision Encode xong! Đang thu hồi Data")
        out_data = torch.load(output_path, map_location="cpu", weights_only=False)
        try: 
            os.remove(input_image_path)
            os.remove(output_path)
            os.remove(payload_path)
        except: pass
        return (out_data,)
NODE_CLASS_MAPPINGS = {
    "ZeroCacheCLIPVision": ZeroCacheCLIPVision
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZeroCacheCLIPVision": "Zero-Cache CLIP Vision Encode (Worker)"
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
        isolated_clip_vision_worker(
            clip_name=data["clip_name"],
            input_image_path=data["input_image_path"],
            output_path=data["output_path"]
        )
    except Exception as e: 
        err_path = data.get("error_log_path", payload_path + ".error.txt") if 'data' in locals() else payload_path + ".error.txt"
        with open(err_path, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        sys.exit(1)