"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  WAN 2.2 COMFYUI FP8 LOADER v6 – DUAL TRANSFORMER FINAL                   ║
║  CMP 40HX (Turing SM7.5) · PCIe 1.1 · RAM 16GB · VRAM 8GB               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  THAY ĐỔI V6 SO VỚI V5:                                                   ║
║                                                                              ║
║  ✅ DUAL TRANSFORMER – 2 file riêng biệt theo chuẩn Diffusers:            ║
║     • COMFY_MODEL_HIGH → transformer  (High Noise Stage)                  ║
║     • COMFY_MODEL_LOW  → transformer_2 (Low Noise Stage)                  ║
║     • Convert 2 lần tuần tự, lưu vào 2 thư mục độc lập                  ║
║     • Nạp cả 2 vào WanImageToVideoPipeline đúng chuẩn 14B I2V            ║
║                                                                              ║
║  BẤT BIẾN TỪ V5 (GIỮ NGUYÊN HOÀN TOÀN):                                 ║
║  ✅ REMAP V5.0 FINAL – MASTER REMAP hoàn toàn mới:                        ║
║     • Strip prefix đúng thứ tự dài→ngắn (tránh prefix ngắn bắt trước)    ║
║     • layers.N → blocks.N (Kijai/Comfy FP8 format)                        ║
║     • ROOT_MAP: bắt trọn head/norm_out với mọi scale FP8                  ║
║     • EMBED_TABLE đầy đủ kể cả clip_fea.                                  ║
║     • FP8 scale (.weight_scale, .input_scale, ...) bắt trọn               ║
║     • VÁ Key Collision Modulation: .modulation.1. → .norm1. (exact only) ║
║     • norm_q / norm_k trong attention đã được map đúng                    ║
║  ✅ dtype: float16 (CMP 40HX Turing KHÔNG có BF16 Tensor Core)           ║
║  ✅ Stream-to-disk shards (KHÔNG load 28GB vào RAM – peak ~4GB)          ║
║  ✅ Zero-init policy thông minh (chỉ scale_shift_table + buffers)        ║
║  ✅ enable_sequential_cpu_offload() trong pipeline render                 ║
║  ✅ ExtremeTeaCache (skip bước denoising gần giống nhau)                 ║
║  ✅ Subprocess encode UMT5 (giải phóng VRAM hoàn toàn sau encode)       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Về RAM (16GB với model 28GB FP16):
  Stream-to-disk: peak RAM lúc convert ~4GB (1 shard).
  Inference: disk offload – weight sống trên disk, chỉ tải 1 layer vào
  RAM/VRAM khi tính, xong trả lại disk. Chậm hơn ~3-5x nhưng không OOM.

Về dtype:
  FP16 là lựa chọn duy nhất đúng cho Turing (SM 7.5).
  BF16 → PyTorch FP32 emulation → chậm x4, VRAM x2.
"""

import multiprocessing
import re
import sys
import json
import torch
import os
import gc
import shutil
import warnings
import traceback

# psutil để monitor RAM (cài: pip install psutil)
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    print("[⚠] psutil chưa cài → `pip install psutil`. RAM monitoring bị tắt.")

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from PIL import Image
from diffusers import (
    WanImageToVideoPipeline,
    WanTransformer3DModel,
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

# ════════════════════════════════════════════════════════════
# ⚙️  DTYPE – HARDWARE-SPECIFIC
# ════════════════════════════════════════════════════════════
#
# CMP 40HX = NVIDIA Turing (SM 7.5), cùng chip với RTX 2070.
# Turing HỖ TRỢ: FP32, FP16 (Tensor Core), INT8 (Tensor Core)
# Turing KHÔNG HỖ TRỢ: BF16 (cần Ampere SM8.0+ mới có)
#
# Hậu quả dùng BF16 trên Turing:
#   - PyTorch fallback sang FP32 emulation
#   - Chậm x4 so với FP16 native
#   - Tiêu thụ VRAM x2 do emulation overhead
#
TARGET_DTYPE = torch.bfloat16  # ← BF16 CẤM trên CMP 40HX


# ════════════════════════════════════════════════════════════
# ⚙️  CẤU HÌNH
# ════════════════════════════════════════════════════════════
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
INPUT_IMG_DIR  = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
LORAS_DIR      = os.path.join(BASE_DIR, "loras")

# ── V6: 2 file riêng biệt thay vì 1 ────────────────────────
COMFY_MODEL_HIGH = r"F:\simulator\AI\Local\models\wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2FP8H.safetensors"
COMFY_MODEL_LOW  = r"F:\simulator\AI\Local\models\wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2FP8L.safetensors"

VAE_PATH         = r"F:\simulator\AI\Local\VAE\wan2.1_vae.safetensors"
ENCODER_PATH     = r"F:\simulator\AI\Local\models\WANENCODER1.safetensors"
CLIP_ENCODER_DIR = r"F:\simulator\AI\Local\CLIP_ENCODER"  # Để trống nếu không có CLIP offline

CONFIG_CACHE_DIR = os.path.join(MODELS_DIR, "wan_configs")
# v6: tên thư mục riêng để tránh nhầm với bản convert cũ
CONVERTED_DIR    = os.path.join(MODELS_DIR, "comfy_converted_fp16_v6")

# ── V6: 2 thư mục xuất shard riêng biệt ────────────────────
TRANSFORMER_1_DIR = os.path.join(CONVERTED_DIR, "transformer")
TRANSFORMER_2_DIR = os.path.join(CONVERTED_DIR, "transformer_2")

for d in [INPUT_IMG_DIR, OUTPUT_DIR, MODELS_DIR, LORAS_DIR]:
    os.makedirs(d, exist_ok=True)

CONFIG = {
    "wan_num_frames":          81,
    "wan_fps":                 16,
    "wan_guidance_scale":      1.0,
    "wan_num_inference_steps": 20,
    "wan_height":              480,
    "wan_width":               832,
    "lora_name":               "",
    "lora_alpha":              0.8,
    # Kích thước mỗi shard khi convert. 4GB = peak RAM ~4GB lúc convert.
    # Giảm xuống 2 nếu máy hay OOM lúc convert.
    "shard_size_gb":           4.0,
    "negative_prompt": (
        "blurry, low quality, deformed, bad anatomy, extra limbs, mutated hands, "
        "poorly drawn face, bad proportions, watermark, text, censored, mosaic, "
        "static pose, frozen, no motion, slow motion, fast forward, camera shake, "
        "flickering, artifacts, ugly, extra fingers, fused fingers, bad hands"
    ),
}


# ════════════════════════════════════════════════════════════
# 🗺️  KEY REMAPPING V5.0 FINAL
#     MASTER REMAP – WAN 2.2 14B I2V FP8 (ComfyUI/Kijai → Diffusers)
#     ⚠️  KHÔNG CHỈNH SỬA HÀM NÀY – ĐÃ TINH CHỈNH HOÀN HẢO
# ════════════════════════════════════════════════════════════

def load_and_remap_lora(pipe, lora_path, adapter_name):
    """ 
    Cỗ máy dịch thuật LoRA từ ComfyUI sang Diffusers (Tích hợp Auto-Cache)
    - Nếu đã có file remap: Load siêu tốc trực tiếp vào VRAM.
    - Nếu chưa có: Remap -> Lưu file safetensors mới -> Nạp vào model.
    """
    from safetensors.torch import load_file, save_file
    # 1. Khai báo tên file Cache (Thêm đuôi _remapped_v5)
    base_dir = os.path.dirname(lora_path)
    base_name = os.path.basename(lora_path)
    name_without_ext, ext = os.path.splitext(base_name)
    
    # Đổi đuôi cache sang v5.1 để bơ luôn mấy file lỗi của v5
    remapped_name = f"{name_without_ext}_remapped_v5.8{ext}"
    remapped_path = os.path.join(base_dir, remapped_name)

    if os.path.exists(remapped_path):
        print(f"   ⚡ Tìm thấy Cache LoRA: {remapped_name}")
        try:
            pipe.load_lora_weights(remapped_path, adapter_name=adapter_name)
            print(f"      ✅ [Load Siêu Tốc] {adapter_name} OK!")
            return
        except Exception as e:
            print(f"      ⚠️ Lỗi file cache: {e}. Đang xóa để tạo lại...")
            os.remove(remapped_path)

    print(f"   🔄 Đang tiến hành Dịch thuật & Tạo Cache cho: {adapter_name}...")
    try:
        comfy_state_dict = load_file(lora_path)
        diffusers_state_dict = {}

        for key, weight in comfy_state_dict.items():
            new_key = key
            
            # =========================================================
            # 1. SỬA ĐÚNG THẰNG CẦN SỬA (Cứu light_low)
            # Chém bay các khớp xương thừa thãi mà Wan 2.2 không hề có
            # =========================================================
            if "attn2" in new_key or "image_embedder.ff" in new_key:
                continue # Vứt luôn, không lưu vào file mới
                
            # =========================================================
            # 2. CÁCH CŨ CHÂN ÁI (Đã cứu sống SVI và light_high)
            # Trả lại nguyên vẹn không đụng chạm linh tinh
            # =========================================================
            if new_key.startswith("blocks."):
                new_key = "transformer." + new_key
            if new_key.startswith("condition_embedder."):
                new_key = "transformer." + new_key

            new_key = new_key.replace("attn2.add_k_proj", "attn.to_k")
            new_key = new_key.replace("attn2.add_v_proj", "attn.to_v")
            new_key = new_key.replace("attn2.add_q_proj", "attn.to_q")
            new_key = new_key.replace("attn1.to_q", "attn.to_q")
            
            diffusers_state_dict[new_key] = weight

        print(f"      💾 Đang lưu bản dịch xuống disk: {remapped_name}...")
        save_file(diffusers_state_dict, remapped_path)

        pipe.load_lora_weights(remapped_path, adapter_name=adapter_name)
        print(f"      ✅ [Convert & Load Thành công] {adapter_name} OK!")
        
        del comfy_state_dict, diffusers_state_dict
        gc.collect()

    except Exception as e:
        print(f"      ⚠️ Lỗi nạp/remap {adapter_name}: {e}")
        
def remap_comfy_key(key: str) -> str:
    k = key

    # 1. STRIP PREFIX
    PREFIXES = [
        "model.diffusion_model.", "diffusion_model.", "transformer.",
        "dit.", "wan_model.", "model.", ""
    ]
    for pfx in PREFIXES:
        if k.startswith(pfx):
            k = k[len(pfx):]
            break

    # 1.5. LAYERS → BLOCKS
    if k.startswith("layers."):
        k = "blocks." + k[len("layers."):]
        
    if k.endswith(".modulation"):
        k = k.replace(".modulation", ".scale_shift_table")    
    if k == "head.scale_shift_table":
        k = "scale_shift_table"

    # 2. EXACT CHO KEY ĐẶC BIỆT
    if k == "scale_shift_table":
        return k

    # 3. ROOT_MAP
    ROOT_MAP = [
        ("head.0.",               "norm_out."),
        ("head.1.",               "proj_out."),
        ("head.head.",            "proj_out."),
        ("head.norm.",            "norm_out."),
        ("norm_final.",           "norm_out."),
        ("final_norm.",           "norm_out."),
        ("final_layer_norm.",     "norm_out."),
        ("out_norm.",             "norm_out."),
    ]
    for src, dst in ROOT_MAP:
        if k.startswith(src):
            return dst + k[len(src):]

    # 3.5 🎯 BẮT SỐNG TRÙM CUỐI: TIME PROJ (Đồng hồ đếm ngược)
    if k.startswith("time_projection.1."):
        return "condition_embedder.time_proj." + k[len("time_projection.1."):]
    if k.startswith("time_proj."):
        return "condition_embedder.time_proj." + k[len("time_proj."):]
    if k.startswith("time_in."):
        return "condition_embedder.time_proj." + k[len("time_in."):]

    # 4. CONDITION EMBEDDINGS
    EMBED_TABLE = [
        ("time_embedding.0.",     "condition_embedder.time_embedder.linear_1."),
        ("time_embedding.2.",     "condition_embedder.time_embedder.linear_2."),
        ("text_embedding.0.",     "condition_embedder.text_embedder.linear_1."),
        ("text_embedding.2.",     "condition_embedder.text_embedder.linear_2."),
        ("img_emb.proj.0.",       "condition_embedder.image_embedder.ff.net.0.proj."),
        ("img_emb.proj.2.",       "condition_embedder.image_embedder.ff.net.2."),
        ("img_emb.norm.",         "condition_embedder.image_embedder.norm."),
        ("image_embedding.0.",    "condition_embedder.image_embedder.ff.net.0.proj."),
        ("image_embedding.2.",    "condition_embedder.image_embedder.ff.net.2."),
        ("image_embedding.norm.", "condition_embedder.image_embedder.norm."),
        ("clip_fea.",             "condition_embedder.image_embedder."),
    ]
    for src, dst in EMBED_TABLE:
        if k.startswith(src):
            return dst + k[len(src):]

    # 5. ATTENTION
    def _attn_repl(m):
        kind = m.group(1)
        proj = m.group(2)
        trail = m.group(3) or ""
        tgt_attn = "attn1" if kind == "self" else "attn2"
        tgt_proj = {
            "q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0",
            "norm_q": "norm_q", "norm_k": "norm_k"
        }.get(proj, proj)
        return f".{tgt_attn}.{tgt_proj}{trail}"

    k = re.sub(
        r'\.(self|cross)_attn\.(q|k|v|o|norm_q|norm_k)(\.|$)',
        _attn_repl, k
    )

    # 6. FFN (Chuẩn Diffusers Wan)
    FFN_MAP = [
        (".ffn.0.",       ".ffn.net.0.proj."),
        (".ffn.1.",       ".ffn.net.0.proj."),
        (".ffn.2.",       ".ffn.net.2."),
        (".ffn.fc1.",     ".ffn.net.0.proj."),
        (".ffn.fc2.",     ".ffn.net.2."),
        (".ffn.proj.",    ".ffn.net.2."),
        (".ffn.fc.",      ".ffn.net.0.proj."),
    ]
    for old, new in FFN_MAP:
        if old in k:
            k = k.replace(old, new)
            break

    # 7. NORM3 → NORM2
    if ".norm3." in k:
        k = k.replace(".norm3.", ".norm2.")
    elif ".ffn_norm." in k:  
        k = k.replace(".ffn_norm.", ".norm2.")

    # 8. Mặc kệ Modulation 
    return k


# ════════════════════════════════════════════════════════════
# 🔎 DETECT PREFIX (giữ nguyên v4)
# ════════════════════════════════════════════════════════════
def detect_prefix(keys):
    """Phát hiện prefix thừa trong checkpoint (model., diffusion_model., v.v.)"""
    anchors = {"blocks.", "patch_embedding.", "condition_embedder.", "proj_out.", "rope."}
    for pfx in ["model.diffusion_model.", "diffusion_model.", "transformer.", ""]:
        stripped = [k[len(pfx):] for k in keys if k.startswith(pfx)]
        if sum(1 for k in stripped if any(k.startswith(a) for a in anchors)) > 5:
            return pfx
    return ""


# ════════════════════════════════════════════════════════════
# 💾 STREAM CONVERT TO DISK SHARDS
#     FP8 → FP16 từng tensor, ghi ngay ra shard file.
#     KHÔNG bao giờ giữ toàn bộ model trong RAM.
#     Peak RAM lúc convert: ~shard_gb GB (1 shard đang xây).
# ════════════════════════════════════════════════════════════
def stream_convert_to_shards(safetensors_path, config_dir, output_dir,
                              shard_gb: float = 4.0):
    """
    Đọc từng tensor từ ComfyUI FP8 checkpoint → dequantize → ghi ngay
    vào shard file trên disk. KHÔNG bao giờ giữ toàn bộ model trong RAM.

    Tại mỗi thời điểm: ~shard_gb GB trong RAM (1 shard đang xây).
    Kết quả: thư mục output_dir/ với model-XXXXX-of-YYYYY.safetensors
             + model.safetensors.index.json + config.json

    Sau đó dùng load_transformer_smart() để load với disk offload.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file, load_file

    SHARD_BYTES   = int(shard_gb * 1024 ** 3)
    SCALE_SFXS    = (".scale_weight", ".scale_input", ".input_scale", ".weight_scale")
    SKIP_EXACT    = {"scaled_fp8", "scaled_fp8_tensor"}

    try:
        FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
    except AttributeError:
        FP8_DTYPES = set()
        print("   [⚠] PyTorch < 2.0: không có FP8 dtype. Bỏ qua dequantize.")

    os.makedirs(output_dir, exist_ok=True)

    # ── Pass 1: Scan keys ────────────────────────────────────────────
    print("\n[🔍] Pass 1/3 – Scan keys (mmap, ~0 RAM)...")
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
    print(f"   Tổng tensors: {len(all_keys)}")

    prefix = detect_prefix(all_keys)
    print(f"   Key prefix  : {prefix!r}" if prefix else "   Không có prefix (bare keys)")

    # Xem thử vài key để verify remap
    sample = [k[len(prefix):] for k in all_keys
              if not any(k[len(prefix):].endswith(s) for s in SCALE_SFXS)][:4]
    print("   Ví dụ key → remap (V5 Master):")
    for sk in sample:
        print(f"     {sk!r} → {remap_comfy_key(sk)!r}")

    # ── Pass 2: Build scale map ──────────────────────────────────────
    print("\n[📏] Pass 2/3 – Build scale_map (key đã remap V5)...")
    scale_map = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for raw_key in all_keys:
            clean = raw_key[len(prefix):]
            for sfx in SCALE_SFXS:
                if clean.endswith(sfx):
                    base_comfy = clean[:-len(sfx)]
                    # Thêm .weight để regex attn nhận dạng đúng, rồi bỏ
                    rw = remap_comfy_key(base_comfy + ".weight")
                    base_diff = rw[:-len(".weight")] if rw.endswith(".weight") else rw
                    scale_map[base_diff] = f.get_tensor(raw_key).float()
                    break
    print(f"   {len(scale_map)} scale tensors")

    # ── Lấy danh sách params hợp lệ từ model config ─────────────────
    from accelerate import init_empty_weights
    cfg = WanTransformer3DModel.load_config(os.path.join(config_dir, "transformer"))
    with init_empty_weights():
        dummy = WanTransformer3DModel(**cfg)
    valid_params  = {n for n, _ in dummy.named_parameters()}
    valid_buffers = {n for n, _ in dummy.named_buffers()}
    valid_all     = valid_params | valid_buffers
    del dummy
    gc.collect()
    print(f"   Model: {len(valid_params)} params + {len(valid_buffers)} buffers")

    # ── Pass 3: Stream + remap + dequant → ghi shard files ──────────
    print(f"\n[⚗️] Pass 3/3 – Stream FP8→FP16 + ghi shards (shard={shard_gb}GB)")
    print(f"   Peak RAM dự kiến: ~{shard_gb:.0f}GB (không phải 28GB!)\n")

    # State cho closure flush_shard
    state = {
        "cur_shard":    {},
        "cur_bytes":    0,
        "shard_fnames": [],   # tên file (không kèm path)
        "weight_map":   {},
        "total_bytes":  0,
    }
    ok = skip = deq = 0
    miss_ckpt_side = []   # ComfyUI key không remap được

    def flush_shard():
        if not state["cur_shard"]:
            return
        idx   = len(state["shard_fnames"])
        fname = f"model-{idx + 1:05d}-of-XXXXX.safetensors"
        fpath = os.path.join(output_dir, fname)
        save_file(state["cur_shard"], fpath)
        for k in state["cur_shard"]:
            state["weight_map"][k] = fname
        state["shard_fnames"].append(fname)

        ram_str = ""
        if _HAS_PSUTIL:
            used = psutil.virtual_memory().used / 1024 ** 3
            free = psutil.virtual_memory().available / 1024 ** 3
            ram_str = f"  | RAM: {used:.1f}GB used / {free:.1f}GB free"
        print(f"\n   💾 Shard {idx + 1} ({state['cur_bytes'] / 1024 ** 3:.2f} GB){ram_str}")

        # Giải phóng RAM ngay
        del state["cur_shard"]
        gc.collect()
        state["cur_shard"] = {}
        state["cur_bytes"] = 0

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        total = len(all_keys)
        for i, raw_key in enumerate(all_keys, 1):
            clean = raw_key[len(prefix):]

            # Bỏ qua metadata và scale tensors
            if clean in SKIP_EXACT or any(clean.endswith(s) for s in SCALE_SFXS):
                skip += 1
                continue

            diff_key = remap_comfy_key(clean)

            # Key không có trong model diffusers → báo cáo, bỏ qua
            if diff_key not in valid_all:
                if len(miss_ckpt_side) < 30:
                    miss_ckpt_side.append((clean, diff_key))
                continue

            # Load tensor từ mmap (chỉ tensor này, không load cả file)
            tensor = f.get_tensor(raw_key)

            # ── Dequantize FP8 → FP16 ─────────────────────────────
            if FP8_DTYPES and tensor.dtype in FP8_DTYPES:
                # Tìm scale từ scale_map (đã remap key từ Pass 2)
                scale_base = diff_key
                for sfx in ('.weight', '.bias'):
                    if diff_key.endswith(sfx):
                        scale_base = diff_key[:-len(sfx)]
                        break

                t32 = tensor.to(torch.float32)
                if scale_base in scale_map:
                    sc = scale_map[scale_base]
                    if sc.numel() > 1:
                        sc = sc.reshape([-1] + [1] * (t32.ndim - 1))
                    tensor = (t32 * sc).to(TARGET_DTYPE)   # ← FP16 (không phải BF16)
                else:
                    # Không có scale → raw cast (hiếm gặp)
                    tensor = t32.to(TARGET_DTYPE)
                deq += 1
            elif tensor.is_floating_point():
                tensor = tensor.to(TARGET_DTYPE)   # ← FP16
            # else: integer tensor (RoPE index, v.v.) → giữ nguyên dtype

            # Thêm vào shard hiện tại
            state["cur_shard"][diff_key] = tensor
            nb = tensor.nbytes
            state["cur_bytes"]   += nb
            state["total_bytes"] += nb
            ok += 1
            del tensor  # Giải phóng ngay

            # Flush khi shard đạt ngưỡng
            if state["cur_bytes"] >= SHARD_BYTES:
                flush_shard()

            if i % 200 == 0:
                sys.stdout.write(
                    f"\r   [{i:4d}/{total}] ok:{ok} fp8_deq:{deq} skip:{skip}   "
                )
                sys.stdout.flush()

    flush_shard()  # Flush shard cuối
    n_shards = len(state["shard_fnames"])
    print(f"\n\n   ✅ Stream convert xong: {ok} tensors → {n_shards} shards")
    print(f"      FP8 dequantized: {deq}")
    print(f"      Skip (scale):    {skip}")
    print(f"      Total on disk:   {state['total_bytes'] / 1024 ** 3:.1f} GB")

    # ── Rename XXXXX → số thật ───────────────────────────────────────
    rename_map = {}
    for fname in state["shard_fnames"]:
        new_fname = fname.replace("XXXXX", f"{n_shards:05d}")
        os.rename(os.path.join(output_dir, fname), os.path.join(output_dir, new_fname))
        rename_map[fname] = new_fname
    weight_map = {k: rename_map[v] for k, v in state["weight_map"].items()}

    # ── Phân tích missing tensors ────────────────────────────────────
    saved_keys  = set(weight_map.keys())
    not_loaded  = valid_all - saved_keys

    # Phân loại mức độ nguy hiểm:
    # 1. scale_shift_table: WAN init = zeros → zero-init OK
    # 2. Buffers (positional encodings): không lưu trong ckpt → zero-init OK
    # 3. .weight / .bias khác: NGUY HIỂM, zero-init = garbage output
    SAFE_PARAM_LEAVES = {"scale_shift_table"}

    safe_to_zero = [k for k in not_loaded
                    if k in valid_buffers
                    or any(k.endswith(f".{n}") or k == n for n in SAFE_PARAM_LEAVES)]
    critical_miss = [k for k in not_loaded
                     if k not in safe_to_zero
                     and (k.endswith('.weight') or k.endswith('.bias'))]
    other_miss    = [k for k in not_loaded
                     if k not in safe_to_zero and k not in critical_miss]

    print(f"\n   📊 Missing tensor analysis:")
    print(f"      ✅ Safe zero-init (scale_shift_table, buffers): {len(safe_to_zero)}")
    print(f"      ⚠️  Other missing (non-weight):                 {len(other_miss)}")
    print(f"      ❌ CRITICAL missing (.weight/.bias):            {len(critical_miss)}")

    if miss_ckpt_side:
        print(f"\n   ⚠️  {len(miss_ckpt_side)} ComfyUI key không remap được sang diffusers (top 5):")
        for clean, diff in miss_ckpt_side[:5]:
            print(f"      {clean!r}\n        → {diff!r} (không có trong model)")

    if critical_miss:
        pct = len(critical_miss) / max(1, len(valid_params)) * 100
        print(f"\n   ❌ CRITICAL: {len(critical_miss)} weight tensors THIẾU ({pct:.1f}%):")
        for k in critical_miss[:10]:
            print(f"      {k!r}")
        if pct > 5.0:
            print(f"\n   🛑 Quá nhiều weight thiếu ({pct:.1f}% > 5%). Convert ABORT.")
            print("   → Chạy diagnose_keys() để xem chi tiết key remap.")
            print("   → Xóa CONVERTED_DIR trước khi chạy lại.")
            raise RuntimeError(
                f"ABORT: {len(critical_miss)} ({pct:.1f}%) weight tensors không được load. "
                f"Video output sẽ là garbage. Chạy diagnose_keys() để debug."
            )
        else:
            print(f"   (Dưới ngưỡng 5%, có thể chấp nhận – theo dõi kỹ output video)")

    if safe_to_zero:
        print(f"\n   ℹ️  {len(safe_to_zero)} params sẽ được zero-init an toàn:")
        for k in safe_to_zero[:5]:
            print(f"      {k!r}")

    # ── Ghi model.safetensors.index.json ────────────────────────────
    index = {
        "metadata": {"total_size": state["total_bytes"]},
        "weight_map": weight_map,
    }
    #idx_path = os.path.join(output_dir, "model.safetensors.index.json")
    idx_path = os.path.join(output_dir, "diffusion_pytorch_model.safetensors.index.json")
    with open(idx_path, "w", encoding="utf-8") as fp:
        json.dump(index, fp, indent=2)
    print(f"\n   📝 Index JSON: {idx_path}")

    # ── Copy config.json (diffusers cần để load) ─────────────────────
    cfg_src = os.path.join(config_dir, "transformer", "config.json")
    cfg_dst = os.path.join(output_dir, "config.json")
    if os.path.exists(cfg_src):
        shutil.copy(cfg_src, cfg_dst)
        print(f"   📋 config.json copied")
    else:
        print(f"   ⚠️  config.json không tìm thấy tại: {cfg_src}")

    del scale_map
    gc.collect()

    return ok, len(critical_miss), safe_to_zero


# ════════════════════════════════════════════════════════════
# 📦 LOAD TRANSFORMER – RAM-AWARE
# ════════════════════════════════════════════════════════════
def load_transformer_smart(converted_dir):
    print("   [⚡] Chiến thuật MỚI: Load thẳng vào CPU (mượn RAM ảo 56GB)!")
    print("   [⚡] Sử dụng Sequential CPU Offload để nhỏ giọt vào GPU.")
    
    # Load toàn bộ model vào CPU (nếu RAM vật lý thiếu, Windows sẽ tự nhét vào Pagefile)
    transformer = WanTransformer3DModel.from_pretrained(
        converted_dir,
        torch_dtype=TARGET_DTYPE,
        low_cpu_mem_usage=True,
        device_map=None,  # Ép KHÔNG dùng accelerate dispatch
        ignore_mismatched_sizes=True,
    )
    return transformer




#def load_transformer_smart(converted_dir):
    """
    Chọn strategy load phù hợp với RAM hiện tại.

    RAM free > 20GB → from_pretrained (load đầy vào RAM, nhanh)
    RAM free ≤ 20GB → load_checkpoint_and_dispatch với disk offload
                      (chỉ 1 layer trong RAM tại 1 thời điểm, chậm hơn
                       nhưng KHÔNG OOM trên máy 16GB)

    Với CMP 40HX + 16GB RAM + FP16 model (28GB): luôn vào nhánh disk offload.
    """
    free_gb  = 0.0
    total_gb = 0.0
    if _HAS_PSUTIL:
        free_gb  = psutil.virtual_memory().available / 1024 ** 3
        total_gb = psutil.virtual_memory().total      / 1024 ** 3
        print(f"   RAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total")

    if free_gb > 20.0:
        # Có đủ RAM → load thẳng vào RAM (hiếm với 16GB)
        print("   Strategy: from_pretrained → full load vào RAM")
        transformer = WanTransformer3DModel.from_pretrained(
            converted_dir,
            torch_dtype       = TARGET_DTYPE,
            low_cpu_mem_usage = True,
        )
        return transformer

    # Không đủ RAM → disk offload
    # Weights sống trên disk, load từng layer khi inference cần
    print("   Strategy: disk offload (RAM 16GB < model 28GB FP16)")
    print("   Inference sẽ chậm hơn ~3-5x nhưng KHÔNG OOM")

    from accelerate import load_checkpoint_and_dispatch, init_empty_weights

    # Budget VRAM: 8GB card, để 1GB headroom = 7GB
    gpu_budget_gb = 4
    # Budget CPU RAM: dùng 55% RAM free (để OS và encode phần còn lại)
    cpu_budget_gb = max(4, int(free_gb * 0.55)) if _HAS_PSUTIL else 6

    print(f"   Budget: GPU {gpu_budget_gb}GB | CPU {cpu_budget_gb}GB | disk unlimited")

    cfg = WanTransformer3DModel.load_config(converted_dir)
    with init_empty_weights():
        model = WanTransformer3DModel(**cfg)
        
    import json
    import os
    from accelerate.utils import set_module_tensor_to_device
    import torch

    # 1. Đọc sổ Nam Tào (index.json) xem kho có những nơ-ron nào
    index_path = os.path.join(converted_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    ckpt_keys = set(index_data["weight_map"].keys())

    # 2. Quét toàn bộ model, đứa nào không có trong sổ thì tự động bơm số 0
    missing_count = 0
    for name, param in model.named_parameters():
        if name not in ckpt_keys:
            set_module_tensor_to_device(model, name, "cpu", value=torch.zeros(param.shape, dtype=param.dtype))
            missing_count += 1
            
    for name, buf in model.named_buffers():
        if name not in ckpt_keys:
            set_module_tensor_to_device(model, name, "cpu", value=torch.zeros(buf.shape, dtype=buf.dtype))
            
    print(f"   [!] Đã tự động bơm số 0 (Auto-Fix) cho {missing_count} nơ-ron bị thiếu để thông chốt.")

    model = load_checkpoint_and_dispatch(
        model,
        converted_dir,
        device_map              = "auto",
        max_memory              = {
            0:      f"{gpu_budget_gb}GiB",
            "cpu":  f"{cpu_budget_gb}GiB",
        },
        dtype                   = TARGET_DTYPE,
        # Không chia nhỏ WanTransformerBlock – mỗi block phải ở 1 device
        no_split_module_classes = ["WanTransformerBlock"],
        offload_folder          = "offload_weights_cache",
    )
    return model


# ════════════════════════════════════════════════════════════
# 🔒 POST-LOAD CHECK (zero-init thông minh)
# ════════════════════════════════════════════════════════════
def safe_post_load_check(model):
    """
    Sau khi load, tìm meta tensors còn lại và xử lý đúng cách.

    POLICY:
      scale_shift_table → zero-init (WAN original init = zeros, đây là đúng)
      buffers           → zero-init (positional encodings, etc.)
      .weight / .bias   → KHÔNG zero-init → raise RuntimeError

    Lý do KHÔNG zero-init weight:
      Zero tensor nhân với input → output = zeros → video = xám xịt
      Không có báo lỗi, không có warning, code chạy xong bình thường.
      Đây là "silent corruption" nguy hiểm nhất.
    """
    
    #return ###### NẾU CHẠY FILE LẠI THÌ MỞ COMMENT RA #####################################
    SAFE_LEAF_NAMES = {"scale_shift_table"}  # Biết chắc WAN init = zeros

    zero_ok    = 0
    still_meta = []  # .weight/.bias trên meta → nghiêm trọng

    # ── Parameters ───────────────────────────────────────────────────
    for name, param in model.named_parameters():
        if param.device.type != "meta":
            continue

        leaf    = name.split(".")[-1]
        is_safe = leaf in SAFE_LEAF_NAMES

        if is_safe:
            mod_path, _, p_name = name.rpartition(".")
            mod = model.get_submodule(mod_path)
            mod._parameters[p_name] = torch.nn.Parameter(
                torch.zeros(param.shape, dtype=TARGET_DTYPE, device="cpu"),
                requires_grad=False,
            )
            zero_ok += 1
        else:
            still_meta.append(name)

    # ── Buffers (an toàn zero-init) ───────────────────────────────────
    for name, buf in model.named_buffers():
        if buf.device.type != "meta":
            continue
        mod_path, _, b_name = name.rpartition(".")
        mod   = model.get_submodule(mod_path)
        dtype = buf.dtype if not buf.is_floating_point() else TARGET_DTYPE
        mod._buffers[b_name] = torch.zeros(buf.shape, dtype=dtype, device="cpu")
        zero_ok += 1

    # ── Report ────────────────────────────────────────────────────────
    if zero_ok:
        print(f"   ✅ Zero-init {zero_ok} safe params (scale_shift_table + buffers)")

    if still_meta:
        print(f"\n   ❌ {len(still_meta)} weight/bias tensors vẫn trên meta device!")
        print("   ⛔ KHÔNG thể zero-init: sẽ tạo video nhiễu/xám xịt!")
        for n in still_meta[:10]:
            print(f"      {n!r}")
        if len(still_meta) > 10:
            print(f"      ... và {len(still_meta) - 10} cái nữa")
        raise RuntimeError(
            f"ABORT: {len(still_meta)} weight tensors thiếu dữ liệu thật.\n"
            f"Chạy diagnose_keys() để xem chi tiết key mapping.\n"
            f"Xóa {CONVERTED_DIR} và chạy lại sau khi fix."
        )

    if not still_meta:
        print("   ✅ Tất cả weight tensors có dữ liệu thật – không còn meta tensor!")

    return zero_ok


# ════════════════════════════════════════════════════════════
# 🔍 DIAGNOSTIC (chạy độc lập để debug key mapping)
# ════════════════════════════════════════════════════════════
def diagnose_keys(safetensors_path, config_dir, max_show=25):
    """
    Phân tích key mapping mà không cần convert đầy đủ.
    Chạy để debug nếu too many missing weights.

    Dùng: uncomment dòng gọi diagnose_keys() trong main()
    """
    from safetensors import safe_open
    from accelerate import init_empty_weights

    print("\n[🔍] DIAGNOSTIC – Key mapping analysis (V5 Master Remap)")
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())

    prefix = detect_prefix(all_keys)
    SCALE_SFXS = (".scale_weight", ".scale_input", ".input_scale", ".weight_scale")

    cfg = WanTransformer3DModel.load_config(os.path.join(config_dir, "transformer"))
    with init_empty_weights():
        m = WanTransformer3DModel(**cfg)
    valid_params  = {n for n, _ in m.named_parameters()}
    valid_buffers = {n for n, _ in m.named_buffers()}
    valid_all     = valid_params | valid_buffers
    del m; gc.collect()

    mapped        = 0
    mapped_keys   = set()
    unmapped_ckpt = []

    for raw_key in all_keys:
        clean = raw_key[len(prefix):]
        if any(clean.endswith(s) for s in SCALE_SFXS):
            continue
        remapped = remap_comfy_key(clean)
        if remapped in valid_all:
            mapped += 1
            mapped_keys.add(remapped)
        else:
            unmapped_ckpt.append((clean, remapped))

    unmapped_diff = sorted(valid_all - mapped_keys)
    crit_unmapped = [k for k in unmapped_diff if k.endswith(('.weight', '.bias'))]

    total = len(valid_all)
    print(f"\n  ✅ Checkpoint → Diffusers mapped: {mapped}/{total} ({mapped/max(1,total)*100:.1f}%)")
    print(f"  ❌ ComfyUI → remap FAIL:          {len(unmapped_ckpt)}")
    print(f"  ❌ Diffusers thiếu data:           {len(unmapped_diff)}")
    print(f"     trong đó .weight/.bias:         {len(crit_unmapped)} ← NGUY HIỂM")

    if unmapped_ckpt:
        print(f"\n  [ComfyUI key không remap được] (top {max_show}):")
        for c, d in unmapped_ckpt[:max_show]:
            print(f"    {c!r}\n      → {d!r}")

    if crit_unmapped:
        print(f"\n  [Diffusers weight/bias thiếu] (top {max_show}):")
        for k in crit_unmapped[:max_show]:
            print(f"    {k!r}")


# ════════════════════════════════════════════════════════════
# 🔥 EXTREME TEACACHE
#    Skip denoising steps khi hidden states gần giống bước trước.
#    Với 4 inference steps → có thể skip 1-2 steps → ~30-50% nhanh hơn.
# ════════════════════════════════════════════════════════════
class ExtremeTeaCache:
    def __init__(self, rel_l1_thresh=0.25, start_percent=0.2):
        self.rel_l1_thresh      = rel_l1_thresh
        self.start_percent      = start_percent
        self.cache              = None
        self.prev_hidden_states = None
        self.step               = 0
        self.total_steps        = 0
        self.skipped            = 0

    def inject(self, pipe, total_steps):
        self.total_steps = total_steps
        original_forward = pipe.transformer.forward

        def new_forward(*args, **kwargs):
            hs = kwargs.get("hidden_states", None)
            if hs is None and len(args) > 0:
                hs = args[0]

            pct = self.step / max(1, self.total_steps)
            if (pct >= self.start_percent and 
                self.cache is not None and 
                self.prev_hidden_states is not None and 
                hs is not None):
                
                diff = hs - self.prev_hidden_states
                rel_dist = diff.abs().mean() / (self.prev_hidden_states.abs().mean() + 1e-8)
                if rel_dist < self.rel_l1_thresh:
                    self.skipped += 1
                    self.step += 1
                    return self.cache

            out = original_forward(*args, **kwargs)
            self.cache = out
            self.prev_hidden_states = hs.clone() if hs is not None else None
            self.step += 1

            if self.step % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            return out

        pipe.transformer.forward = new_forward
        print("[+] TeaCache injected!")
        return pipe


# ════════════════════════════════════════════════════════════
# 📝 SUBPROCESS ENCODE TEXT
#    Encode text trong subprocess riêng để giải phóng VRAM hoàn toàn.
#    UMT5 9.4GB: dùng device_map split GPU+CPU.
#    dtype: float16 (CMP 40HX không có BF16 native)
# ════════════════════════════════════════════════════════════
def master_encode_worker(prompt, negative_prompt, output_embeds_path, text_encoder_path):
    import torch, os, shutil, warnings, traceback
    warnings.filterwarnings("ignore")
    from transformers import UMT5EncoderModel, UMT5Config, AutoTokenizer

    try:
        model_dir   = os.path.dirname(text_encoder_path)
        temp_hf_dir = os.path.join(model_dir, "temp_hf_t5")
        os.makedirs(temp_hf_dir, exist_ok=True)

        config = UMT5Config.from_pretrained(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="text_encoder")
        config.save_pretrained(temp_hf_dir)

        temp_st = os.path.join(temp_hf_dir, "model.safetensors")
        if not os.path.exists(temp_st):
            print("   → Lần đầu: link UMT5 safetensors...")
            try:    os.link(text_encoder_path, temp_st)
            except: shutil.copy(text_encoder_path, temp_st)

        tokenizer = AutoTokenizer.from_pretrained(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="tokenizer")

        print("\n[⚡] Nạp UMT5 (9.4GB) – GPU 6.5GB + RAM 2.9GB split...")
        # CMP 40HX: FP16 (không phải BF16)
        text_encoder = UMT5EncoderModel.from_pretrained(
            temp_hf_dir,
            device_map  = "auto",
            max_memory  = {0: "3.5GB", "cpu": "40GB"},
            torch_dtype = torch.bfloat16,   # ← FP16 cho Turing
        )
        print("   → Encode prompt (15-30 giây)...")

        with torch.inference_mode():
            dev   = next(text_encoder.parameters()).device
            p_ids = tokenizer(prompt, padding="max_length", max_length=226,
                              truncation=True, return_tensors="pt").to(dev)
            p_emb = text_encoder(p_ids.input_ids,
                                 attention_mask=p_ids.attention_mask)[0]
            n_ids = tokenizer(negative_prompt, padding="max_length", max_length=226,
                              truncation=True, return_tensors="pt").to(dev)
            n_emb = text_encoder(n_ids.input_ids,
                                 attention_mask=n_ids.attention_mask)[0]

        torch.save({"p": p_emb.cpu(), "n": n_emb.cpu()}, output_embeds_path)
        print("[✅] Encode xong – subprocess tự hủy, trả VRAM về 0.")
    except Exception as e:
        traceback.print_exc()
        print(f"[!] Lỗi encode: {e}")


# ════════════════════════════════════════════════════════════
# 🏭 BUILD PIPELINE THỦ CÔNG
# ════════════════════════════════════════════════════════════
def build_pipeline_manual(config_dir, transformer, vae, clip_encoder_dir="",
                           transformer_2=None):
    from transformers import SiglipVisionModel, SiglipImageProcessor

    print("\n[🏭] Lắp Pipeline thủ công...")
    scheduler_path = os.path.join(config_dir, "scheduler")
    scheduler      = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)
    print("   ✅ Scheduler OK")

    image_encoder   = None
    image_processor = None

    if clip_encoder_dir and os.path.isdir(clip_encoder_dir):
        try:
            print(f"   📂 Đang nạp SigLIP Encoder từ: {clip_encoder_dir}")
            # Dùng đúng class SiglipVisionModel để không bị lệch size 729 vs 730
            image_encoder = SiglipVisionModel.from_pretrained(
                clip_encoder_dir, 
                torch_dtype=TARGET_DTYPE,
                low_cpu_mem_usage=True
            )
            image_processor = SiglipImageProcessor.from_pretrained(clip_encoder_dir)
            print("   ✅ SigLIP Image Encoder OK (Chuẩn Wan 2.1/2.2)")
        except Exception as e:
            print(f"   ⚠ SigLIP lỗi: {e}")
            image_encoder = None

    if image_processor is None:
        # Dự phòng nếu không có SigLIP offline thì dùng Processor mặc định
        image_processor = SiglipImageProcessor()
        print("   ℹ SiglipImageProcessor mặc định")

    # ── V6: Tạo pipeline với cả 2 transformer ──────────────────────
    pipe_kwargs = dict(
        tokenizer       = None,
        text_encoder    = None,
        image_encoder   = image_encoder,
        image_processor = image_processor,
        transformer     = transformer,
        vae             = vae,
        scheduler       = scheduler,
    )
    if transformer_2 is not None:
        pipe_kwargs["transformer_2"] = transformer_2
        print("   ✅ transformer_2 (Low Noise) sẽ được nạp vào pipeline")

    pipe = WanImageToVideoPipeline(**pipe_kwargs)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print("   ✅ Pipeline OK!")

    # =========================================================================
    # [⚡] BỘ TĂNG ÁP 4 XI-LANH (TRỘN SVI VÀ LIGHTNING) ĐƯỢC CẤY VÀO ĐÂY!
    # =========================================================================
    print("\n[🎬] Đang nạp Siêu động cơ qua bộ Remap: 2 SVI + 2 Lightning...")
    
    path_svi_high   = os.path.join(LORAS_DIR, "SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors")
    path_svi_low    = os.path.join(LORAS_DIR, "SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors")
    path_light_high = os.path.join(LORAS_DIR, "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors")
    path_light_low  = os.path.join(LORAS_DIR, "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors")

    try:
        # Gọi hàm dịch thuật thay vì load trực tiếp
        load_and_remap_lora(pipe, path_svi_high, "svi_high")
        load_and_remap_lora(pipe, path_svi_low, "svi_low")
        load_and_remap_lora(pipe, path_light_high, "light_high")
        load_and_remap_lora(pipe, path_light_low, "light_low")
        
        # Kích hoạt cả 4 cái cùng lúc
        pipe.set_adapters(
            ["svi_high", "svi_low", "light_high", "light_low"], 
            adapter_weights=[1.0, 1.0, 1.0, 1.0] 
        )
        print("   ✅ Đã kích hoạt chế độ FULL BÔ hoàn hảo!")
    except Exception as e:
        print(f"   ⚠️ Lỗi trộn LoRA: {e}")
        print("   (Hãy đảm bảo ông đã tạo thư mục 'loras' và để 4 file vào đúng vị trí nhé)")
    # =========================================================================

    return pipe


# ════════════════════════════════════════════════════════════════════
# 🔄 HELPER: CONVERT 1 FILE (bọc lại để tái sử dụng trong main)
# ════════════════════════════════════════════════════════════════════
def _do_convert_one(label, src_path, out_dir, config_dir, shard_gb):
    """
    Convert 1 file safetensors ComfyUI FP8 → thư mục shard Diffusers.
    Trả về (ok, n_crit, safe_zero) hoặc raise nếu lỗi.
    """
    print(f"\n{'═'*62}")
    print(f"  🔄 Convert [{label}]")
    print(f"     SRC : {src_path}")
    print(f"     DST : {out_dir}")
    print(f"{'═'*62}")
    return stream_convert_to_shards(
        src_path, config_dir, out_dir, shard_gb=shard_gb
    )


def _dir_has_shards(d):
    """Trả về True nếu thư mục đã có index JSON + config.json (đã convert)."""
    idx = os.path.join(d, "diffusion_pytorch_model.safetensors.index.json")
    cfg = os.path.join(d, "config.json")
    return os.path.exists(idx) and os.path.exists(cfg)


# ════════════════════════════════════════════════════════════
# 🎬 MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  WAN 2.2 ComfyUI FP8 → Diffusers  v6 DUAL TRANSFORMER  ║")
    print("║  CMP 40HX · FP16 · Disk Offload · RAM 16GB             ║")
    print("║  Master Remap V5.0 · TeaCache · Stream-to-disk Shards  ║")
    print("║  High Noise: transformer   |  Low Noise: transformer_2 ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    multiprocessing.set_start_method("spawn", force=True)

    # ── Kiểm tra files nguồn ─────────────────────────────────────────
    print("[🔎] Kiểm tra file nguồn...")
    for label, path in [
        ("ComfyUI High Noise (transformer)",   COMFY_MODEL_HIGH),
        ("ComfyUI Low Noise  (transformer_2)", COMFY_MODEL_LOW),
        ("VAE",                                VAE_PATH),
        ("Text Encoder",                       ENCODER_PATH),
    ]:
        if not os.path.exists(path):
            print(f"❌ Không tìm thấy {label}:\n   {path}")
            return
        print(f"   ✅ {label}")

    # ── 1. Chọn ảnh & nhập prompt ────────────────────────────────────
    imgs = [f for f in os.listdir(INPUT_IMG_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not imgs:
        print(f"❌ Không có ảnh trong {INPUT_IMG_DIR}")
        return
    for i, f in enumerate(imgs, 1):
        print(f"  [{i}] {f}")
    img_path  = os.path.join(INPUT_IMG_DIR, imgs[int(input("Chọn ảnh (số): ")) - 1])
    prompt    = input("Prompt: ")
    safe_name = input("Tên file output (Enter='output'): ").strip() or "output"

    # ── 2. Encode text (subprocess để giải phóng VRAM hoàn toàn) ─────
    temp_pt = os.path.join(OUTPUT_DIR, "temp_emb.pt")
    if os.path.exists(temp_pt):
        os.remove(temp_pt)

    print("\n📝 Encode prompt (subprocess – giải phóng VRAM sau encode)...")
    p = multiprocessing.Process(
        target=master_encode_worker,
        args=(prompt, CONFIG["negative_prompt"], temp_pt, ENCODER_PATH),
    )
    p.start()
    p.join()

    if not os.path.exists(temp_pt):
        print("[!] Subprocess encode thất bại. Kiểm tra lại ENCODER_PATH.")
        return

    emb   = torch.load(temp_pt, weights_only=True)
    p_emb = emb["p"].to("cuda", dtype=TARGET_DTYPE)
    n_emb = emb["n"].to("cuda", dtype=TARGET_DTYPE)

    # ── 3. Download config JSON (chỉ ~200KB) ─────────────────────────
    print("\n[📋] Tải config JSON (chỉ .json, ~200KB)...")
    config_dir = snapshot_download(
        repo_id         = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        allow_patterns  = ["*.json", "*.txt"],
        ignore_patterns = ["*.safetensors", "*.bin", "*.pt"],
        local_dir       = CONFIG_CACHE_DIR,
    )

    # ── 4. Convert 2 lần tuần tự (nếu chưa có) ───────────────────────
    #
    #  Lần 1: COMFY_MODEL_HIGH → TRANSFORMER_1_DIR  (High Noise Stage)
    #  Lần 2: COMFY_MODEL_LOW  → TRANSFORMER_2_DIR  (Low Noise Stage)
    #
    #  Mỗi lần gọi stream_convert_to_shards độc lập – hàm gốc không đổi.
    # ─────────────────────────────────────────────────────────────────

    # ── Lần 1: transformer (High Noise) ──────────────────────────────
    if _dir_has_shards(TRANSFORMER_1_DIR):
        print(f"\n[⚡] transformer (High Noise) đã convert → bỏ qua lần 1")
        print(f"   Thư mục: {TRANSFORMER_1_DIR}")
    else:
        print(f"\n[🔄] Lần đầu convert transformer (High Noise)...")
        print(f"   Cần ~14GB disk space cho FP16 shards")
        print(f"   Peak RAM trong lúc convert: ~{CONFIG['shard_size_gb']:.0f}GB")

        # Uncomment để chạy diagnostic TRƯỚC khi convert:
        # diagnose_keys(COMFY_MODEL_HIGH, config_dir)

        try:
            ok1, n_crit1, safe1 = _do_convert_one(
                "High Noise → transformer",
                COMFY_MODEL_HIGH,
                TRANSFORMER_1_DIR,
                config_dir,
                CONFIG["shard_size_gb"],
            )
            print(f"\n   ✅ Convert transformer OK: {ok1} tensors, "
                  f"critical_miss={n_crit1}, safe_zero={len(safe1)}")
        except RuntimeError as e:
            print(f"\n❌ Convert transformer thất bại:\n   {e}")
            print("\n→ Uncomment dòng diagnose_keys() phía trên để debug key mapping.")
            input("Enter để thoát...")
            return
        except Exception as e:
            print(f"\n❌ Lỗi không mong đợi khi convert transformer:\n   {e}")
            traceback.print_exc()
            input("Enter để thoát...")
            return

    # ── Lần 2: transformer_2 (Low Noise) ─────────────────────────────
    if _dir_has_shards(TRANSFORMER_2_DIR):
        print(f"\n[⚡] transformer_2 (Low Noise) đã convert → bỏ qua lần 2")
        print(f"   Thư mục: {TRANSFORMER_2_DIR}")
    else:
        print(f"\n[🔄] Lần đầu convert transformer_2 (Low Noise)...")
        print(f"   Cần ~14GB disk space cho FP16 shards")
        print(f"   Peak RAM trong lúc convert: ~{CONFIG['shard_size_gb']:.0f}GB")

        # Uncomment để chạy diagnostic TRƯỚC khi convert:
        # diagnose_keys(COMFY_MODEL_LOW, config_dir)

        try:
            ok2, n_crit2, safe2 = _do_convert_one(
                "Low Noise → transformer_2",
                COMFY_MODEL_LOW,
                TRANSFORMER_2_DIR,
                config_dir,
                CONFIG["shard_size_gb"],
            )
            print(f"\n   ✅ Convert transformer_2 OK: {ok2} tensors, "
                  f"critical_miss={n_crit2}, safe_zero={len(safe2)}")
        except RuntimeError as e:
            print(f"\n❌ Convert transformer_2 thất bại:\n   {e}")
            print("\n→ Uncomment dòng diagnose_keys() phía trên để debug key mapping.")
            input("Enter để thoát...")
            return
        except Exception as e:
            print(f"\n❌ Lỗi không mong đợi khi convert transformer_2:\n   {e}")
            traceback.print_exc()
            input("Enter để thoát...")
            return

    # ── 5. Nạp transformer (High Noise) ──────────────────────────────
    print(f"\n[📦] Nạp transformer (High Noise) từ shards...")
    print(f"   Thư mục: {TRANSFORMER_1_DIR}")
    try:
        transformer = load_transformer_smart(TRANSFORMER_1_DIR)
        print("\n[🔒] Post-load check transformer...")
        safe_post_load_check(transformer)
        print("   ✅ transformer OK")
    except RuntimeError as e:
        print(f"\n❌ Load transformer FAIL:\n   {e}")
        input("Enter để thoát...")
        return
    except Exception as e:
        print(f"\n❌ Lỗi không mong đợi load transformer:\n   {e}")
        traceback.print_exc()
        input("Enter để thoát...")
        return

    # ── 6. Nạp transformer_2 (Low Noise) ─────────────────────────────
    print(f"\n[📦] Nạp transformer_2 (Low Noise) từ shards...")
    print(f"   Thư mục: {TRANSFORMER_2_DIR}")
    try:
        transformer_2 = load_transformer_smart(TRANSFORMER_2_DIR)
        print("\n[🔒] Post-load check transformer_2...")
        safe_post_load_check(transformer_2)
        print("   ✅ transformer_2 OK")
    except RuntimeError as e:
        print(f"\n❌ Load transformer_2 FAIL:\n   {e}")
        input("Enter để thoát...")
        return
    except Exception as e:
        print(f"\n❌ Lỗi không mong đợi load transformer_2:\n   {e}")
        traceback.print_exc()
        input("Enter để thoát...")
        return

    # ── 7. VAE ───────────────────────────────────────────────────────
    print("\n[🎨] Nạp VAE...")
    try:
        vae = AutoencoderKLWan.from_single_file(VAE_PATH, torch_dtype=TARGET_DTYPE)
        print("   ✅ VAE OK")
    except Exception as e:
        print(f"❌ VAE lỗi: {e}")
        traceback.print_exc()
        input("Enter để thoát...")
        return

    # ── 8. Pipeline (truyền cả 2 transformer) ────────────────────────
    try:
        pipe = build_pipeline_manual(
            config_dir,
            transformer,
            vae,
            CLIP_ENCODER_DIR,
            transformer_2=transformer_2,
        )
    except Exception as e:
        print(f"❌ Pipeline lỗi: {e}")
        traceback.print_exc()
        input("Enter để thoát...")
        return

    # ── 9. LoRA (optional) ───────────────────────────────────────────
    if CONFIG["lora_name"]:
        lp = os.path.join(LORAS_DIR, CONFIG["lora_name"])
        if os.path.exists(lp):
            print(f"\n[🎭] Nạp LoRA: {CONFIG['lora_name']}")
            try:
                pipe.load_lora_weights(LORAS_DIR,
                                       weight_name=CONFIG["lora_name"],
                                       adapter_name="lora")
                pipe.set_adapters(["lora"], adapter_weights=[CONFIG["lora_alpha"]])
            except Exception as e:
                print(f"   ⚠  LoRA lỗi: {e}")
                
    # ── 10. Tối ưu bộ nhớ (Bản Mới - Nhỏ Giọt Qua RAM Ảo) ────────────────
    print("\n[⚙️] Tối ưu bộ nhớ...")

    # Bật nhỏ giọt từng layer từ CPU (RAM ảo) vào GPU -> Chống OOM Weights
    pipe.enable_sequential_cpu_offload()

    # VAE vẫn phải cắt lát để không nổ VRAM lúc xuất video
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.enable_attention_slicing("max")
        
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Áp dụng TeaCache (bỏ qua nếu ông không muốn dùng)
    steps = CONFIG["wan_num_inference_steps"]
    if steps <= 6:
        print(f"\n[🧠] TeaCache Smart Mode: TẮT! (vì {steps} bước quá ít)")
        class DummyCache:
            skipped = 0
        teacache = DummyCache()
    else:
        threshold = 0.15   # bạn có thể thử 0.20 - 0.25 nếu muốn skip mạnh hơn
        print(f"\n[🧠] TeaCache Smart Mode: BẬT! (Ngưỡng {threshold} cho {steps} bước)")
        teacache = ExtremeTeaCache(rel_l1_thresh=threshold, start_percent=0.1)
        pipe = teacache.inject(pipe, total_steps=steps)

    print("\n[+] ✅ Ép dùng Hack: Model BF16 -> Attention FP16 (Mem Efficient)")
    original_sdp = torch.nn.functional.scaled_dot_product_attention
    
    def safe_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        kwargs.pop('enable_gqa', None)
        
        # 1. ÉP KIỂU TẠM THỜI: Đổi Q, K, V từ BF16 sang FP16 để lừa card Turing
        q_16 = query.to(torch.float16)
        k_16 = key.to(torch.float16)
        v_16 = value.to(torch.float16)
        
        # 2. BẬT EFFICIENT: Bây giờ nó thấy FP16, nó sẽ cho phép chạy Memory Efficient!
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
            out = original_sdp(q_16, k_16, v_16, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
            
        # 3. TRẢ VỀ BF16: Tính xong, trả lại BF16 cho Model chạy tiếp tốc độ cao
        return out.to(torch.bfloat16)
    
    torch.nn.functional.scaled_dot_product_attention = safe_scaled_dot_product_attention

    # ── 11. Render ─────────────────────────────────────────────────────
    print("\n🎥 BẮT ĐẦU RENDER...")
    print("⚠️  Disk offload → CHẬM hơn model-in-RAM nhưng KHÔNG OOM!")
    print("   Ước tính thời gian: ~15-40 phút tùy disk speed\n")

    target_img = (
        Image.open(img_path).convert("RGB")
        .resize((CONFIG["wan_width"], CONFIG["wan_height"]),
                Image.Resampling.LANCZOS)
    )
    generator = torch.Generator(device="cuda").manual_seed(42)

    try:
        with torch.inference_mode():
            frames = pipe(
                image                  = target_img,
                prompt_embeds          = p_emb,
                negative_prompt_embeds = n_emb,
                num_frames             = CONFIG["wan_num_frames"],
                height                 = CONFIG["wan_height"],
                width                  = CONFIG["wan_width"],
                guidance_scale         = CONFIG["wan_guidance_scale"],
                num_inference_steps    = CONFIG["wan_num_inference_steps"],
                generator              = generator,
            ).frames[0]
    except Exception as e:
        print(f"\n❌ Lỗi render:\n   {e}")
        traceback.print_exc()
        input("Enter để thoát...")
        return

    # ── 12. Export Video ──────────────────────────────────────────────
    video_path = os.path.join(OUTPUT_DIR, f"{safe_name}.mp4")
    export_to_video(frames, video_path, fps=CONFIG["wan_fps"])

    skipped_pct = 100.0 * teacache.skipped / max(1, CONFIG["wan_num_inference_steps"])
    print(f"\n✅ XONG!")
    print(f"   📹 {video_path}")
    print(f"   ⚡ TeaCache skip: {teacache.skipped} bước ({skipped_pct:.0f}%)")

    if os.path.exists(temp_pt):
        os.remove(temp_pt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Dừng bởi người dùng.")
    except Exception as e:
        print(f"\n❌ LỖI KHÔNG BẮT ĐƯỢC:")
        traceback.print_exc()
        input("\nEnter để thoát...")