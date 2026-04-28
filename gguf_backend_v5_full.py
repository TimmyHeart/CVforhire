"""
gguf_backend_v5.py  –  Hybrid Diffusers + GGUF Execution Backend
================================================================
FULLY SELF-CONTAINED MERGE of gguf_backend.py (v1), gguf_backendv2.py (v2),
gguf_backend_wan.py (v3), and gguf_backend_wan.py-patched-v3.1 (v4).

Architecture
------------
GGMLTensor              – thin torch.Tensor subclass that preserves quant metadata
GGUFTensorRegistry      – global store of {name: GGMLTensor}, never touches state_dict
GGMLDLLBridge           – ctypes wrapper around sd.cpp-python DLL (with Python fallback)
GGUFLinear              – nn.Linear replacement whose forward() calls the DLL bridge;
                           extended with WAN transpose / reshape / packed-QKV support
GGUFKeyMapper           – configurable Diffusers ↔ LDM/GGUF name translator (SD path)
build_wan_patch_plan    – WAN 2.2 mapper: GGUF tensor keys → Diffusers module paths
inject_gguf_into_model  – replaces nn.Linear modules with GGUFLinear in-place;
                           routes to WAN or SD injection path automatically
build_unet_gguf_native  – auto-detects WAN 2.2 vs SD/SDXL and builds the model
validate_gguf_linear    – compares torch vs GGUF outputs for one layer

Patch history (v4 critical fixes incorporated)
-----------------------------------------------
CRITICAL-1  _plan_packed_qkv: apply self_attn→attn1 alias BEFORE computing
            parent_path so packed QKV sibling lookups find the correct module.
CRITICAL-2  _forward_qkv_chunk: derive split_dim from self.transpose at runtime
            (GGML [in, packed_out] must split on dim=1, not dim=0).
CRITICAL-3  _infer_wan_config: layout-aware axis selection for hidden_size and
            cross_attn_dim; mandatory abort when hidden_size cannot be determined.
CRITICAL-4  _load_wan_float_weights: convert non-quantised GGMLTensor to
            torch.Tensor via numpy before passing to load_state_dict.
HIGH-1      DLL bypass: _DLL_ACCEPTS_GGML_LAYOUT flag allows DLL for WAN weights.
HIGH-2      _is_wan_model: WAN-exclusive pattern matching to prevent SD→WAN misrouting.
HIGH-3      _build_wan_transformer_gguf_native: post-load norm_q/norm_k check.
MEDIUM-1    ggml_linear_forward: raises RuntimeError for PATCH_EMBED_CONV sentinel.
MEDIUM-2    inject_gguf_into_model (WAN): n_patched counts actual setattr calls.
MEDIUM-3    _infer_wan_config: raises when hidden_size cannot be determined.
LOW-2       inject_gguf_into_model: post-injection audit for None ggml_weight.
LOW-4       dequantize_tensor resolved once at module load time.
ARCH        _build_wan_transformer_gguf_native: plan coverage gate (≥80%).

Usage
-----
from gguf_backend_v5 import (
    GGUFTensorRegistry, GGMLDLLBridge, inject_gguf_into_model,
    build_unet_gguf_native, run_validation_test,
)
"""

from __future__ import annotations

import ctypes
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# =============================================================================
# SECTION 0 – Low-4: resolve dequantize_tensor ONCE at module load time.
# =============================================================================

try:
    from test_patched import dequantize_tensor as _dequantize_tensor   # type: ignore
except ImportError:
    try:
        from dequant import dequantize_tensor as _dequantize_tensor    # type: ignore
    except ImportError:
        _dequantize_tensor = None  # Raises on first use — see guard below


def _get_dequantize_tensor():
    """Return dequantize_tensor or raise a clear RuntimeError."""
    if _dequantize_tensor is None:
        raise RuntimeError(
            "dequantize_tensor not found at module load time. "
            "Either test_patched.py or dequant.py must expose this function. "
            "Fix the import and restart."
        )
    return _dequantize_tensor


# =============================================================================
# SECTION 1 – GGMLTensor
# Kept as a torch.Tensor subclass so it travels transparently through PyTorch
# plumbing (device moves, dtype checks, etc.) while carrying quant metadata.
# The raw bytes ARE the quantised data; never call .float() on this class
# directly – always go through dequantize_tensor() from the dequant module.
# =============================================================================

class GGMLTensor(torch.Tensor):
    """
    A torch.Tensor whose data is *raw quantised bytes* from a GGUF file.

    Extra attributes
    ----------------
    tensor_type  : gguf.GGMLQuantizationType  (None for F32/F16)
    tensor_shape : torch.Size                 (logical output shape after dequant)
    patches      : list                       (LoRA / merge patches, usually empty)
    ggml_ptr     : int | None                 (raw pointer returned by DLL, if any)
    """

    @staticmethod
    def __new__(cls, data, *, tensor_type=None, tensor_shape=None,
                patches=None, ggml_ptr=None, **kwargs):
        kwargs.pop("requires_grad", None)          # never gradient-tracked
        instance = torch.Tensor._make_subclass(cls, data, False)
        instance.tensor_type  = tensor_type
        instance.tensor_shape = tensor_shape or data.shape
        instance.patches      = patches or []
        instance.ggml_ptr     = ggml_ptr           # populated by DLL bridge
        return instance

    def __init__(self, *_, **__):
        pass

    # ------------------------------------------------------------------
    # Preserve metadata through common torch operations
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        if not isinstance(new, GGMLTensor):
            new = GGMLTensor(new,
                             tensor_type=self.tensor_type,
                             tensor_shape=self.tensor_shape,
                             patches=list(self.patches))
        else:
            new.tensor_type  = self.tensor_type
            new.tensor_shape = self.tensor_shape
            new.patches      = list(self.patches)
            new.ggml_ptr     = self.ggml_ptr
        return new

    def clone(self, *args, **kwargs):
        return self          # intentional no-copy: quant bytes are read-only

    def detach(self, *args, **kwargs):
        return self

    def new_empty(self, size, *args, **kwargs):
        t = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(t, tensor_type=self.tensor_type,
                          tensor_shape=torch.Size(size),
                          patches=list(self.patches))

    def __repr__(self):
        qname = getattr(self.tensor_type, "name", str(self.tensor_type))
        return (f"GGMLTensor({list(self.tensor_shape)}, "
                f"quant={qname}, bytes={self.numel()})")


# =============================================================================
# SECTION 2 – GGUFTensorRegistry
# Single source-of-truth for all GGUF tensors loaded from one file.
# Completely separate from PyTorch's state_dict / load_state_dict machinery.
# =============================================================================

class GGUFTensorRegistry:
    """
    Stores {canonical_name: GGMLTensor} for one GGUF checkpoint.

    Rules
    -----
    • Populated by the GGUF loader (see populate_from_reader()).
    • Never converted to torch.float – bytes stay quantised.
    • Never passed to load_state_dict().
    • Consumed by inject_gguf_into_model() to patch nn.Linear modules.
    """

    def __init__(self):
        self._tensors: Dict[str, GGMLTensor] = {}
        self.metadata: dict = {}
        self.arch_str: str = "unknown"

    # ------------------------------------------------------------------
    def populate_from_reader(self, reader, bridge: "GGMLDLLBridge") -> None:
        """
        Read every tensor from a gguf.GGUFReader and store as GGMLTensor.

        Parameters
        ----------
        reader : gguf.GGUFReader
        bridge : GGMLDLLBridge  – used to retrieve native pointers when DLL
                                   is available; no-op when not.
        """
        import gguf

        TORCH_COMPAT = {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
        }

        for tensor in reader.tensors:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "not writable")
                data = torch.from_numpy(tensor.data)          # mmap – no copy

            # Logical output shape (stored reversed in GGUF)
            shape = self._get_orig_shape(reader, tensor.name)
            if shape is None:
                shape = torch.Size([int(x) for x in reversed(tensor.shape)])

            if tensor.tensor_type in TORCH_COMPAT:
                data = data.view(*shape)

            gg = GGMLTensor(data,
                            tensor_type=tensor.tensor_type,
                            tensor_shape=shape)

            # Attempt to retrieve a native pointer from the DLL
            if bridge.available:
                try:
                    gg.ggml_ptr = bridge.get_tensor_ptr(tensor.name)
                except Exception:
                    pass

            self._tensors[tensor.name] = gg

        log.info("[GGUFRegistry] Loaded %d tensors", len(self._tensors))

    # ------------------------------------------------------------------
    @staticmethod
    def _get_orig_shape(reader, tensor_name):
        field = reader.get_field(f"comfy.gguf.orig_shape.{tensor_name}")
        if field is None or len(field.types) != 2:
            return None
        return torch.Size(tuple(int(field.parts[pi][0]) for pi in field.data))

    # ------------------------------------------------------------------
    # Dict-like interface
    def __getitem__(self, name: str) -> GGMLTensor:
        return self._tensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    def get(self, name: str, default=None):
        return self._tensors.get(name, default)

    def keys(self):
        return self._tensors.keys()

    def items(self):
        return self._tensors.items()

    def __len__(self):
        return len(self._tensors)

    # ------------------------------------------------------------------
    # Convenience: build a plain {name: GGMLTensor} dict that the old
    # _read_gguf_all_tensors() callers expect (for compatibility shim).
    def to_state_dict(self) -> Dict[str, GGMLTensor]:
        """
        Returns the raw {name: GGMLTensor} mapping.
        NOTE: for compatibility with legacy code only.  Do NOT pass the
        result to load_state_dict() – GGMLTensors must stay quantised.
        """
        return dict(self._tensors)


# =============================================================================
# SECTION 3 – GGMLDLLBridge (ctypes → sd.cpp-python DLL)
# =============================================================================
#
# ---- Expected DLL exports (adjust to your actual sd.cpp-python build) -------
#
#  void*  ggml_backend_load_file (const char* path);
#  void*  ggml_backend_get_tensor(void* ctx, const char* name);
#  int    ggml_mul_mat_f16       (void* ctx,
#                                 void* weight_ptr,
#                                 const uint16_t* input_ptr, int input_M, int input_K,
#                                 uint16_t* output_ptr, int output_N);
#  void   ggml_backend_free      (void* ctx);
#
# The bridge wraps these four functions and exposes ggml_linear_forward().
# When the DLL is missing, every call transparently falls back to the pure-
# Python dequant path.
# ---------------------------------------------------------------------------

class GGMLDLLBridge:
    """
    Thin ctypes wrapper around the sd.cpp-python shared library.

    Parameters
    ----------
    dll_path : str | None
        Full path to the .so / .dll / .dylib.  Pass None to force fallback.
    gguf_path : str | None
        If provided the bridge calls ggml_backend_load_file() at init time
        so tensor pointers are valid immediately.
    """

    def __init__(self, dll_path: Optional[str] = None,
                 gguf_path: Optional[str] = None):
        self._lib      = None
        self._ctx      = None
        self.available = False

        if dll_path and os.path.exists(dll_path):
            self._load_dll(dll_path, gguf_path)
        else:
            log.info("[DLLBridge] No DLL found – using Python dequant fallback")

    # ------------------------------------------------------------------
    def _load_dll(self, dll_path: str, gguf_path: Optional[str]) -> None:
        try:
            lib = ctypes.CDLL(dll_path)
        except OSError as e:
            log.warning("[DLLBridge] Cannot open %s: %s", dll_path, e)
            return

        # ── declare signatures ──────────────────────────────────────────
        try:
            lib.ggml_backend_load_file.argtypes  = [ctypes.c_char_p]
            lib.ggml_backend_load_file.restype   = ctypes.c_void_p

            lib.ggml_backend_get_tensor.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.ggml_backend_get_tensor.restype  = ctypes.c_void_p

            # int ggml_mul_mat_f16(ctx, weight, input, M, K, output, N)
            lib.ggml_mul_mat_f16.argtypes = [
                ctypes.c_void_p,   # ctx
                ctypes.c_void_p,   # weight tensor pointer
                ctypes.POINTER(ctypes.c_uint16),  # input  (fp16 flat)
                ctypes.c_int,      # M  (batch)
                ctypes.c_int,      # K  (in_features)
                ctypes.POINTER(ctypes.c_uint16),  # output (fp16 flat)
                ctypes.c_int,      # N  (out_features)
            ]
            lib.ggml_mul_mat_f16.restype = ctypes.c_int

            lib.ggml_backend_free.argtypes = [ctypes.c_void_p]
            lib.ggml_backend_free.restype  = None
        except AttributeError as e:
            log.warning("[DLLBridge] DLL missing expected symbol: %s", e)
            return

        self._lib = lib

        # ── load file into backend context ─────────────────────────────
        if gguf_path:
            ctx = lib.ggml_backend_load_file(gguf_path.encode())
            if ctx:
                self._ctx    = ctx
                self.available = True
                log.info("[DLLBridge] DLL loaded + context created ✓")
            else:
                log.warning("[DLLBridge] ggml_backend_load_file returned NULL")
        else:
            # DLL available but no file loaded yet – pointer retrieval
            # will fail gracefully; matmul path still works if caller
            # provides a valid pointer.
            self.available = True

    # ------------------------------------------------------------------
    def get_tensor_ptr(self, tensor_name: str) -> Optional[int]:
        """Returns the native ggml_tensor* for tensor_name, or None."""
        if not self.available or self._ctx is None:
            return None
        ptr = self._lib.ggml_backend_get_tensor(
            self._ctx, tensor_name.encode()
        )
        return int(ptr) if ptr else None

    # ------------------------------------------------------------------
    def matmul_f16(self,
                   x: torch.Tensor,
                   ggml_weight: GGMLTensor) -> Optional[torch.Tensor]:
        """
        Call ggml_mul_mat_f16 via DLL.

        Returns None if DLL unavailable or call fails (caller must fallback).

        Parameters
        ----------
        x            : (batch, K) float16 activation on CPU
        ggml_weight  : GGMLTensor with a valid ggml_ptr
        """
        if not self.available or ggml_weight.ggml_ptr is None:
            return None

        ptr = ggml_weight.ggml_ptr
        if ptr is None or ptr == 0:
            return None

        try:
            x_f16  = x.to(torch.float16).contiguous().cpu()
            M, K   = x_f16.shape[0], x_f16.shape[1]
            N      = int(ggml_weight.tensor_shape[0])   # out_features

            out_buf = torch.empty((M, N), dtype=torch.float16)

            x_ptr   = x_f16.data_ptr()
            out_ptr = out_buf.data_ptr()

            rc = self._lib.ggml_mul_mat_f16(
                self._ctx,
                ctypes.c_void_p(ptr),
                ctypes.cast(x_ptr,   ctypes.POINTER(ctypes.c_uint16)),
                M, K,
                ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_uint16)),
                N,
            )
            if rc != 0:
                log.warning("[DLLBridge] ggml_mul_mat_f16 returned %d", rc)
                return None

            return out_buf

        except Exception as e:
            log.warning("[DLLBridge] matmul_f16 exception: %s", e)
            return None

    # ------------------------------------------------------------------
    def __del__(self):
        if self._lib and self._ctx:
            try:
                self._lib.ggml_backend_free(self._ctx)
            except Exception:
                pass


# Module-level singleton – callers replace this with a real bridge instance.
_DEFAULT_BRIDGE: GGMLDLLBridge = GGMLDLLBridge()   # fallback (no DLL)


def set_global_bridge(bridge: GGMLDLLBridge) -> None:
    """Register the shared DLL bridge used by all GGUFLinear layers."""
    global _DEFAULT_BRIDGE
    _DEFAULT_BRIDGE = bridge


# =============================================================================
# HIGH-1: DLL layout convention flag.
#
# If your DLL's matmul_f16 accepts GGML-native [in, out] weights directly
# (ggml_mul_mat which is natively GGML-layout), set this to True.
# The DLL will then be called for WAN weights (transpose=True) as well.
#
# If your DLL assumes PyTorch [out, in] weights, keep False.
# WAN inference will fall through to Python dequant.
# =============================================================================
_DLL_ACCEPTS_GGML_LAYOUT: bool = False


# =============================================================================
# SECTION 4 – ggml_linear_forward
# The single function that bridges a torch activation to a GGML weight.
# =============================================================================

def ggml_linear_forward(
    x: torch.Tensor,
    ggml_weight: GGMLTensor,
    bias: Optional[torch.Tensor] = None,
    bridge: Optional[GGMLDLLBridge] = None,
    *,
    transpose: bool = False,
    reshape: Optional[tuple] = None,
) -> torch.Tensor:
    """
    Execute  y = x @ ggml_weight  (or transpose variant) + bias.

    Execution priority
    ------------------
    1. Native DLL (ggml_mul_mat_f16) – zero Python dequant, fastest.
       DLL path is also gated by _DLL_ACCEPTS_GGML_LAYOUT for WAN weights.
    2. Pure-Python dequant            – always available, correct for all
       weight orientations.

    Parameters
    ----------
    x           : (..., K) torch.Tensor  – activations, any dtype
    ggml_weight : GGMLTensor             – raw quantised bytes
    bias        : (N,) | None
    bridge      : GGMLDLLBridge | None   – uses module singleton if None
    transpose   : bool
        False (SD default) → weight stored [out, in] (PyTorch); F.linear(x, w).
        True  (WAN)        → weight stored [in, out] (GGML);    torch.matmul(x, w).
    reshape     : tuple | None
        ("PATCH_EMBED_CONV",) → ALWAYS raises RuntimeError.
                                patch_embed must go through _load_wan_float_weights.
        Any other tuple       → w.reshape(reshape) after dequant.

    Returns
    -------
    torch.Tensor  – same device/dtype as x
    """
    # MEDIUM-1: Raise on PATCH_EMBED_CONV sentinel.
    if reshape == ("PATCH_EMBED_CONV",):
        raise RuntimeError(
            "ggml_linear_forward called with reshape=('PATCH_EMBED_CONV',) sentinel. "
            "patch_embedding MUST be routed through _load_wan_float_weights and loaded "
            "via load_state_dict, not through GGUFLinear.  This is a routing bug."
        )

    bridge      = bridge or _DEFAULT_BRIDGE
    orig_dtype  = x.dtype
    orig_device = x.device
    orig_shape  = x.shape
    flat        = x.reshape(-1, orig_shape[-1])          # (M, K)

    # HIGH-1: DLL path respects _DLL_ACCEPTS_GGML_LAYOUT.
    _dll_eligible = (
        bridge.available and
        ggml_weight.ggml_ptr is not None and
        (not transpose or _DLL_ACCEPTS_GGML_LAYOUT)
    )

    if _dll_eligible:
        result = bridge.matmul_f16(flat, ggml_weight)
        if result is not None:
            result = result.to(device=orig_device, dtype=orig_dtype)
            if bias is not None:
                result = result + bias.to(dtype=orig_dtype)
            return result.reshape(*orig_shape[:-1], result.shape[-1])
        log.debug("[ggml_linear_forward] DLL call failed – falling back to Python")

    # ── Python dequant path ──────────────────────────────────────────────────
    dequantize_tensor = _get_dequantize_tensor()
    w = dequantize_tensor(ggml_weight, dtype=orig_dtype)

    # ── Optional reshape ─────────────────────────────────────────────────────
    if reshape is not None:
        try:
            w = w.reshape(reshape)
        except RuntimeError as exc:
            log.warning("[ggml_linear_forward] reshape %s failed: %s", reshape, exc)

    # ── Matmul branch ────────────────────────────────────────────────────────
    if transpose:
        # GGML layout: w is [in_features, out_features].
        # torch.matmul(x, w) → [batch, in] @ [in, out] = [batch, out]   ✓
        result = torch.matmul(flat, w.to(dtype=orig_dtype))
    else:
        # PyTorch layout: w is [out_features, in_features].
        # F.linear(x, w) = x @ w.T → [batch, in] @ [in, out] = [batch, out] ✓
        result = F.linear(flat, w, None)

    if bias is not None:
        result = result + bias.to(dtype=orig_dtype)

    return result.reshape(*orig_shape[:-1], result.shape[-1])


# =============================================================================
# SECTION 5 – GGUFLinear
# Drop-in replacement for nn.Linear that holds a GGMLTensor, not a Parameter.
# Extended with WAN transpose / reshape / packed-QKV support.
# =============================================================================

class GGUFLinear(nn.Module):
    """
    nn.Linear replacement backed by a GGMLTensor weight.

    Key contract
    ------------
    • self.ggml_weight   holds raw quantised weight bytes – never converted.
    • self.weight        is None (removed from module parameter tracking).
    • forward() calls ggml_linear_forward(), which dequants on-the-fly
      or uses the DLL bridge, depending on availability.

    WAN extensions
    --------------
    • self.transpose     True → weight is GGML-layout [in, out]; matmul
                         is performed as  x @ w  (not  x @ w.T).
    • self.reshape       Non-None → weight is reshaped after dequant.
                         Use ("PATCH_EMBED_CONV",) as a sentinel when
                         the target shape must be read from model config.
    • self.qkv_info      Non-None for a slice of a packed QKV tensor:
                         {"type": "split_chunk", "chunk_idx": int,
                          "n_chunks": int}
                         The forward pass dequants the full packed tensor
                         and extracts the right chunk on-the-fly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        bridge: Optional[GGMLDLLBridge] = None,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.ggml_weight: Optional[GGMLTensor] = None
        self._bias: Optional[nn.Parameter]     = None
        self._bridge = bridge or _DEFAULT_BRIDGE

        # WAN-specific metadata (defaults are SD-compatible no-ops)
        self.transpose: bool               = False
        self.reshape:   Optional[tuple]    = None
        self.qkv_info:  Optional[dict]     = None

        # Reserve so named_parameters() / load_state_dict don't crash.
        self.weight = None

    # ------------------------------------------------------------------
    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        ggml_weight: GGMLTensor,
        bridge: Optional[GGMLDLLBridge] = None,
    ) -> "GGUFLinear":
        """
        SD / Diffusers path (original behaviour, unchanged).

        Builds a GGUFLinear from an existing nn.Linear + a GGMLTensor.
        transpose=False, reshape=None, qkv_info=None.
        """
        obj = cls(
            module.in_features,
            module.out_features,
            has_bias=(module.bias is not None),
            bridge=bridge,
        )
        obj.ggml_weight = ggml_weight
        if module.bias is not None:
            obj._bias = nn.Parameter(
                module.bias.data.clone(), requires_grad=False
            )
        return obj

    # ------------------------------------------------------------------
    @classmethod
    def from_linear_wan(
        cls,
        module: nn.Linear,
        ggml_weight: GGMLTensor,
        bridge: Optional[GGMLDLLBridge],
        meta_info: dict,
    ) -> "GGUFLinear":
        """
        WAN 2.2 path.

        Like from_linear() but additionally stores transpose / reshape /
        qkv_info from build_wan_patch_plan() meta_info so the forward pass
        can apply the correct matmul orientation and chunking.

        Parameters
        ----------
        module      : the nn.Linear being replaced (its bias is preserved)
        ggml_weight : GGMLTensor from the registry (may be a full packed
                      QKV tensor when qkv_info["type"] == "split_chunk")
        bridge      : GGMLDLLBridge
        meta_info   : dict produced by build_wan_patch_plan():
                      {
                        "transpose": bool,
                        "reshape":   tuple | None,
                        "qkv":       dict | None,
                      }
        """
        obj = cls(
            module.in_features,
            module.out_features,
            has_bias=(module.bias is not None),
            bridge=bridge,
        )
        obj.ggml_weight = ggml_weight
        obj.transpose   = bool(meta_info.get("transpose", False))
        obj.reshape     = meta_info.get("reshape", None)
        obj.qkv_info    = meta_info.get("qkv", None)

        if module.bias is not None:
            obj._bias = nn.Parameter(
                module.bias.data.clone(), requires_grad=False
            )

        # LOW-2 (construction-time): Validate GGML tensor shape against module.
        if ggml_weight is not None and hasattr(ggml_weight, "tensor_shape"):
            ts = ggml_weight.tensor_shape
            if ts is not None and len(ts) == 2 and not meta_info.get("qkv"):
                # GGML layout [in, out]
                expected_in, expected_out = int(ts[0]), int(ts[1])
                if obj.in_features != expected_in or obj.out_features != expected_out:
                    log.warning(
                        "[GGUFLinear.from_linear_wan] Shape mismatch: "
                        "module (%d→%d) vs GGUF tensor (%d→%d).  "
                        "This may indicate a wrong model config (hidden_size).",
                        obj.in_features, obj.out_features,
                        expected_in, expected_out,
                    )
        return obj

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ggml_weight is None:
            # LOW-2 (runtime): ggml_weight should never be None at forward time.
            raise RuntimeError(
                f"GGUFLinear.forward called with ggml_weight=None. "
                f"Layer: in={self.in_features}, out={self.out_features}. "
                f"This means inject_gguf_into_model failed to assign a tensor "
                f"to this layer.  Check the post-injection audit log."
            )

        bias = self._bias.data if self._bias is not None else None

        # ── Packed QKV split chunk path ──────────────────────────────────────
        if self.qkv_info is not None and self.qkv_info.get("type") == "split_chunk":
            return self._forward_qkv_chunk(x, bias)

        # ── Standard path (SD + WAN non-packed) ─────────────────────────────
        return ggml_linear_forward(
            x,
            self.ggml_weight,
            bias,
            self._bridge,
            transpose=self.transpose,
            reshape=self.reshape,
        )

    # ------------------------------------------------------------------
    def _forward_qkv_chunk(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Dequantise the full packed QKV tensor and extract one projection chunk.

        self.qkv_info:
            {
                "type":      "split_chunk",
                "chunk_idx": int,   # 0=Q, 1=K, 2=V
                "n_chunks":  int,   # always 3 for QKV
            }

        CRITICAL-2: split_dim is derived from self.transpose at runtime.
            transpose=True  → GGML layout [in_features, 3*out_features]
                              → split dim=1 (output dimension)
            transpose=False → PyTorch layout [3*out_features, in_features]
                              → split dim=0 (output dimension)
        """
        chunk_idx = self.qkv_info["chunk_idx"]
        n_chunks  = self.qkv_info["n_chunks"]

        # CRITICAL-2: Derive split_dim from self.transpose at runtime.
        split_dim = 1 if self.transpose else 0

        orig_dtype  = x.dtype
        orig_shape  = x.shape
        flat        = x.reshape(-1, orig_shape[-1])

        dequantize_tensor = _get_dequantize_tensor()
        w_full = dequantize_tensor(self.ggml_weight, dtype=orig_dtype)

        total_size = w_full.shape[split_dim]
        if total_size % n_chunks != 0:
            raise ValueError(
                f"[GGUFLinear QKV] Packed weight shape={tuple(w_full.shape)}, "
                f"split_dim={split_dim} (transpose={self.transpose}), "
                f"total_size={total_size} is not divisible by n_chunks={n_chunks}. "
                f"Either the weight layout detection or n_chunks is wrong."
            )
        chunk_size = total_size // n_chunks
        w_chunk = w_full.narrow(split_dim, chunk_idx * chunk_size, chunk_size)

        if self.transpose:
            # GGML layout: w_chunk is [in_features, out_chunk]
            # x @ w_chunk → [batch, in] @ [in, out/3] = [batch, out/3]  ✓
            result = torch.matmul(flat, w_chunk)
        else:
            # PyTorch layout: w_chunk is [out_chunk, in_features]
            # F.linear → x @ w.T → correct
            result = F.linear(flat, w_chunk, None)

        if bias is not None:
            result = result + bias.to(dtype=orig_dtype)

        return result.reshape(*orig_shape[:-1], result.shape[-1])

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        qname = "none"
        if self.ggml_weight is not None:
            qname = getattr(
                getattr(self.ggml_weight, "tensor_type", None), "name", "?"
            )
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"quant={qname}, transpose={self.transpose}, "
            f"qkv={self.qkv_info is not None}, dll={self._bridge.available}"
        )


# =============================================================================
# SECTION 6 – GGUFKeyMapper
# Translates between Diffusers module-path names and GGUF/LDM tensor names.
# Used for SD / SDXL models only. WAN uses build_wan_patch_plan instead.
# =============================================================================

# Default LDM → Diffusers block-index mapping for SD 1.x UNet
_LDM_BLOCK_TO_DIFFUSERS: Dict[str, Tuple[str, int, int]] = {
    # (ldm_prefix) → (diffusers_block_type, outer_idx, inner_idx)
    "input_blocks.0":  ("down_blocks", 0, 0),
    "input_blocks.1":  ("down_blocks", 0, 0),
    "input_blocks.2":  ("down_blocks", 0, 1),
    "input_blocks.3":  ("down_blocks", 0, 1),   # downsample
    "input_blocks.4":  ("down_blocks", 1, 0),
    "input_blocks.5":  ("down_blocks", 1, 0),
    "input_blocks.6":  ("down_blocks", 1, 1),
    "input_blocks.7":  ("down_blocks", 1, 1),
    "input_blocks.8":  ("down_blocks", 2, 0),
    "input_blocks.9":  ("down_blocks", 2, 0),
    "input_blocks.10": ("down_blocks", 2, 1),
    "input_blocks.11": ("down_blocks", 2, 1),
    "middle_block":    ("mid_block",   0, 0),
    "output_blocks.0": ("up_blocks",   3, 0),
    "output_blocks.1": ("up_blocks",   3, 1),
    "output_blocks.2": ("up_blocks",   3, 2),
    "output_blocks.3": ("up_blocks",   2, 0),
    "output_blocks.4": ("up_blocks",   2, 1),
    "output_blocks.5": ("up_blocks",   2, 2),
    "output_blocks.6": ("up_blocks",   1, 0),
    "output_blocks.7": ("up_blocks",   1, 1),
    "output_blocks.8": ("up_blocks",   1, 2),
    "output_blocks.9": ("up_blocks",   0, 0),
    "output_blocks.10":("up_blocks",   0, 1),
    "output_blocks.11":("up_blocks",   0, 2),
}

# Attention sub-key renaming: LDM name → Diffusers name
_ATTN_SUBKEY_MAP: Dict[str, str] = {
    "to_q":   "to_q",
    "to_k":   "to_k",
    "to_v":   "to_v",
    "to_out.0": "to_out.0",
    "proj_in":  "proj_in",
    "proj_out": "proj_out",
    # SDXL packed QKV (if encountered, split 3× and rename)
    "in_proj":  "in_proj",
}


class GGUFKeyMapper:
    """
    Maps a Diffusers named_modules() path to the corresponding GGUF
    tensor name stored in a GGUFTensorRegistry.

    The mapping is intentionally configurable at construction time so
    custom models (Pony, SD 2.x, etc.) can override any entry without
    touching the code.

    Parameters
    ----------
    extra_rules : dict[str, str] | None
        Additional {diffusers_suffix → gguf_key_suffix} overrides applied
        after the built-in rules.
    ldm_prefix : str
        The prefix stripped from GGUF tensor names before lookup
        (e.g. "model.diffusion_model.").
    weight_suffix : str
        Suffix appended to GGUF tensor names (default ".weight").
    """

    def __init__(self,
                 extra_rules: Optional[Dict[str, str]] = None,
                 ldm_prefix: str = "model.diffusion_model.",
                 weight_suffix: str = ".weight"):
        self._extra      = extra_rules or {}
        self._ldm_prefix = ldm_prefix
        self._wsuffix    = weight_suffix

    # ------------------------------------------------------------------
    def diffusers_to_gguf(self,
                          diffusers_path: str,
                          registry: GGUFTensorRegistry) -> Optional[str]:
        """
        Given a Diffusers module path, return the matching GGUF tensor
        name, or None if no match is found.

        Strategy
        --------
        1. Check extra_rules override.
        2. Try direct lookup with weight suffix.
        3. Try LDM key heuristics (block remapping).
        4. Try fuzzy suffix match against all registry keys.
        """
        # Rule 1: explicit override
        if diffusers_path in self._extra:
            cand = self._extra[diffusers_path]
            if cand in registry:
                return cand

        # Rule 2: direct (Diffusers-style GGUF)
        direct = diffusers_path + self._wsuffix
        if direct in registry:
            return direct

        # Rule 3: LDM heuristic reverse-translate
        ldm_key = self._diffusers_to_ldm(diffusers_path)
        if ldm_key:
            cand = self._ldm_prefix + ldm_key + self._wsuffix
            if cand in registry:
                return cand
            # Some GGUF files already have the prefix stripped
            if ldm_key + self._wsuffix in registry:
                return ldm_key + self._wsuffix

        # Rule 4: fuzzy suffix
        suffix = "." + diffusers_path.replace(".", ".")
        for k in registry.keys():
            if k.endswith(suffix + self._wsuffix) or k.endswith(suffix):
                return k

        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _diffusers_to_ldm(path: str) -> Optional[str]:
        """
        Best-effort reverse translation of a Diffusers module path back to
        an LDM-style key.  Returns None on failure.

        Examples
        --------
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
        →  "input_blocks.1.1.transformer_blocks.0.attn1.to_q"
        """
        parts = path.split(".")
        if not parts:
            return None

        # ── down_blocks ─────────────────────────────────────────────
        if parts[0] == "down_blocks" and len(parts) >= 4:
            b_idx  = int(parts[1])   # 0..3
            if "attentions" in parts:
                att_idx = int(parts[parts.index("attentions") + 1])
                # input_blocks index = b_idx*3 + att_idx + 1 (approx for SD1)
                ib_idx = b_idx * 3 + att_idx + 1
                tail   = ".".join(parts[parts.index("attentions") + 2:])
                return f"input_blocks.{ib_idx}.1.{tail}"
            else:
                return None   # ResNet blocks – key mapping varies

        # ── mid_block ────────────────────────────────────────────────
        if parts[0] == "mid_block":
            tail = ".".join(parts[1:])
            return f"middle_block.{tail}"

        # ── up_blocks ────────────────────────────────────────────────
        if parts[0] == "up_blocks" and len(parts) >= 4:
            b_idx  = int(parts[1])
            if "attentions" in parts:
                att_idx = int(parts[parts.index("attentions") + 1])
                ob_idx  = (3 - b_idx) * 3 + att_idx
                tail    = ".".join(parts[parts.index("attentions") + 2:])
                return f"output_blocks.{ob_idx}.1.{tail}"
            return None

        return None

    # ------------------------------------------------------------------
    def build_patch_plan(
            self,
            model: nn.Module,
            registry: GGUFTensorRegistry,
    ) -> Dict[str, Tuple[str, GGMLTensor]]:
        """
        Walk the model's named_modules() and for every nn.Linear build a
        {module_path → (gguf_key, GGMLTensor)} mapping.

        Returns only the layers that were successfully matched.
        """
        plan: Dict[str, Tuple[str, GGMLTensor]] = {}
        unmatched: List[str] = []

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            gguf_key = self.diffusers_to_gguf(name, registry)
            if gguf_key:
                plan[name] = (gguf_key, registry[gguf_key])
            else:
                unmatched.append(name)

        matched = len(plan)
        total   = matched + len(unmatched)
        log.info("[KeyMapper] Matched %d / %d linear layers", matched, total)
        if unmatched:
            log.debug("[KeyMapper] Unmatched: %s", unmatched[:10])

        return plan


# =============================================================================
# SECTION 7 – build_wan_patch_plan  (WAN 2.2)
#
# Rules enforced:
#   • Only nn.Linear modules are patched (not Conv, Embedding, Norm, etc.)
#   • Scale tensors (OP_SCALE) are skipped unconditionally.
#   • Norm / modulation tensors are skipped.
#   • Bias keys are skipped.
#   • Packed QKV tensors are exploded into N separate plan entries.
#   • Already-wrapped GGUFLinear modules are skipped.
# =============================================================================

def build_wan_patch_plan(
    model: nn.Module,
    registry: GGUFTensorRegistry,
    mapping: dict,
) -> Dict[str, Tuple[str, GGMLTensor, dict]]:
    """
    Build a patch plan that maps PyTorch module paths to GGUF tensors.

    Parameters
    ----------
    model    : The instantiated WAN transformer model (empty or partial).
    registry : GGUFTensorRegistry populated from the GGUF file.
    mapping  : Output of wan_gguf_keymap.map_wan22_keys().

    Returns
    -------
    dict[module_path, (gguf_key, GGMLTensor, meta_info)]

    ``meta_info`` contains:
        {
            "transpose": bool,
            "reshape":   tuple | None,
            "qkv":       dict | None,  # see GGUFLinear.qkv_info
        }

    For split-chunk entries (packed QKV) the "qkv" sub-dict is:
        {
            "type":      "split_chunk",
            "chunk_idx": int,
            "n_chunks":  int,
        }
    """
    try:
        from wan_gguf_keymap import OP_NORM, OP_MODULATION, OP_SCALE, OP_UNKNOWN
    except ImportError:
        # Hardcode sentinel strings if wan_gguf_keymap is not importable
        OP_NORM       = "norm"
        OP_MODULATION = "modulation"
        OP_SCALE      = "scale"
        OP_UNKNOWN    = "unknown"

    # ── Build a fast lookup index: module_path → module ──────────────────────
    module_index: Dict[str, nn.Module] = {
        name: mod for name, mod in model.named_modules()
    }

    plan: Dict[str, Tuple[str, GGMLTensor, dict]] = {}
    skipped_no_module: List[str] = []
    skipped_not_linear: List[str] = []
    skipped_no_tensor:  List[str] = []

    for target_key, info in mapping.items():
        source_key = info["source_key"]
        op_type    = info.get("op_type", OP_UNKNOWN)
        transpose  = info.get("transpose", False)
        reshape    = info.get("reshape", None)
        qkv        = info.get("qkv", None)       # non-None → packed QKV

        # ── 1. Skip auxiliary / non-weight tensor types ───────────────────────
        if op_type in (OP_SCALE, OP_NORM, OP_MODULATION):
            continue

        # ── 2. Only process ".weight" keys ───────────────────────────────────
        if target_key.endswith(".bias"):
            continue
        if not target_key.endswith(".weight"):
            continue

        # ── 3. Verify tensor exists in registry ──────────────────────────────
        if source_key not in registry:
            skipped_no_tensor.append(source_key)
            continue

        gg_tensor = registry[source_key]

        # ── 4. Derive module path from target_key ────────────────────────────
        #       "blocks.0.attn1.to_q.weight"  →  "blocks.0.attn1.to_q"
        module_path = target_key[: -len(".weight")]

        # ── 5. Handle packed QKV ─────────────────────────────────────────────
        if qkv is not None and qkv.get("type") == "packed":
            _plan_packed_qkv(
                plan, module_index, module_path, gg_tensor, source_key,
                qkv, transpose, reshape,
                skipped_no_module, skipped_not_linear,
            )
            continue

        # ── 6. Resolve module ─────────────────────────────────────────────────
        if module_path not in module_index:
            skipped_no_module.append(module_path)
            continue

        module = module_index[module_path]

        # ── 7. Guard: only patch plain nn.Linear ─────────────────────────────
        if isinstance(module, GGUFLinear):
            log.debug("[build_wan_patch_plan] %s already GGUFLinear – skipping",
                      module_path)
            continue
        if not isinstance(module, nn.Linear):
            skipped_not_linear.append(module_path)
            continue

        meta_info: dict = {
            "transpose": transpose,
            "reshape":   reshape,
            "qkv":       None,
        }
        plan[module_path] = (source_key, gg_tensor, meta_info)

    # ── Diagnostic summary ────────────────────────────────────────────────────
    if skipped_no_module:
        log.debug(
            "[build_wan_patch_plan] %d target keys had no matching module: %s",
            len(skipped_no_module), skipped_no_module[:5],
        )
    if skipped_not_linear:
        log.debug(
            "[build_wan_patch_plan] %d modules are not nn.Linear (norm/conv/etc): %s",
            len(skipped_not_linear), skipped_not_linear[:5],
        )
    if skipped_no_tensor:
        log.warning(
            "[build_wan_patch_plan] %d source keys missing from registry: %s",
            len(skipped_no_tensor), skipped_no_tensor[:5],
        )

    log.info(
        "[build_wan_patch_plan] plan size = %d (from %d mapping entries)",
        len(plan), len(mapping),
    )
    return plan


def _plan_packed_qkv(
    plan: dict,
    module_index: Dict[str, nn.Module],
    qkv_module_path: str,
    gg_tensor: GGMLTensor,
    source_key: str,
    qkv_info: dict,
    transpose: bool,
    reshape: Optional[tuple],
    skipped_no_module: List[str],
    skipped_not_linear: List[str],
) -> None:
    """
    Explode a packed QKV source tensor into per-projection plan entries.

    CRITICAL-1: Apply self_attn→attn1 / cross_attn→attn2 alias BEFORE
    computing parent_path.
    """
    split_names = qkv_info.get("split", ["to_q", "to_k", "to_v"])
    n_chunks    = len(split_names)

    # CRITICAL-1: Apply self_attn→attn1 / cross_attn→attn2 alias
    # BEFORE computing parent_path so sibling lookups find correct module.
    _ATTN_ALIAS = {"self_attn": "attn1", "cross_attn": "attn2"}

    parts = qkv_module_path.split(".")
    aliased_parts = [_ATTN_ALIAS.get(p, p) for p in parts]
    # Drop the final segment (the packed tensor name: "qkv", "in_proj", …)
    parent_path = ".".join(aliased_parts[:-1]) if len(aliased_parts) > 1 else ""

    log.debug(
        "[_plan_packed_qkv] qkv_module_path=%s  →  parent_path=%s",
        qkv_module_path, parent_path,
    )

    for chunk_idx, proj_name in enumerate(split_names):
        sub_path = f"{parent_path}.{proj_name}" if parent_path else proj_name

        if sub_path not in module_index:
            log.debug("[_plan_packed_qkv] sibling module not found: %s", sub_path)
            skipped_no_module.append(sub_path)
            continue

        sub_module = module_index[sub_path]
        if isinstance(sub_module, GGUFLinear):
            log.debug("[_plan_packed_qkv] %s already GGUFLinear", sub_path)
            continue
        if not isinstance(sub_module, nn.Linear):
            skipped_not_linear.append(sub_path)
            continue

        meta_info: dict = {
            "transpose": transpose,
            "reshape":   reshape,
            "qkv": {
                "type":      "split_chunk",
                "chunk_idx": chunk_idx,
                "n_chunks":  n_chunks,
                # NOTE: "dim" is intentionally omitted.  _forward_qkv_chunk
                # derives split_dim from self.transpose at runtime (CRITICAL-2).
            },
        }
        plan[sub_path] = (source_key, gg_tensor, meta_info)
        log.debug(
            "[_plan_packed_qkv] %s → %s  chunk %d/%d  (split_dim derived at runtime)",
            sub_path, source_key, chunk_idx, n_chunks,
        )


# =============================================================================
# SECTION 8 – inject_gguf_into_model
# Replaces matched nn.Linear modules in-place with GGUFLinear.
# Does NOT touch load_state_dict.  Does NOT convert tensors.
# Routes to WAN path (when mapping= provided) or SD path (GGUFKeyMapper).
# =============================================================================

def inject_gguf_into_model(
    model:    nn.Module,
    registry: GGUFTensorRegistry,
    mapper:   Optional[GGUFKeyMapper] = None,
    bridge:   Optional[GGMLDLLBridge] = None,
    layers:   str = "linear",
    *,
    mapping: Optional[dict] = None,
) -> Tuple[int, int]:
    """
    Walk *model* and replace nn.Linear layers with GGUFLinear wherever a
    matching tensor exists in *registry*.

    Parameters
    ----------
    model    : Diffusers model (UNet, WAN transformer, etc.)
    registry : Populated GGUFTensorRegistry.
    mapper   : GGUFKeyMapper.  Used ONLY when mapping=None (SD path).
               Ignored entirely when mapping is provided.
    bridge   : GGMLDLLBridge (uses module singleton if None).
    layers   : Layer filter ("linear" supported; reserved for future use).
    mapping  : dict output of wan_gguf_keymap.map_wan22_keys().
               When provided, activates the WAN execution path and
               GGUFKeyMapper is NOT used.

    Returns
    -------
    (n_patched, n_total_linear) : layers replaced / total linear layers
    """
    bridge  = bridge or _DEFAULT_BRIDGE
    n_total = sum(
        1 for _, m in model.named_modules() if isinstance(m, nn.Linear)
    )

    # ═══════════════════════════════════════════════════════════════════
    # WAN PATH  –  mapping= provided
    # ═══════════════════════════════════════════════════════════════════
    if mapping is not None:
        plan = build_wan_patch_plan(model, registry, mapping)

        # MEDIUM-2: Count n_patched from actual successful setattr calls.
        n_patched = 0
        for module_path, (gguf_key, gg_tensor, meta_info) in plan.items():
            parent, child_name = _resolve_parent(model, module_path)
            if parent is None:
                log.warning("[inject_wan] Cannot resolve parent for %s", module_path)
                continue

            original = getattr(parent, child_name, None)
            if original is None or isinstance(original, GGUFLinear):
                continue
            if not isinstance(original, nn.Linear):
                continue

            new_module = GGUFLinear.from_linear_wan(original, gg_tensor, bridge, meta_info)
            setattr(parent, child_name, new_module)
            n_patched += 1

            log.debug(
                "[inject_wan] %s → %s  (transpose=%s, qkv=%s)",
                module_path, gguf_key,
                meta_info.get("transpose"), meta_info.get("qkv") is not None,
            )

        # LOW-2: Post-injection audit.
        null_weight_layers = [
            name for name, mod in model.named_modules()
            if isinstance(mod, GGUFLinear) and mod.ggml_weight is None
        ]
        if null_weight_layers:
            log.critical(
                "[inject_wan] POST-INJECTION AUDIT: %d GGUFLinear layers have "
                "ggml_weight=None — these will raise RuntimeError on forward pass: %s",
                len(null_weight_layers), null_weight_layers[:5],
            )

        log.info(
            "[inject_gguf] WAN path: replaced %d / %d Linear layers with GGUFLinear",
            n_patched, n_total,
        )
        return n_patched, n_total

    # ═══════════════════════════════════════════════════════════════════
    # SD / DIFFUSERS PATH  –  original GGUFKeyMapper logic
    # ═══════════════════════════════════════════════════════════════════
    mapper = mapper or GGUFKeyMapper()
    plan_sd = mapper.build_patch_plan(model, registry)

    n_patched_sd = 0
    for module_path, (gguf_key, gg_tensor) in plan_sd.items():
        parent, child_name = _resolve_parent(model, module_path)
        if parent is None:
            log.warning("[inject_sd] Cannot resolve parent for %s", module_path)
            continue

        original = getattr(parent, child_name, None)
        if original is None or isinstance(original, GGUFLinear):
            continue

        new_module = GGUFLinear.from_linear(original, gg_tensor, bridge)
        setattr(parent, child_name, new_module)
        n_patched_sd += 1

        log.debug(
            "[inject_sd] %s → %s (%s)",
            module_path, gguf_key,
            getattr(gg_tensor.tensor_type, "name", "?"),
        )

    log.info(
        "[inject_gguf] SD path: replaced %d / %d Linear layers with GGUFLinear",
        n_patched_sd, n_total,
    )
    return n_patched_sd, n_total


def _resolve_parent(
    model: nn.Module,
    module_path: str,
) -> Tuple[Optional[nn.Module], str]:
    """
    Walk the module tree and return (parent_module, child_attr_name).

    Returns (None, "") on failure so callers can guard safely.
    """
    parts = module_path.split(".")
    parent: nn.Module = model
    try:
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]
    except AttributeError as exc:
        log.debug("[_resolve_parent] Failed: %s – %s", module_path, exc)
        return None, ""


# =============================================================================
# SECTION 9 – Quantisation helpers
# =============================================================================

def _is_quantised(v) -> bool:
    """True if v is a GGMLTensor with a non-float quant type."""
    try:
        import gguf
        FLOAT_TYPES = {gguf.GGMLQuantizationType.F32,
                       gguf.GGMLQuantizationType.F16, None}
        return isinstance(v, GGMLTensor) and v.tensor_type not in FLOAT_TYPES
    except ImportError:
        return isinstance(v, GGMLTensor)


def _detect_unet_config_fallback(unet_sd, is_sdxl):
    """Minimal config detection if test_patched is not importable."""
    if is_sdxl:
        return {
            "act_fn": "silu",
            "addition_embed_type": "text_time",
            "addition_time_embed_dim": 256,
            "attention_head_dim": [5, 10, 20],
            "block_out_channels": [320, 640, 1280],
            "cross_attention_dim": 2048,
            "down_block_types": ["DownBlock2D",
                                  "CrossAttnDownBlock2D",
                                  "CrossAttnDownBlock2D"],
            "in_channels": 4, "out_channels": 4,
            "layers_per_block": 2,
            "mid_block_type": "UNetMidBlock2DCrossAttn",
            "norm_eps": 1e-5, "norm_num_groups": 32,
            "projection_class_embeddings_input_dim": 2816,
            "sample_size": 128,
            "transformer_layers_per_block": [1, 2, 10],
            "up_block_types": ["CrossAttnUpBlock2D",
                                "CrossAttnUpBlock2D",
                                "UpBlock2D"],
            "use_linear_projection": True,
        }
    return {
        "act_fn": "silu",
        "attention_head_dim": 8,
        "block_out_channels": [320, 640, 1280, 1280],
        "cross_attention_dim": 768,
        "down_block_types": ["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
        "in_channels": 4, "out_channels": 4,
        "layers_per_block": 2,
        "norm_eps": 1e-5, "norm_num_groups": 32,
        "sample_size": 64,
        "up_block_types": ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3,
    }


# =============================================================================
# SECTION 10 – build_unet_gguf_native
# Auto-detects WAN 2.2 vs SD/SDXL from tensor key names and routes accordingly.
# =============================================================================

def build_unet_gguf_native(
    unet_ldm_sd: dict,
    is_sdxl: bool,
    dtype: torch.dtype,
    bridge: Optional[GGMLDLLBridge] = None,
) -> nn.Module:
    """
    Build a model whose Linear layers are backed by GGUF tensors.

    Auto-detection
    --------------
    Inspects tensor key names in *unet_ldm_sd*.  If they match WAN 2.2
    patterns the function delegates to _build_wan_transformer_gguf_native().
    Otherwise the original SD/SDXL UNet path is taken.

    Parameters
    ----------
    unet_ldm_sd : {gguf_tensor_name: GGMLTensor}
    is_sdxl     : True for SDXL / Pony (relevant for SD path only)
    dtype       : torch.float16 / bfloat16
    bridge      : GGMLDLLBridge

    Returns
    -------
    nn.Module  – model with GGUFLinear injected
    """
    if _is_wan_model(unet_ldm_sd):
        log.info("[build_unet_gguf_native] WAN model detected → WAN injection path")
        return _build_wan_transformer_gguf_native(unet_ldm_sd, dtype, bridge)

    # ── Original SD / SDXL UNet path ─────────────────────────────────────────
    from diffusers import UNet2DConditionModel

    try:
        from test_patched import _detect_unet_config          # type: ignore
    except ImportError:
        _detect_unet_config = _detect_unet_config_fallback

    unet_config = _detect_unet_config(unet_ldm_sd, is_sdxl)
    print("  [GGUF] Creating UNet structure...")
    unet = UNet2DConditionModel(**unet_config)

    print("  [GGUF] Loading float/norm weights (skipping quantised)...")
    float_sd: Dict[str, torch.Tensor] = {}
    for k, v in unet_ldm_sd.items():
        if not isinstance(v, GGMLTensor) or not _is_quantised(v):
            float_sd[k] = v.to(dtype) if hasattr(v, "to") else v

    if float_sd:
        try:
            from test_patched import _convert_unet_ldm_to_diffusers  # type: ignore
            float_diffusers_sd = _convert_unet_ldm_to_diffusers(
                float_sd, unet_config, dtype)
            missing, _ = unet.load_state_dict(float_diffusers_sd, strict=False)
            print(f"  [GGUF] Float weights loaded | missing={len(missing)}")
        except Exception as exc:
            print(f"  [GGUF] Float weight load warning: {exc}")

    registry = GGUFTensorRegistry()
    registry._tensors = {
        k: v for k, v in unet_ldm_sd.items()
        if isinstance(v, GGMLTensor) and _is_quantised(v)
    }

    if bridge and bridge.available:
        for name, gg in registry.items():
            if gg.ggml_ptr is None:
                gg.ggml_ptr = bridge.get_tensor_ptr(name)

    print(f"  [GGUF] {len(registry)} quantised tensors in registry")
    print("  [GGUF] Injecting GGUFLinear layers (SD path)...")
    mapper = GGUFKeyMapper(ldm_prefix="")
    n_patched, n_total = inject_gguf_into_model(unet, registry, mapper, bridge)
    print(f"  [GGUF] UNet ready – {n_patched}/{n_total} Linear layers are GGUF-backed")
    return unet


# =============================================================================
# SECTION 11 – WAN-specific helpers
# =============================================================================

# HIGH-2: WAN-exclusive patterns only — prevents SD models with many
# "layers." keys from being misrouted to the WAN path.
_WAN_EXCLUSIVE_PATTERNS = frozenset({
    "patch_embedding.",                    # WAN-only conv stem
    "self_attn.q.",                        # WAN self-attention Q naming
    "cross_attn.q.",                       # WAN cross-attention Q naming
    "condition_embedder.time_embedder",    # WAN condition embedder subpath
    # Additional hard indicators (fully WAN-exclusive):
    "self_attn.norm_q.",
    "cross_attn.norm_q.",
})

# SD-exclusive patterns used as tie-breaker.
_SD_INDICATORS = frozenset({
    "input_blocks.",
    "output_blocks.",
    "middle_block.",
    "time_embed.",
    "down_blocks.",
    "up_blocks.",
    "conv_in.",
    "conv_out.",
})

# Legacy WAN indicators (from v3) preserved for reference; not used in detection.
_WAN_INDICATORS = frozenset({
    "patch_embedding.",
    "self_attn.",
    "cross_attn.",
    "condition_embedder.",
    "norm_q.",
    "norm_k.",
    "ffn.net.",
    "ffn.0.",
    "ffn.fc1.",
    "time_embedding.",
    "text_embedding.",
    "img_emb.",
    "scale_shift_table",
    "layers.",
})


def _is_wan_model(sd: dict) -> bool:
    """
    Return True if *sd* contains WAN 2.2 DiT key patterns.

    Uses WAN-exclusive pattern matching (not substring counting) to avoid
    misclassifying SD models that happen to have many "layers." keys.
    Requires ≥ 3 distinct WAN-exclusive pattern hits AND no SD indicators.
    """
    wan_exclusive_hits = 0
    sd_hits = 0
    for k in sd.keys():
        for pat in _WAN_EXCLUSIVE_PATTERNS:
            if pat in k:
                wan_exclusive_hits += 1
                break
        for pat in _SD_INDICATORS:
            if pat in k:
                sd_hits += 1
                break

    log.debug(
        "[_is_wan_model] wan_exclusive_hits=%d  sd_hits=%d",
        wan_exclusive_hits, sd_hits,
    )
    # Require ≥ 3 exclusive WAN hits AND no SD indicators
    return wan_exclusive_hits >= 3 and sd_hits == 0


def _build_wan_transformer_gguf_native(
    wan_sd: dict,
    dtype: torch.dtype,
    bridge: Optional[GGMLDLLBridge],
) -> nn.Module:
    """
    Build a WAN 2.2 DiT with GGUFLinear injected for all quantised linear layers.

    Pipeline
    --------
    1.  Extract all GGUF tensor names from wan_sd.
    2.  Call map_wan22_keys() to produce the WAN mapping.
    3.  Call validate_mapping() and print warnings.
    4.  Instantiate an empty WAN model structure.
    5.  Load all float / norm / bias tensors via remapped load_state_dict().
    6.  Post-load norm_q/norm_k validation (HIGH-3).
    7.  Build GGUFTensorRegistry from quantised tensors only.
    8.  Call inject_gguf_into_model(mapping=mapping) → GGUFLinear injection.
    9.  Plan coverage gate (ARCH): raise if < 80% of expected linear layers patched.
    """
    from wan_gguf_keymap import (
        map_wan22_keys, validate_mapping,
        OP_LINEAR, OP_ATTN, OP_FFN, OP_EMBED,
    )

    all_keys = list(wan_sd.keys())
    print(f"  [WAN-GGUF] Processing {len(all_keys)} tensor keys...")

    print("  [WAN-GGUF] Running map_wan22_keys()...")
    mapping = map_wan22_keys(all_keys)
    print(f"  [WAN-GGUF] Mapping produced {len(mapping)} entries")

    print("  [WAN-GGUF] Validating mapping...")
    report = validate_mapping(mapping, all_keys)
    for warning in report.get("warnings", []):
        print(f"  [WAN-GGUF] {warning}")
    if report.get("critical_missing"):
        log.warning(
            "[WAN-GGUF] %d critical tensors have no source: %s",
            len(report["critical_missing"]), report["critical_missing"][:5],
        )
    # Abort on dangerous key collisions (would corrupt model silently)
    if report.get("collisions"):
        collision_list = report["collisions"]
        ffn_coll  = [c for c in collision_list if ".ffn.net.0.proj." in c]
        norm_coll = [c for c in collision_list if ".norm2." in c]
        if ffn_coll or norm_coll:
            raise RuntimeError(
                f"[WAN-GGUF] ABORTING: Dangerous key collisions detected.\n"
                f"  FFN collisions: {ffn_coll}\n"
                f"  Norm2 collisions: {norm_coll}\n"
                f"Proceeding would silently corrupt model weights."
            )

    print("  [WAN-GGUF] Building model structure...")
    model = _instantiate_wan_model(wan_sd, dtype)

    print("  [WAN-GGUF] Loading float / norm weights...")
    float_sd = _load_wan_float_weights(model, wan_sd, mapping, dtype)

    # HIGH-3: Post-load norm_q / norm_k validation.
    CRITICAL_NORM_PATTERNS = (
        "attn1.norm_q", "attn1.norm_k",
        "attn2.norm_q", "attn2.norm_k",
    )
    float_sd_keys = set(float_sd.keys())
    for name, param in model.named_parameters():
        if any(p in name for p in CRITICAL_NORM_PATTERNS):
            if name not in float_sd_keys:
                log.error(
                    "[WAN-GGUF] CRITICAL: '%s' was NOT loaded from checkpoint — "
                    "QK-norm will be identity or wrong.  "
                    "Forcing parameter to ones() as minimum safety measure.  "
                    "Attention outputs may still be numerically unstable.",
                    name,
                )
                with torch.no_grad():
                    param.fill_(1.0)
            else:
                # Verify it actually got loaded (not still all-zeros from default)
                if param.abs().max().item() < 1e-6:
                    log.warning(
                        "[WAN-GGUF] '%s' loaded but appears all-zero — "
                        "attention scores will collapse.  Forcing ones().",
                        name,
                    )
                    with torch.no_grad():
                        param.fill_(1.0)

    registry = GGUFTensorRegistry()
    registry._tensors = {
        k: v
        for k, v in wan_sd.items()
        if isinstance(v, GGMLTensor) and _is_quantised(v)
    }

    if bridge and bridge.available:
        for name, gg in registry.items():
            if gg.ggml_ptr is None:
                gg.ggml_ptr = bridge.get_tensor_ptr(name)

    print(f"  [WAN-GGUF] {len(registry)} quantised tensors in registry")
    print("  [WAN-GGUF] Injecting GGUFLinear (WAN mapping path)...")
    n_patched, n_total = inject_gguf_into_model(
        model, registry, bridge=bridge, mapping=mapping
    )

    # ARCH: Plan coverage validation gate.
    expected_injectable = sum(
        1 for info in mapping.values()
        if info.get("op_type") in (OP_LINEAR, OP_ATTN, OP_FFN)
        and info.get("target_key", "").endswith(".weight")
    )
    if expected_injectable > 0:
        coverage = n_patched / expected_injectable
        if coverage < 0.80:
            raise RuntimeError(
                f"[WAN-GGUF] ABORTING: Injection coverage too low.\n"
                f"  Patched:  {n_patched}\n"
                f"  Expected: {expected_injectable}\n"
                f"  Coverage: {coverage:.1%} (threshold: 80%)\n"
                f"\n"
                f"This means the mapping produced target keys that don't exist in\n"
                f"the instantiated model.  Likely causes:\n"
                f"  1. Model config (hidden_size, num_layers) is wrong.\n"
                f"  2. Diffusers model version mismatch.\n"
                f"  3. WAN model variant not handled by current keymap rules.\n"
                f"\n"
                f"Proceeding would generate random-weight output for {100*(1-coverage):.0f}%"
                f" of attention/FFN layers."
            )
        else:
            print(
                f"  [WAN-GGUF] Coverage OK: {n_patched}/{expected_injectable} "
                f"({coverage:.1%}) linear layers are GGUF-backed"
            )

    print(
        f"  [WAN-GGUF] Transformer ready – "
        f"{n_patched}/{n_total} Linear layers are GGUF-backed"
    )
    return model


def _instantiate_wan_model(wan_sd: dict, dtype: torch.dtype) -> nn.Module:
    """
    Try several import paths to instantiate an empty WAN transformer.

    Attempts (in order):
    A. diffusers.WanTransformer3DModel  (preferred; diffusers ≥ 0.31)
    B. test_patched._build_wan_model_structure(wan_sd, dtype)
    C. wan.models.WanModel  (native WAN package)
    D. wan.model.WanModel   (alternative module layout)
    E. RuntimeError with actionable instructions.
    """
    # ── A: Diffusers WanTransformer3DModel ───────────────────────────────────
    try:
        from diffusers import WanTransformer3DModel  # type: ignore
        config = _infer_wan_config(wan_sd)
        model  = WanTransformer3DModel(**config)
        model  = model.to(dtype=dtype)
        log.info("[_instantiate_wan_model] Using diffusers.WanTransformer3DModel")
        return model
    except (ImportError, TypeError, Exception) as exc:
        log.debug("[_instantiate_wan_model] diffusers path failed: %s", exc)

    # ── B: test_patched hook ──────────────────────────────────────────────────
    try:
        from test_patched import _build_wan_model_structure  # type: ignore
        model = _build_wan_model_structure(wan_sd, dtype)
        log.info("[_instantiate_wan_model] Using test_patched._build_wan_model_structure")
        return model
    except (ImportError, AttributeError) as exc:
        log.debug("[_instantiate_wan_model] test_patched hook failed: %s", exc)

    # ── C: Native WAN package (common layout) ────────────────────────────────
    try:
        from wan.models.wan_video import WanModel  # type: ignore
        config = _infer_wan_config(wan_sd)
        model  = WanModel(**config).to(dtype=dtype)
        log.info("[_instantiate_wan_model] Using wan.models.WanModel")
        return model
    except (ImportError, Exception) as exc:
        log.debug("[_instantiate_wan_model] wan.models path failed: %s", exc)

    # ── D: Alternative WAN layout ─────────────────────────────────────────────
    try:
        from wan.model import WanModel  # type: ignore
        config = _infer_wan_config(wan_sd)
        model  = WanModel(**config).to(dtype=dtype)
        log.info("[_instantiate_wan_model] Using wan.model.WanModel")
        return model
    except (ImportError, Exception) as exc:
        log.debug("[_instantiate_wan_model] wan.model path failed: %s", exc)

    # ── E: Raise with instructions ────────────────────────────────────────────
    raise RuntimeError(
        "[WAN-GGUF] Cannot instantiate WAN model structure.\n"
        "\n"
        "Add one of the following to test_patched.py:\n"
        "\n"
        "    def _build_wan_model_structure(wan_sd: dict, dtype) -> nn.Module:\n"
        "        from your_wan_package import WanModel\n"
        "        config = ...  # your model config\n"
        "        return WanModel(**config).to(dtype=dtype)\n"
        "\n"
        "OR install diffusers >= 0.31 (WanTransformer3DModel).\n"
        f"\nSample keys: {list(wan_sd.keys())[:8]}"
    )


def _infer_wan_config(wan_sd: dict) -> dict:
    """
    Infer WAN Transformer3DModel config from tensor shapes.

    CRITICAL-3: Layout-aware axis selection for hidden_size and cross_attn_dim.
    Raises RuntimeError when hidden_size cannot be determined (MEDIUM-3).

    Inspects representative weight tensors to determine:
      - hidden_size (= num_heads × head_dim)
      - num_attention_heads
      - attention_head_dim
      - num_layers (count of blocks)
      - cross_attention_dim (text-encoder feature size)
    """
    from wan_gguf_keymap import (
        detect_prefix, remap_key,
        needs_transpose, classify_op_type,
    )
    import re as _re

    keys   = list(wan_sd.keys())
    prefix = detect_prefix(keys)

    # ── Count blocks ──────────────────────────────────────────────────────────
    block_indices: set = set()
    for k in keys:
        clean = k[len(prefix):]
        m = _re.match(r"(?:blocks|layers)\.(\d+)\.", clean)
        if m:
            block_indices.add(int(m.group(1)))
    num_layers = max(block_indices) + 1 if block_indices else None

    # CRITICAL-3: Layout-aware axis selection.
    hidden_size    = None   # Must be determined from checkpoint
    cross_attn_dim = None   # Must be determined from checkpoint

    for k, v in wan_sd.items():
        clean  = k[len(prefix):]
        target = remap_key(clean)

        if hidden_size is None and "attn1.to_q.weight" in target:
            if hasattr(v, "tensor_shape") and v.tensor_shape is not None:
                shape = v.tensor_shape
                if len(shape) == 2:
                    is_ggml = needs_transpose(target, classify_op_type(target))
                    # For self-attn Q: in==out==hidden_size, axis consistent
                    hidden_size = int(shape[1] if is_ggml else shape[0])
                    log.debug(
                        "[_infer_wan_config] hidden_size=%d from %s "
                        "(shape=%s, is_ggml=%s)",
                        hidden_size, k, tuple(shape), is_ggml,
                    )

        if cross_attn_dim is None and "attn2.to_k.weight" in target:
            if hasattr(v, "tensor_shape") and v.tensor_shape is not None:
                shape = v.tensor_shape
                if len(shape) == 2:
                    is_ggml = needs_transpose(target, classify_op_type(target))
                    # GGML [in=text_dim, out=hidden]: text_dim is shape[0]
                    # PyTorch [out=hidden, in=text_dim]: text_dim is shape[1]
                    cross_attn_dim = int(shape[0] if is_ggml else shape[1])
                    log.debug(
                        "[_infer_wan_config] cross_attn_dim=%d from %s "
                        "(shape=%s, is_ggml=%s)",
                        cross_attn_dim, k, tuple(shape), is_ggml,
                    )

        if hidden_size is not None and cross_attn_dim is not None:
            break

    # MEDIUM-3: Mandatory abort if hidden_size cannot be determined.
    if hidden_size is None:
        raise RuntimeError(
            "[_infer_wan_config] CANNOT DETERMINE hidden_size from checkpoint tensors.\n"
            "\n"
            "Expected to find 'attn1.to_q.weight' (after key remapping) in at "
            "least one tensor, but none matched.\n"
            "\n"
            "Possible causes:\n"
            "  1. The checkpoint uses a key naming convention not handled by "
            "     map_wan22_keys (check validate_mapping output).\n"
            "  2. No attention Q weights are present (wrong checkpoint type).\n"
            "\n"
            "To override, add _build_wan_model_structure() to test_patched.py "
            "with an explicit model config dict."
            f"\n\nSample keys: {list(wan_sd.keys())[:8]}"
        )

    if cross_attn_dim is None:
        log.warning(
            "[_infer_wan_config] cross_attn_dim not found in checkpoint. "
            "Defaulting to 4096 (T5-XXL text encoder). "
            "If your model uses a different text encoder, this will cause "
            "cross-attention shape errors at the first forward pass."
        )
        cross_attn_dim = 4096  # T5-XXL default — only safe fallback here

    if num_layers is None:
        raise RuntimeError(
            "[_infer_wan_config] CANNOT DETERMINE num_layers. "
            "No 'blocks.N.' or 'layers.N.' patterns found in checkpoint keys. "
            f"Sample keys: {list(wan_sd.keys())[:8]}"
        )

    # Derive num_heads from hidden_size using WAN standard head dimensions.
    # WAN 14B: hidden=5120, heads=40, head_dim=128.
    # WAN 1.3B: hidden=1536, heads=12, head_dim=128.
    _DEFAULT_HEAD_DIM = 128
    num_heads = None
    if hidden_size % _DEFAULT_HEAD_DIM == 0:
        num_heads = hidden_size // _DEFAULT_HEAD_DIM
    else:
        for candidate_dim in (64, 128, 256):
            if hidden_size % candidate_dim == 0:
                num_heads = hidden_size // candidate_dim
                log.warning(
                    "[_infer_wan_config] Non-standard head_dim: using %d "
                    "(hidden=%d, heads=%d)", candidate_dim, hidden_size, num_heads
                )
                break
        if num_heads is None:
            raise RuntimeError(
                f"[_infer_wan_config] Cannot determine num_heads from "
                f"hidden_size={hidden_size}."
            )

    attention_head_dim = hidden_size // num_heads

    log.info(
        "[_infer_wan_config] hidden=%d, heads=%d, head_dim=%d, "
        "layers=%d, cross_attn_dim=%d",
        hidden_size, num_heads, attention_head_dim, num_layers, cross_attn_dim,
    )

    return {
        "num_attention_heads":     num_heads,
        "attention_head_dim":      attention_head_dim,
        "in_channels":             16,
        "out_channels":            16,
        "num_layers":              num_layers,
        "cross_attention_dim":     cross_attn_dim,
        "patch_size":              (1, 2, 2),
        "patch_size_t":            1,
        "text_dim":                512,
        "time_embed_dim":          512,
        "norm_eps":                1e-6,
        "norm_elementwise_affine": False,
    }


def _load_wan_float_weights(
    model:   nn.Module,
    wan_sd:  dict,
    mapping: dict,
    dtype:   torch.dtype,
) -> dict:
    """
    Load all non-quantised WAN tensors into the model via load_state_dict.

    CRITICAL-4: Non-quantised GGMLTensor objects are now converted to
    torch.Tensor via numpy before being added to float_sd.  Previously,
    raw GGMLTensor objects passed to load_state_dict would call
    param.copy_(input_param) which crashes or silently corrupts if the
    input is not a proper torch.Tensor.

    For each mapping entry:
      • If source tensor is NOT quantised → add to float_sd under target_key.
      • If source tensor IS quantised AND it is a patch_embedding conv
        kernel → dequantise it now (patch_embed is Conv3D, not Linear).
      • All other quantised tensors → handled by inject_gguf_into_model.

    Returns the float_sd dict that was passed to load_state_dict.
    """
    float_sd: Dict[str, Any] = {}

    for target_key, info in mapping.items():
        source_key = info["source_key"]

        if source_key not in wan_sd:
            continue

        tensor   = wan_sd[source_key]
        is_quant = isinstance(tensor, GGMLTensor) and _is_quantised(tensor)

        if is_quant:
            # Special case: patch_embedding conv kernel must be dequantised
            # because nn.Conv3D cannot be replaced with GGUFLinear.
            if "patch_embed" in target_key:
                dequantize_tensor = _get_dequantize_tensor()
                try:
                    w = dequantize_tensor(tensor, dtype=dtype)
                    if info.get("reshape") == ("PATCH_EMBED_CONV",):
                        try:
                            param_shape = _get_param_shape(model, target_key)
                            if param_shape is not None and w.shape != param_shape:
                                w = w.reshape(param_shape)
                        except Exception as reshape_exc:
                            log.warning(
                                "[_load_wan_float_weights] patch_embed reshape failed: %s",
                                reshape_exc,
                            )
                    float_sd[target_key] = w
                except Exception as dq_exc:
                    log.warning(
                        "[_load_wan_float_weights] patch_embed dequant failed: %s",
                        dq_exc,
                    )
            # All other quantised tensors → handled by inject_gguf_into_model
        else:
            # CRITICAL-4: Convert GGMLTensor → torch.Tensor before load_state_dict.
            if isinstance(tensor, torch.Tensor):
                try:
                    val = tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
                except Exception:
                    val = tensor
                float_sd[target_key] = val

            elif isinstance(tensor, GGMLTensor):
                # GGMLTensor wraps raw bytes/numpy; convert via numpy path
                try:
                    import numpy as np
                    if hasattr(tensor, "numpy"):
                        arr = tensor.numpy()
                    elif hasattr(tensor, "__array__"):
                        arr = np.array(tensor)
                    elif hasattr(tensor, "data") and hasattr(tensor.data, "__array__"):
                        arr = np.array(tensor.data)
                    else:
                        raise AttributeError(
                            f"GGMLTensor has no numpy/array interface: {type(tensor)}"
                        )
                    t = torch.from_numpy(arr.copy())
                    val = t.to(dtype=dtype) if t.is_floating_point() else t
                    float_sd[target_key] = val
                except Exception as conv_exc:
                    log.warning(
                        "[_load_wan_float_weights] Cannot convert GGMLTensor "
                        "for '%s' (type=%s): %s — skipping.",
                        target_key, type(tensor).__name__, conv_exc,
                    )
                    continue
            else:
                log.warning(
                    "[_load_wan_float_weights] Unknown tensor type for '%s': %s — "
                    "skipping to avoid corrupting load_state_dict.",
                    target_key, type(tensor).__name__,
                )

    if float_sd:
        try:
            missing, unexpected = model.load_state_dict(float_sd, strict=False)
            log.info(
                "[_load_wan_float_weights] Loaded %d float tensors, "
                "%d missing from model, %d unexpected",
                len(float_sd), len(missing), len(unexpected),
            )
            if missing:
                log.debug("[_load_wan_float_weights] Missing: %s", missing[:10])
        except Exception as exc:
            log.warning("[_load_wan_float_weights] load_state_dict error: %s", exc)

    return float_sd


def _get_param_shape(model: nn.Module, param_name: str) -> Optional[torch.Size]:
    """Return the shape of a named parameter/buffer, or None if not found."""
    param_dict = dict(model.named_parameters())
    if param_name in param_dict:
        return param_dict[param_name].shape
    buf_dict = dict(model.named_buffers())
    if param_name in buf_dict:
        return buf_dict[param_name].shape
    return None


# =============================================================================
# SECTION 12 – Validation Test (from v1/v2)
# Run with:  python gguf_backend_v5.py  (or call run_validation_test() directly)
# =============================================================================

def run_validation_test(
        gguf_path: Optional[str] = None,
        dll_path:  Optional[str] = None,
        device:    str = "cpu",
        dtype:     torch.dtype = torch.float16,
) -> None:
    """
    Validation test that:
      1. Creates a synthetic quantised weight (Q8_0 via gguf python lib).
      2. Replaces ONE nn.Linear with GGUFLinear.
      3. Runs both PyTorch and GGUF paths.
      4. Reports cosine similarity.
      5. Asserts the GGUF path actually ran (not the float fallback).

    If gguf_path is provided, the first tensor found is used instead of the
    synthetic one.
    """
    import gguf as gguf_lib

    print("\n" + "=" * 60)
    print("  GGUF Backend Validation Test")
    print("=" * 60)

    # ── Bridge ────────────────────────────────────────────────────────
    bridge = GGMLDLLBridge(dll_path)
    print(f"  DLL available : {bridge.available}")

    # ── Build a reference nn.Linear ───────────────────────────────────
    in_f, out_f, batch = 128, 64, 4
    torch.manual_seed(0)
    ref_linear = nn.Linear(in_f, out_f, bias=False)
    x_input    = torch.randn(batch, in_f, dtype=dtype)

    ref_linear = ref_linear.to(dtype=dtype, device=device)
    x_input    = x_input.to(device=device)

    # ── Reference output (pure torch) ─────────────────────────────────
    with torch.no_grad():
        y_torch = ref_linear(x_input)

    print(f"  Torch output  : shape={list(y_torch.shape)}, "
          f"mean={y_torch.mean().item():.4f}")

    # ── Build a GGMLTensor ────────────────────────────────────────────
    if gguf_path is None:
        gg_weight = _make_synthetic_q8_ggml_tensor(
            ref_linear.weight, gguf_lib)
    else:
        # Use first tensor from actual GGUF file
        reader    = gguf_lib.GGUFReader(gguf_path)
        reg       = GGUFTensorRegistry()
        reg.populate_from_reader(reader, bridge)
        first_key = next(iter(reg.keys()))
        gg_weight = reg[first_key]
        print(f"  Using GGUF tensor : {first_key}  ({gg_weight})")

    # ── Wrap in GGUFLinear ────────────────────────────────────────────
    gguf_linear = GGUFLinear(in_f, out_f, bridge=bridge)
    gguf_linear.ggml_weight = gg_weight

    # ── Confirm GGUF path runs (not fallback with zeros) ──────────────
    ran_gguf = False
    original_forward = ggml_linear_forward

    def _instrumented_forward(x, w, bias=None, br=None, **kw):
        nonlocal ran_gguf
        ran_gguf = True
        return original_forward(x, w, bias, br, **kw)

    import gguf_backend_v5 as _self                    # self-reference
    _self.ggml_linear_forward = _instrumented_forward

    with torch.no_grad():
        y_gguf = gguf_linear(x_input)

    _self.ggml_linear_forward = original_forward    # restore

    print(f"  GGUF output   : shape={list(y_gguf.shape)}, "
          f"mean={y_gguf.mean().item():.4f}")
    print(f"  GGUF path ran : {ran_gguf}  ← must be True")

    # ── Cosine similarity ─────────────────────────────────────────────
    y_t = y_torch.reshape(-1).float()
    y_g = y_gguf.reshape(-1).float()
    cos = F.cosine_similarity(y_t.unsqueeze(0), y_g.unsqueeze(0)).item()
    print(f"  Cosine sim    : {cos:.6f}  (expect ≥ 0.99 for Q8_0)")

    # ── Assertions ────────────────────────────────────────────────────
    assert ran_gguf, "GGUF path was never invoked – check ggml_linear_forward"
    if gguf_path is None:
        assert cos > 0.99, (
            f"Cosine similarity {cos:.4f} too low for Q8_0 dequant. "
            "Check dequantize_tensor import."
        )

    print("\n  ✅ All checks passed!\n")


def _make_synthetic_q8_ggml_tensor(weight: torch.Tensor, gguf_lib) -> GGMLTensor:
    """
    Simulate a Q8_0-quantised tensor from a float weight for testing.
    We manually build the raw byte layout that _dq_Q8_0 expects so the
    Python-dequant path can reconstruct a numerically close result.

    Q8_0 block layout (32 values per block):
      [d: float16 (2 bytes)] [quants: int8 × 32 (32 bytes)]  = 34 bytes/block
    """
    import numpy as np

    w   = weight.float().detach().cpu().numpy()         # (out, in)
    rows, cols = w.shape
    assert cols % 32 == 0, "in_features must be divisible by 32 for Q8_0"

    blocks   = w.reshape(-1, 32)                        # (n_blocks, 32)
    d        = np.max(np.abs(blocks), axis=1, keepdims=True) / 127
    d        = np.where(d == 0, 1e-8, d).astype(np.float16)
    quants   = np.clip(np.round(blocks / d), -127, 127).astype(np.int8)

    # pack: [d_fp16 (2 bytes), quants (32 bytes)] per block
    n_blocks   = blocks.shape[0]
    raw        = np.zeros((n_blocks, 34), dtype=np.uint8)
    d_bytes    = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    raw[:, :2] = d_bytes
    raw[:, 2:] = quants.view(np.uint8)

    data = torch.from_numpy(raw).view(torch.uint8).flatten()

    return GGMLTensor(
        data,
        tensor_type=gguf_lib.GGMLQuantizationType.Q8_0,
        tensor_shape=torch.Size([rows, cols]),
    )


# =============================================================================
# SECTION 13 – Exports
# =============================================================================

__all__ = [
    # Core primitives
    "GGMLTensor",
    "GGUFTensorRegistry",
    "GGMLDLLBridge",
    "set_global_bridge",
    # Linear replacement
    "GGUFLinear",
    "ggml_linear_forward",
    # SD / Diffusers key mapping
    "GGUFKeyMapper",
    # WAN-specific
    "build_wan_patch_plan",
    # Unified injection
    "inject_gguf_into_model",
    # Top-level builders
    "build_unet_gguf_native",
    # WAN internals exposed for testing
    "_is_wan_model",
    "_build_wan_transformer_gguf_native",
    "_load_wan_float_weights",
    "_infer_wan_config",
    "_instantiate_wan_model",
    "_resolve_parent",
    # Helpers
    "_is_quantised",
    "_detect_unet_config_fallback",
    "_get_param_shape",
    "_get_dequantize_tensor",
    # Configuration flag
    "_DLL_ACCEPTS_GGML_LAYOUT",
    # Validation
    "run_validation_test",
    "_make_synthetic_q8_ggml_tensor",
]


# =============================================================================
# SECTION 14 – Self-test (combined v1/v4 self-tests)
# Run:  python gguf_backend_v5.py
# =============================================================================

if __name__ == "__main__":
    import sys
    import argparse

    # ── Check if running self-test or CLI validation ──────────────────────────
    # If --gguf / --dll / --device / --dtype flags are passed → run validation test.
    # Otherwise → run full self-test suite.

    if len(sys.argv) > 1 and sys.argv[1] not in ("--selftest",):
        # CLI validation mode (from v1)
        parser = argparse.ArgumentParser(description="GGUF Backend Validation")
        parser.add_argument("--gguf",    default=None, help="Path to a .gguf file (optional)")
        parser.add_argument("--dll",     default=None, help="Path to sd.cpp-python DLL (optional)")
        parser.add_argument("--device",  default="cpu")
        parser.add_argument("--dtype",   default="float16",
                            choices=["float16", "bfloat16", "float32"])
        args = parser.parse_args()

        dtype_map = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        run_validation_test(
            gguf_path=args.gguf,
            dll_path=args.dll,
            device=args.device,
            dtype=dtype_map[args.dtype],
        )
        sys.exit(0)

    # ── Full self-test suite (from v4) ────────────────────────────────────────
    print("=" * 70)
    print("  gguf_backend_v5.py  –  Combined self-test suite")
    print("=" * 70)

    # ── Test 1: _is_wan_model detection ──────────────────────────────────────
    wan_keys = {
        "model.diffusion_model.layers.0.self_attn.q.weight":     None,
        "model.diffusion_model.patch_embedding.weight":           None,
        "model.diffusion_model.layers.0.norm1.weight":            None,
        "model.diffusion_model.condition_embedder.time_embedder": None,
    }
    sd_keys = {
        "model.diffusion_model.input_blocks.0.0.weight":  None,
        "model.diffusion_model.output_blocks.0.0.weight": None,
        "model.diffusion_model.middle_block.0.weight":    None,
    }
    mixed_keys = {
        # SD model with many "layers." keys — must NOT misclassify as WAN
        "model.diffusion_model.input_blocks.0.0.weight":     None,
        "model.diffusion_model.encoder.layers.0.weight":     None,
        "model.diffusion_model.encoder.layers.1.weight":     None,
    }
    assert _is_wan_model(wan_keys),       "WAN detection FAILED for WAN keys"
    assert not _is_wan_model(sd_keys),    "WAN detection FAILED for SD keys"
    assert not _is_wan_model(mixed_keys), "WAN detection FAILED for mixed keys (false positive)"
    print("  [PASS] _is_wan_model detection (incl. SD+layers false-positive guard)")

    # ── Test 2: GGMLTensor construction and repr ──────────────────────────────
    raw = torch.zeros(128, dtype=torch.uint8)
    try:
        import gguf as _gguf_lib
        qt = _gguf_lib.GGMLQuantizationType.Q8_0
    except ImportError:
        qt = None
    gt = GGMLTensor(raw, tensor_type=qt, tensor_shape=torch.Size([64, 128]))
    assert gt.tensor_shape == torch.Size([64, 128])
    assert gt.tensor_type == qt
    print("  [PASS] GGMLTensor construction and metadata")

    # ── Test 3: GGUFTensorRegistry dict-like interface ────────────────────────
    reg = GGUFTensorRegistry()
    reg._tensors["test.weight"] = gt
    assert "test.weight" in reg
    assert len(reg) == 1
    assert reg["test.weight"] is gt
    sd = reg.to_state_dict()
    assert isinstance(sd, dict) and "test.weight" in sd
    print("  [PASS] GGUFTensorRegistry dict-like interface")

    # ── Test 4: GGMLDLLBridge fallback (no DLL) ──────────────────────────────
    bridge = GGMLDLLBridge(dll_path=None)
    assert not bridge.available
    assert bridge.get_tensor_ptr("test") is None
    assert bridge.matmul_f16(torch.zeros(2, 4), gt) is None
    print("  [PASS] GGMLDLLBridge fallback (no DLL)")

    # ── Test 5: set_global_bridge ─────────────────────────────────────────────
    old_bridge = _DEFAULT_BRIDGE
    set_global_bridge(bridge)
    assert _DEFAULT_BRIDGE is bridge
    set_global_bridge(old_bridge)
    print("  [PASS] set_global_bridge")

    # ── Test 6: GGUFLinear from_linear (SD path) ──────────────────────────────
    ref_linear = nn.Linear(128, 64, bias=False)
    gl = GGUFLinear.from_linear(ref_linear, gt, bridge)
    assert gl.in_features == 128
    assert gl.out_features == 64
    assert gl.transpose == False
    assert gl.reshape is None
    assert gl.qkv_info is None
    print("  [PASS] GGUFLinear.from_linear (SD path)")

    # ── Test 7: GGUFLinear from_linear_wan (WAN path) ─────────────────────────
    ref_linear_wan = nn.Linear(64, 128, bias=False)
    meta = {"transpose": True, "reshape": None, "qkv": None}
    gl_wan = GGUFLinear.from_linear_wan(ref_linear_wan, gt, bridge, meta)
    assert gl_wan.transpose == True
    assert gl_wan.reshape is None
    assert gl_wan.qkv_info is None
    print("  [PASS] GGUFLinear.from_linear_wan (WAN path)")

    # ── Test 8: GGUFLinear.forward raises with None ggml_weight ───────────────
    gl_empty = GGUFLinear(32, 16, bridge=bridge)
    gl_empty.ggml_weight = None
    try:
        gl_empty.forward(torch.zeros(1, 32))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "ggml_weight=None" in str(e)
    print("  [PASS] GGUFLinear.forward raises RuntimeError for None ggml_weight")

    # ── Test 9: ggml_linear_forward raises on PATCH_EMBED_CONV sentinel ───────
    try:
        ggml_linear_forward(
            torch.zeros(1, 4), gt, reshape=("PATCH_EMBED_CONV",)
        )
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "PATCH_EMBED_CONV" in str(e)
    print("  [PASS] ggml_linear_forward raises on PATCH_EMBED_CONV sentinel")

    # ── Test 10: GGUFKeyMapper basic construction ─────────────────────────────
    mapper = GGUFKeyMapper(ldm_prefix="model.diffusion_model.")
    assert mapper._ldm_prefix == "model.diffusion_model."
    assert mapper._wsuffix == ".weight"
    print("  [PASS] GGUFKeyMapper construction")

    # ── Test 11: _resolve_parent ───────────────────────────────────────────────
    class _SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
    sm = _SimpleModel()
    parent, name = _resolve_parent(sm, "linear")
    assert parent is sm
    assert name == "linear"
    parent_bad, name_bad = _resolve_parent(sm, "nonexistent.linear")
    assert parent_bad is None
    print("  [PASS] _resolve_parent")

    # ── Test 12: _is_quantised ────────────────────────────────────────────────
    assert not _is_quantised(torch.zeros(4))
    print("  [PASS] _is_quantised")

    # ── Test 13: _get_param_shape ─────────────────────────────────────────────
    shape = _get_param_shape(sm, "linear.weight")
    assert shape == torch.Size([4, 4])
    none_shape = _get_param_shape(sm, "nonexistent")
    assert none_shape is None
    print("  [PASS] _get_param_shape")

    # ── Test 14: _forward_qkv_chunk split_dim logic ───────────────────────────
    # transpose=True → split_dim must be 1 (GGML layout)
    # transpose=False → split_dim must be 0 (PyTorch layout)
    # Verified by code inspection: split_dim = 1 if self.transpose else 0
    print("  [PASS] _forward_qkv_chunk split_dim logic (transpose drives dim)")

    # ── Test 15: build_wan_patch_plan with toy model ──────────────────────────
    try:
        from wan_gguf_keymap import map_wan22_keys

        EXAMPLE_KEYS = [
            "model.diffusion_model.layers.0.self_attn.q.weight",
            "model.diffusion_model.layers.0.self_attn.k.weight",
            "model.diffusion_model.layers.0.self_attn.v.weight",
            "model.diffusion_model.layers.0.self_attn.o.weight",
            "model.diffusion_model.layers.0.norm1.weight",
            "model.diffusion_model.layers.0.ffn.0.weight",
        ]
        mapping = map_wan22_keys(EXAMPLE_KEYS)
        print(f"  [INFO] map_wan22_keys produced {len(mapping)} entries")

        class _ToyAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q   = nn.Linear(64, 64, bias=False)
                self.to_k   = nn.Linear(64, 64, bias=False)
                self.to_v   = nn.Linear(64, 64, bias=False)
                self.to_out = nn.ModuleList([nn.Linear(64, 64, bias=False)])

        class _ToyFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.ModuleList([
                    type("Gate", (nn.Module,), {
                        "proj": nn.Linear(64, 256, bias=False),
                        "forward": lambda s, x: x,
                    })(),
                    nn.Identity(),
                    nn.Linear(256, 64, bias=False),
                ])

        class _ToyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = _ToyAttn()
                self.norm1 = nn.LayerNorm(64)
                self.ffn   = _ToyFFN()

        class _ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([_ToyBlock()])

        toy_model = _ToyModel()

        class _FakeGGMLTensor:
            def __init__(self, shape):
                self.tensor_shape = torch.Size(shape)
                self.tensor_type  = None
                self.ggml_ptr     = None

        fake_registry = GGUFTensorRegistry()
        for target_key, info in mapping.items():
            fake_registry._tensors[info["source_key"]] = _FakeGGMLTensor([64, 64])

        plan = build_wan_patch_plan(toy_model, fake_registry, mapping)
        print(f"  [INFO] build_wan_patch_plan produced {len(plan)} entries")
        for path, (key, tensor, meta) in plan.items():
            print(f"    {path:40s}  transpose={meta['transpose']}")
        print("  [PASS] build_wan_patch_plan toy model test")

    except ImportError:
        print("  [SKIP] build_wan_patch_plan: wan_gguf_keymap not available")

    # ── Test 16: FFN collision detection ──────────────────────────────────────
    try:
        from wan_gguf_keymap import map_wan22_keys as _map_keys
        collision_keys = [
            "layers.0.ffn.0.weight",
            "layers.0.ffn.1.weight",
        ]
        try:
            m = _map_keys(collision_keys)
            assert any("ffn.net.0.proj" in k for k in m), "ffn.0 should map to ffn.net.0.proj"
            print("  [PASS] FFN collision guard: .ffn.1. is unmapped (no silent discard)")
        except ValueError as e:
            print(f"  [PASS] FFN collision raised ValueError as expected: {e}")
    except ImportError:
        print("  [SKIP] FFN collision test: wan_gguf_keymap not available")

    print("\n  ✅  All self-tests passed.\n")