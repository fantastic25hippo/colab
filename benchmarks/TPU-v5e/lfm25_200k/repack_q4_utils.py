import numpy as np, torch

def unpack_q4_0_nibbles_and_scales(raw_bytes, out_features: int, in_features: int):
    num_elements = out_features * in_features
    num_blocks = num_elements // 32
    if isinstance(raw_bytes, bytes):
        raw_u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
    elif isinstance(raw_bytes, torch.Tensor):
        raw_u8 = raw_bytes.detach().cpu().numpy().view(np.uint8).reshape(-1)
    else:
        raw_u8 = np.asarray(raw_bytes, dtype=np.uint8).reshape(-1)
    blocks = raw_u8.reshape(num_blocks, 18)
    scales = blocks[:, :2].copy().view(np.float16).reshape(out_features, in_features // 32)
    qs = blocks[:, 2:18]
    lo = qs & 0x0F
    hi = (qs >> 4) & 0x0F
    u_nibbles = np.concatenate([lo, hi], axis=1).reshape(out_features, in_features)
    return u_nibbles, scales

def pack_nibbles_to_ct_int32(u_nibbles: np.ndarray) -> torch.Tensor:
    out_f, in_f = u_nibbles.shape
    u_reshaped = u_nibbles.reshape(out_f, in_f // 8, 8).astype(np.int32)
    shifts = np.arange(8, dtype=np.int32) * 4
    packed = np.bitwise_or.reduce(u_reshaped << shifts, axis=-1).astype(np.int32)
    return torch.from_numpy(packed)

def unpack_gguf_q4_0_to_ct_int32(
    raw_bytes, out_features: int, in_features: int, group_size: int = 32
):
    u_nibbles, scales_fp16 = unpack_q4_0_nibbles_and_scales(raw_bytes, out_features, in_features)
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int32)
    if group_size == 32:
        w_packed = pack_nibbles_to_ct_int32(u_nibbles)
        w_scale = torch.from_numpy(scales_fp16.copy()).to(torch.bfloat16)
        return w_packed, w_scale, weight_shape
    if group_size == 128:
        out_f, in_f = out_features, in_features
        q_signed = u_nibbles.astype(np.float32) - 8.0
        w_fp32 = (q_signed.reshape(out_f, in_f // 32, 32) * scales_fp16[..., None].astype(np.float32)).reshape(
            out_f, in_f // 128, 128
        )
        absmax = np.max(np.abs(w_fp32), axis=-1, keepdims=True)
        scale_128 = np.maximum(absmax / 7.0, 1e-7)
        q_new = np.clip(np.round(w_fp32 / scale_128), -8.0, 7.0).astype(np.int32) + 8
        u_new = q_new.astype(np.uint8).reshape(out_f, in_f)
        w_packed = pack_nibbles_to_ct_int32(u_new)
        w_scale = torch.from_numpy(scale_128.squeeze(-1).copy()).to(torch.bfloat16)
        return w_packed, w_scale, weight_shape
    raise ValueError(f"Unsupported group_size: {group_size}. Choose 32 or 128.")

def dequantize_q4_0_to_float(raw_bytes, out_features: int, in_features: int) -> np.ndarray:
    u_nibbles, scales = unpack_q4_0_nibbles_and_scales(raw_bytes, out_features, in_features)
    q_signed = u_nibbles.astype(np.float32) - 8.0
    dequant = q_signed.reshape(out_features, in_features // 32, 32) * scales[..., None].astype(np.float32)
    return dequant.reshape(out_features, in_features)

def quantize_bf16_to_w8a8_channel(tensor_bf16):
    if isinstance(tensor_bf16, torch.Tensor):
        w = tensor_bf16.detach().float()
    else:
        w = torch.from_numpy(tensor_bf16).float()
    absmax = torch.amax(torch.abs(w), dim=-1, keepdim=True)
    scale = torch.clamp(absmax / 127.0, min=1e-7).to(torch.bfloat16)
    scale_f = scale.float()
    q = torch.clamp(torch.round(w / scale_f), -127, 127).to(torch.int8)
    return q, scale
