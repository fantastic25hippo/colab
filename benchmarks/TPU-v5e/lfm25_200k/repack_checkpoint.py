import json, os, re, numpy as np, torch
from safetensors.torch import save_file
from .config_builder import build_lfm2_config, save_config
from .repack_q4_utils import unpack_gguf_q4_0_to_ct_int32, dequantize_q4_0_to_float, quantize_bf16_to_w8a8_channel

class ShardWriter:
    def __init__(self, out_dir: str, max_bytes: int = int(1.5 * 1024**3), upload_repo: str = None, token: str = None):
        self.out_dir, self.max_bytes, self.upload_repo, self.token = out_dir, max_bytes, upload_repo, token
        self.current_tensors, self.weight_map, self.shard_idx, self.current_bytes = {}, {}, 1, 0
        os.makedirs(out_dir, exist_ok=True)
        self.api = None
        if upload_repo and token:
            from huggingface_hub import HfApi
            self.api = HfApi(token=token)

    def add(self, name: str, tensor: torch.Tensor):
        t = tensor.contiguous()
        self.current_tensors[name] = t
        self.current_bytes += t.numel() * t.element_size()
        if self.current_bytes >= self.max_bytes: self.flush()

    def flush(self):
        if not self.current_tensors: return
        fname = f"model-{self.shard_idx:05d}.safetensors"
        fpath = os.path.join(self.out_dir, fname)
        save_file(self.current_tensors, fpath)
        for k in self.current_tensors: self.weight_map[k] = fname
        if self.api and self.upload_repo:
            print(f"[REPACK] Uploading {fname} to {self.upload_repo}...", flush=True)
            self.api.upload_file(path_or_fileobj=fpath, path_in_repo=fname, repo_id=self.upload_repo, token=self.token)
            try: os.remove(fpath)
            except Exception: pass
        self.current_tensors, self.current_bytes = {}, 0
        self.shard_idx += 1

    def close(self):
        self.flush()
        idx_doc = {"metadata": {"total_size": 0}, "weight_map": self.weight_map}
        idx_p = os.path.join(self.out_dir, "model.safetensors.index.json")
        with open(idx_p, "w") as f: json.dump(idx_doc, f, indent=2)
        if self.api and self.upload_repo:
            print(f"[REPACK] Uploading model.safetensors.index.json to {self.upload_repo}...", flush=True)
            self.api.upload_file(path_or_fileobj=idx_p, path_in_repo="model.safetensors.index.json", repo_id=self.upload_repo, token=self.token)

MAP_DIRECT = {"token_embd_norm.weight": "model.embedding_norm.weight", "output_norm.weight": "model.embedding_norm.weight"}
MAP_W8A8 = {
    "attn_q.weight": "self_attn.q_proj", "attn_k.weight": "self_attn.k_proj",
    "attn_v.weight": "self_attn.v_proj", "attn_output.weight": "self_attn.out_proj",
    "shortconv.in_proj.weight": "conv.in_proj", "shortconv.out_proj.weight": "conv.out_proj",
    "ffn_gate.weight": "feed_forward.w1", "ffn_down.weight": "feed_forward.w2", "ffn_up.weight": "feed_forward.w3"
}
MAP_BF16 = {
    "attn_norm.weight": "operator_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
    "attn_q_norm.weight": "self_attn.q_layernorm.weight", "attn_k_norm.weight": "self_attn.k_layernorm.weight",
    "shortconv.conv.weight": "conv.conv.weight", "ffn_gate_inp.weight": "feed_forward.gate.weight",
    "exp_probs_b.bias": "feed_forward.expert_bias"
}
MAP_EXPS = {"ffn_gate_exps.weight": "w1", "ffn_down_exps.weight": "w2", "ffn_up_exps.weight": "w3"}

def dequant_tensor_to_bf16(tensor) -> torch.Tensor:
    from gguf import GGMLQuantizationType, quants
    ttype = tensor.tensor_type
    shape = tuple(int(x) for x in reversed(tensor.shape))
    if ttype == GGMLQuantizationType.F32: return torch.from_numpy(tensor.data.view(np.float32).reshape(shape).copy()).to(torch.bfloat16)
    elif ttype == GGMLQuantizationType.F16: return torch.from_numpy(tensor.data.view(np.float16).reshape(shape).copy()).to(torch.bfloat16)
    elif ttype == GGMLQuantizationType.BF16:
        return torch.from_numpy(((tensor.data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)).reshape(shape).copy()).to(torch.bfloat16)
    elif ttype == GGMLQuantizationType.Q4_0:
        return torch.from_numpy(dequantize_q4_0_to_float(tensor.data, shape[0], shape[1]).copy()).to(torch.bfloat16)
    else:
        return torch.from_numpy(quants.dequantize(tensor.data, ttype).reshape(shape).copy()).to(torch.bfloat16)

def repack_lfm25_gguf(gguf_p: str, out_dir: str, moe_group_size: int = 128, quantize_embeddings: bool = False, upload_repo: str = None, token: str = None):
    from gguf import GGUFReader, GGMLQuantizationType
    writer = ShardWriter(out_dir, upload_repo=upload_repo, token=token)
    reader = GGUFReader(gguf_p, mode="r")
    for t in reader.tensors:
        n = t.name
        if n == "token_embd.weight":
            sh = tuple(int(x) for x in reversed(t.shape))
            if quantize_embeddings and t.tensor_type == GGMLQuantizationType.Q4_0:
                wp, ws, sh_t = unpack_gguf_q4_0_to_ct_int32(t.data, sh[0], sh[1], group_size=32)
                writer.add("model.embed_tokens.weight_packed", wp)
                writer.add("model.embed_tokens.weight_scale", ws)
                writer.add("model.embed_tokens.weight_shape", sh_t)
            else: writer.add("model.embed_tokens.weight", dequant_tensor_to_bf16(t))
            continue
        if n in MAP_DIRECT:
            writer.add(MAP_DIRECT[n], dequant_tensor_to_bf16(t))
            continue
        m = re.match(r"^blk\.(\d+)\.(.+)$", n)
        if not m: continue
        layer_i, suffix = int(m.group(1)), m.group(2)
        prefix = f"model.layers.{layer_i}"
        if suffix in MAP_BF16:
            arr_t = dequant_tensor_to_bf16(t)
            if suffix == "shortconv.conv.weight" and arr_t.ndim == 2: arr_t = arr_t.unsqueeze(1)
            writer.add(f"{prefix}.{MAP_BF16[suffix]}", arr_t)
        elif suffix in MAP_W8A8:
            q, sc = quantize_bf16_to_w8a8_channel(dequant_tensor_to_bf16(t))
            base = f"{prefix}.{MAP_W8A8[suffix]}"
            writer.add(f"{base}.weight", q); writer.add(f"{base}.weight_scale", sc)
        elif suffix in MAP_EXPS:
            proj, c_shape = MAP_EXPS[suffix], tuple(int(x) for x in reversed(t.shape))
            num_exp, out_f, in_f = c_shape
            b_per_exp = (out_f * in_f // 32) * 18
            for e in range(num_exp):
                exp_bytes = t.data[e] if t.data.ndim == 3 else t.data.reshape(-1)[e * b_per_exp : (e + 1) * b_per_exp]
                wp, ws, sh_t = unpack_gguf_q4_0_to_ct_int32(exp_bytes, out_f, in_f, group_size=moe_group_size)
                eb = f"{prefix}.feed_forward.experts.{e}.{proj}"
                writer.add(f"{eb}.weight_packed", wp); writer.add(f"{eb}.weight_scale", ws); writer.add(f"{eb}.weight_shape", sh_t)
    writer.close()
    cfg_path = os.path.join(out_dir, "config.json")
    save_config(build_lfm2_config(group_size=moe_group_size, quantize_embeddings=quantize_embeddings), out_dir)
    if writer.api and upload_repo:
        print(f"[REPACK] Uploading config.json to {upload_repo}...", flush=True)
        writer.api.upload_file(path_or_fileobj=cfg_path, path_in_repo="config.json", repo_id=upload_repo, token=token)
