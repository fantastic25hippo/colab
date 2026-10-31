import json, os, re

def build_lfm2_config(group_size: int = 128, quantize_embeddings: bool = False) -> dict:
    ignore_targets = [
        "re:.*feed_forward\\.gate.*$",
        "re:.*feed_forward\\.expert_bias.*$",
        "re:.*(short_conv|conv)\\.conv.*",
        "re:.*operator_norm.*",
        "re:.*ffn_norm.*",
        "re:.*embedding_norm.*",
        "re:.*q_layernorm.*",
        "re:.*k_layernorm.*",
        "lm_head",
    ]
    if not quantize_embeddings:
        ignore_targets.append("model.embed_tokens")

    moe_targets = [
        "Linear",
        "re:.*feed_forward\\.experts\\.(?:[0-9]+\\.)?(w1|w2|w3)$",
    ]
    if quantize_embeddings:
        moe_targets.append("model.embed_tokens")

    quant_cfg = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "ignore": ignore_targets,
        "config_groups": {
            "group_0_w8a8_int8_backbone": {
                "targets": [
                    "re:.*self_attn\\.(q_proj|k_proj|v_proj|qkv_proj|out_proj)$",
                    "re:.*(short_conv|conv)\\.(in_proj|out_proj)$",
                    "re:^model\\.layers\\.[01]\\.feed_forward\\.(w1|w2|w3|w13)$",
                ],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
                "format": "int-quantized",
            },
            "group_1_w4a16_moe_experts": {
                "targets": moe_targets,
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": group_size,
                    "dynamic": False,
                },
                "input_activations": None,
                "format": "pack-quantized",
            },
        },
    }

    layer_types = [
        "full_attention" if i in (2, 6, 10, 14, 18, 22, 26, 30, 34, 38) else "conv"
        for i in range(40)
    ]

    return {
        "architectures": ["Lfm2MoeForCausalLM"],
        "model_type": "lfm2_moe",
        "hidden_size": 2048,
        "intermediate_size": 11776,
        "moe_intermediate_size": 1536,
        "num_hidden_layers": 40,
        "num_dense_layers": 2,
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "conv_L_cache": 3,
        "conv_bias": False,
        "layer_types": layer_types,
        "norm_eps": 1e-05,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "use_expert_bias": True,
        "tie_word_embeddings": True,
        "vocab_size": 128000,
        "torch_dtype": "bfloat16",
        "dtype": "bfloat16",
        "bos_token_id": 124894,
        "eos_token_id": 124900,
        "pad_token_id": 124893,
        "max_position_embeddings": 200000,
        "rope_parameters": {
            "rope_theta": 10000000.0,
            "rope_type": "default",
        },
        "use_cache": True,
        "transformers_version": "5.12.1",
        "quantization_config": quant_cfg,
    }

def match_target(name: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return bool(re.search(pattern[3:], name))
    return name == pattern or name.endswith(f".{pattern}")

def match_layer_to_quant_group(layer_name: str, config: dict) -> str:
    q_cfg = config.get("quantization_config", {})
    for ign in q_cfg.get("ignore", []):
        if match_target(layer_name, ign):
            return "ignore"
    groups = q_cfg.get("config_groups", {})
    for g_name, g_val in groups.items():
        for t in g_val.get("targets", []):
            if t != "Linear" and match_target(layer_name, t):
                return g_name
    for g_name, g_val in groups.items():
        if "Linear" in g_val.get("targets", []):
            return g_name
    return "unmatched"

def save_config(config_dict: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
