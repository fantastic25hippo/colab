# GGUF Model Inventory

Deposited Unsloth Dynamic 4-bit GGUF models (`UD-Q4_K_XL`) in llama.cpp format.

## Deposited Models

### 1. LFM2.5-24B-A2B

- **Destination Repository**: `avgrocketman/LFM2.5-24B-A2B-GGUF` (private)
- **File**: `LFM2.5-24B-A2B-UD-Q4_K_XL.gguf`
- **Size**: 14,876,901,632 bytes (~14.9 GB)
- **SHA256**: `a235d23638cc5549cb1ff61da7522c8a7e97f056f27093462a1e56f35aba91d8`
- **Source**: `unsloth/LFM2.5-24B-A2B-GGUF`
- **Deposit Method**: Colab CPU `hf_hub_download` + `upload_file`
- **Performance**: ~98.3 MB/s download, ~92.5 MB/s upload
- **Resource Usage**: 0 CU consumed; session `run-6a5375` auto-stopped upon completion.

### 2. Flint

- **Destination Repository**: `avgrocketman/flint-GGUF` (private)
- **File**: `flint-UD-Q4_K_XL.gguf`
- **Size**: 25,054,688,704 bytes (~25.1 GB)
- **SHA256**: `63c2ecc461ca545134b670c660e76fa20f47240f8d39b6510bfd21c4dd6e072f`
- **Source**: `unsloth/flint-GGUF`
- **Deposit Method**: Local `huggingface_hub.copy_files` (Hub-to-Hub server-side copy)
- **Performance**: Completed in 3.24 seconds; 0 bytes transferred through local machine
- **Resource Usage**: 0 CU consumed.

## Future Ingestion Protocol

- **Hub-to-Hub**: Use `copy_files` or `hf cp` whenever the target weights already reside on Hugging Face (server-side copy within the same cloud storage region).
- **External Sources**: Only use download + upload streaming if the upstream model is not hosted on Hugging Face Hub.
