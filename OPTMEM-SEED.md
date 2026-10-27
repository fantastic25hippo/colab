fantastic25hippo/colab (public) holds Sahil's notes for running open-source models on Google Colab GPUs and TPUs; work on the main branch only.
Goal: Sahil downloads a model, asks Grok to bench it, and it benches on Colab with no trial and error; the fastest single-stream decode for coding work is the current focus.
Colab is billed against a Google AI Pro compute-unit (CU) pool; ops.md says 200 CU a month, but `colab usage` read 400.00 CU on 2026-10-27, so check `colab usage` before and after every run.
This repo is public: never put keys, tokens, the OAuth token file, memory contents or any Google account email in it.
Layout: README.md (focus and index), ops.md (Colab CLI, Drive, tunnel, heartbeat, quota), hardware.md (the 7 accelerators), research/ (model speed and precision notes), benchmarks/ (measured results).
Read benchmarks/README.md before benching anything: it fixes the entry format, the 10-step runbook and the rules. Results go in benchmarks/<Model>/<GPU>/RESULTS.md plus one row in benchmarks/INDEX.md.
Research files in research/2026_model_research/ are estimates, not measurements; where notes disagree prefer the newer date. Only numbers marked MEASURED in benchmarks/ came from a real run.
The Colab CLI is `colab` (google-colab-cli); put the global `--auth=oauth2` flag before the subcommand. `exec` times out after 30 s by default, so pass a longer `--timeout`.
`colab drivemount` needs the human's browser approval on every new VM, so a bot must stop and ask Sahil. Bots never log in or type Google codes.
Weights live in Sahil's Google Drive at /content/drive/MyDrive/models/<name>.gguf/<name>.gguf; copy them to local /tmp first, never run a model from the Drive mount, and never re-download from Hugging Face when Drive has them.
Always stop the Colab runtime when done and check that `colab --auth=oauth2 sessions` is empty; an idle runtime keeps burning CU.
Shared private OptMem store: fantastic25hippo/optmem-memory, synced by grok-agy-memo; nothing from it is stored in this repo. See docs/OPTMEM.md.
Keep every new file under 150 lines and every commit email on the GitHub noreply form.
