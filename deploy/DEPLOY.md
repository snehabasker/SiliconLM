# Deploying to a free CPU HF Space

## 1. Quantize to GGUF (do this once, on Kaggle CPU or locally)
    pip install llama-cpp-python huggingface_hub
    git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
    pip install -r requirements.txt
    # merge the LoRA adapter into the base first (peft merge_and_unload), then:
    python convert_hf_to_gguf.py /path/to/merged --outfile siliconlm-f16.gguf
    ./llama-quantize siliconlm-f16.gguf siliconlm-q4_k_m.gguf Q4_K_M
    huggingface-cli upload you/siliconlm-gguf siliconlm-q4_k_m.gguf

Worth running silicon_eval on both f16 and Q4_K_M and putting both rows in the ablation table, so you can see what quantization actually cost you.

## 2. Create the Space
- New Space -> Gradio -> CPU basic (free)
- Upload the repo (`app_file: deploy/app.py` in the Space README metadata), or copy `deploy/*` to the Space root alongside `silicon_eval/` and `rag/`
- `packages.txt` installs iverilog on the Space (already included)
- Set the Space variable `MODEL_PATH=/data/siliconlm-q4_k_m.gguf` after downloading the GGUF in a startup step, or leave it unset — the app runs in demo mode (Verify + Ablation + Ask docs all work without a model)

## 3. Notes
- 1.5B @ Q4_K_M is about 1.1GB RAM, ~5-10 tok/s on the free 2-vCPU tier — fine for a demo.
- The Verify tab doesn't need a model at all, so the Space is interactive from day one, before training even finishes.
