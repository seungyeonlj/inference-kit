export PYTHONPATH=$(pwd):$PYTHONPATH
# python src/run_vllm_vision_language.py --config configs/molmo.yaml
python src/run_vllm_vision_language.py --config configs/llava-onevision.yaml
# python src/run_vllm_vision_language.py --config configs/qwen2_5_vl.yaml