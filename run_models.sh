export PYTHONPATH=$(pwd):$PYTHONPATH
# python judge_model/models/vision_language.py --config configs/molmo.yaml
# python judge_model/models/vision_language.py --config configs/llava-onevision.yaml
python judge_model/models/vision_language.py --config configs/qwen2_5_vl.yaml