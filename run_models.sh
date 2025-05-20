export PYTHONPATH=$(pwd):$PYTHONPATH
python src/models/vision_language.py --config configs/molmo.yaml
python src/models/vision_language.py --config configs/llava-onevision.yaml
python src/models/vision_language.py --config configs/qwen2_5_vl.yaml