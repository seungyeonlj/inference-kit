import json

from src.dataset.vl_rewardbench import process_vl_rewardbench
from src.dataset.multimodal_rewardbench import process_multimodal_rewardbench

dataset_func_map = {
    "multimodal_rewardbench": process_multimodal_rewardbench,
    "vl_rewardbench": process_vl_rewardbench,
}
dataset_hf_map = {
    "vl_rewardbench": "MMInstruction/VL-RewardBench",
}


def prepare_dataset(args):
    """
    Prepare the dataset for the model.
    """
    dataset_name = args["dataset_name"]
    process_func = dataset_func_map.get(dataset_name)
    if process_func is None:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    if dataset_name in dataset_hf_map:
        from datasets import load_dataset

        dataset = load_dataset(
            dataset_hf_map[dataset_name], split=args["dataset_split"]
        )
    else:
        assert "dataset_path" in args, "dataset_path is required for local dataset"
        dataset_path = args["dataset_path"]  # required for local dataset
        if dataset_path.endswith("jsonl"):
            dataset = [
                json.loads(line) for line in open(dataset_path, "r", encoding="utf-8")
            ]
        elif dataset_path.endswith("json"):
            dataset = json.load(open(dataset_path, "r", encoding="utf-8"))

    dataset = process_func(dataset, args.get("load_local_image", False))
    return dataset
