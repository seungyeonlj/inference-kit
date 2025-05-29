import json
import os
import time
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser
from dataclasses import asdict

from src.models.api_models import model_example_map
from src.dataset.preprocessing import prepare_dataset


def get_multi_modal_input(modality, args):
    """
    return {
        "data": a list of paths for images or videos,
        "question": a list of questions,
    }
    """
    if modality == "image":
        dataset = prepare_dataset(args)

        if args.get("load_local_image", False):
            for ex in dataset:
                ex["Image"] = os.path.join(args["image_folder"], ex["Image"])
        else:
            for ex in dataset:
                tmp_filepath = os.path.join(args["image_folder"], f"{ex['ID']}.jpg")
                if not os.path.exists(tmp_filepath):
                    ex["Image"].convert("RGB").save(
                        os.path.join(args["image_folder"], f"{ex['ID']}.jpg")
                    )
                ex["Image"] = tmp_filepath

        return dataset

    if modality == "video":
        pass

    msg = f"Modality {modality} is not supported."
    raise ValueError(msg)


def get_output_filepath(args, dataset_name, swap=False):
    filename = f"{args['model_type']}-{dataset_name}"
    if swap:
        filename += "-swap"
    output_filepath = os.path.join(args["output_dir"], f"{filename}.jsonl")
    return output_filepath


# TODO: truncate left if the text exceeds the model max length
def truncate_text(text, model):
    pass


def prompt_wrapper(args):
    ex, max_try, sampling_params = args
    return llm.generate(ex, sampling_params=sampling_params)


def main(llm, args):
    modality = args["modality"]

    sampling_params = {
        "temperature": args.get("temperature", 0.0),
        "top_p": args.get("temperature", 1.0),
        # args.get("max_seq_len", 1024)
    }

    for _dataset in args["datasets"]:
        output_filepath = get_output_filepath(
            args, _dataset["dataset_name"], _dataset["swap"]
        )
        print(f"Output file: {output_filepath}")

        dataset = get_multi_modal_input(modality, _dataset)

        inputs = dataset[: min(args["num_prompts"], len(dataset))]
        inputs = [[ex] for ex in inputs]

        chunk_size = args["data_chunk_size"] if args["data_chunk_size"] else len(inputs)
        total = inputs.copy()
        len_total = len(total)
        print(f"Total: {len_total}")
        print(f"Example: {total[0]}")
        for i in range(args["start_idx"], len_total, chunk_size):
            start_idx = i
            end_idx = min(i + chunk_size, len_total)
            print(
                f"Processing from {start_idx} idx to {end_idx - 1} idx. ID=={dataset[start_idx]['ID']} to ID=={dataset[end_idx - 1]['ID']}"
            )
            inputs = total[start_idx:end_idx]

            print(f"Started generation")
            start_time = time.time()
            with Pool(args.get("num_process", 1)) as p:
                outputs = list(
                    tqdm(
                        p.imap(
                            prompt_wrapper,
                            [
                                (
                                    ex,
                                    args.get("max_try", 4),
                                    sampling_params,
                                )
                                for ex in inputs
                            ],
                        ),
                        total=len(inputs),
                        desc="Running model",
                    )
                )
            elapsed_time = time.time() - start_time
            print("-- generate time = {}".format(elapsed_time))
            assert len(outputs) == len(
                inputs
            ), f"Output length mismatch: {len(outputs)} != {len(inputs)}"

            assertion_err_list = []
            with open(output_filepath, "a", encoding="utf-8") as f:
                for idx, (out, ex) in enumerate(
                    zip(outputs, dataset[start_idx:end_idx])
                ):
                    try:
                        generated_text = out
                        ex["output"] = generated_text
                        ex["model"] = llm.get_model_name()
                        ex["swap"] = _dataset["swap"]
                        ex.pop("Image")
                        f.write(json.dumps(ex) + "\n")
                    except AssertionError:
                        print(f"output: {out}")
                        print(f"ID: {ex['ID']}")
                        assertion_err_list.append(start_idx + idx)
                        continue

        print(f"{_dataset['dataset_name']} Assertion error list: {assertion_err_list}")


if __name__ == "__main__":
    import yaml

    parser = ArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for text generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gpt-41.yaml",
        help="Path for YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)

    # get llm
    model = configs["model_type"]
    if (model_prefix := model.split("-", 1)[0]) not in model_example_map:
        raise ValueError(f"Model tkaype {model} is not supported.")
    model_class = model_example_map[model_prefix]
    llm = model_class(configs["model_type"])

    # inference
    main(llm, configs)
