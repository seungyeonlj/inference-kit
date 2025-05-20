# https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py
# SPDX-License-Identifier: Apache-2.0


"""

This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.

"""


import json
import os
from PIL import Image
from argparse import ArgumentParser
from dataclasses import asdict

from vllm import LLM, SamplingParams

from judge_model.models.vision_language_models import (
    model_example_map,
    model_hf_map,
)
from judge_model.dataset.preprocessing import prepare_dataset


def get_multi_modal_input(modality, args):
    """
    return {
        "data": a list of images or videos,
        "question": a list of questions,
    }
    """
    if modality == "image":
        dataset = prepare_dataset(args)

        if args.get("load_local_image", False):
            for ex in dataset:
                ex["Image"] = Image.open(
                    os.path.join(args["image_folder"], ex["Image"])
                )

        return_dict = {
            "data": [ex["Image"].convert("RGB") for ex in dataset],
            "questions": [ex["Text"] for ex in dataset],
        }
        return return_dict, dataset

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


def main(args):
    model = args["model_type"]
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")
    model_hf = model_hf_map.get(model, model)
    modality = args["modality"]

    # engine_args are irrelevant for the input data.
    req_data = model_example_map[model](
        ["dummy question for engine_args"], modality, args
    )
    engine_args = asdict(req_data.engine_args) | {"seed": args["seed"]}
    engine_args["dtype"] = args["dtype"]
    if "tensor_parallel_size" in args:
        engine_args["tensor_parallel_size"] = args["tensor_parallel_size"]

    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=args.get("temperature", 0.0),
        max_tokens=args.get("max_seq_len") or engine_args.get("max_model_len", 200),
        stop_token_ids=req_data.stop_token_ids,
    )

    for _dataset in args["datasets"]:
        output_filepath = get_output_filepath(
            args, _dataset["dataset_name"], _dataset["swap"]
        )
        print(f"Output file: {output_filepath}")

        mm_input, dataset = get_multi_modal_input(modality, _dataset)
        data = mm_input["data"]
        questions = mm_input["questions"]

        req_data = model_example_map[model](questions, modality, args)

        # To maintain code compatibility in this script, we add LoRA here.
        # You can also add LoRA using:
        # llm.generate(prompts, lora_request=lora_request,...)
        if req_data.lora_requests:
            for lora_request in req_data.lora_requests:
                llm.llm_engine.add_lora(lora_request=lora_request)

        # Don't want to check the flag multiple times, so just hijack `prompts`.
        prompts = req_data.prompts
        assert len(prompts) == len(data)
        assert args["num_prompts"] > 0
        # Batch inference
        # Use the different image for each prompt
        inputs = [
            {
                "prompt": prompts[i],
                "multi_modal_data": {modality: data[i]},
            }
            for i in range(min(args["num_prompts"], len(data)))
        ]

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

            if args["time_generate"]:
                print(f"Started generation")
                import time

                start_time = time.time()
                outputs = llm.generate(inputs, sampling_params=sampling_params)
                elapsed_time = time.time() - start_time
                print("-- generate time = {}".format(elapsed_time))
            else:
                outputs = llm.generate(inputs, sampling_params=sampling_params)
            assert len(outputs) == end_idx - start_idx

            assertion_err_list = []
            with open(output_filepath, "a", encoding="utf-8") as f:
                for idx, (o, ex) in enumerate(zip(outputs, dataset[start_idx:end_idx])):
                    try:
                        assert ex["Text"] in o.prompt

                        generated_text = o.outputs[0].text
                        ex["output"] = generated_text
                        ex["model"] = model_hf
                        ex["swap"] = _dataset["swap"]
                        ex.pop("Image")
                        f.write(json.dumps(ex) + "\n")
                    except AssertionError:
                        print(f"output: {o}")
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
        help="Path for YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)

    main(configs)
