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
from judge_model.dataset.multimodel_rewardbench import prepare_judge_example


def get_multi_modal_input(args):
    """
    return {
        "data": a list of images or videos,
        "question": a list of questions,
    }
    """
    if args["modality"] == "image":
        if args["dataset"].endswith("jsonl"):
            dataset = [
                json.loads(line)
                for line in open(args["dataset"], "r", encoding="utf-8")
            ]
        elif args["dataset"].endswith("json"):
            dataset = json.load(open(args["dataset"], "r", encoding="utf-8"))
        else:
            from datasets import load_dataset

            dataset = load_dataset(args["dataset"], split=args["dataset_split"])
            dataset = dataset.to_list()

        # if args.img_desc:
        #     img_desc_map = {}
        #     with open(args.img_desc_path, "r", encoding="utf-8") as f:
        #         for line in f:
        #             ex = json.loads(line)
        #             img_desc_map[ex["ID"]] = ex["prompt_output"]

        #     dataset = dataset.filter(
        #         lambda x: x["ID"] in img_desc_map.keys(),
        #         num_proc=4,
        #         load_from_cache_file=True,
        #         desc="Filtering dataset",
        #     )
        #     dataset = dataset.map(
        #         lambda x: {"prompt_output": img_desc_map[x["ID"]]},
        #         num_proc=4,
        #         load_from_cache_file=True,
        #         # remove_columns=["Image_desc"],
        #         desc="Adding image description to text",
        #     )
        #     dataset = dataset.map(
        #         prepare_judge_example_img_desc,
        #         num_proc=4,
        #         load_from_cache_file=True,
        #         fn_kwargs={"swap": args.swap, "prompt_key": args.prompt_key},
        #         remove_columns=["Category", "Output1", "Output2", "Better"],
        #         desc="Preparing judge examples",
        #     )
        # else:
        #     dataset = dataset.map(
        #         prepare_judge_example,
        #         num_proc=4,
        #         load_from_cache_file=True,
        #         fn_kwargs={"swap": args.swap},
        #         remove_columns=["Category", "Output1", "Output2", "Better"],
        #         desc="Preparing judge examples",
        #     )

        # from transformers import AutoProcessor

        # if args.model_type == "molmo":
        #     processor = AutoProcessor.from_pretrained(
        #         "allenai/Molmo-7B-D-0924",
        #         trust_remote_code=True,
        #         torch_dtype="auto",
        #         device_map="auto",
        #         use_fast=True,
        #     )
        #     dataset = [
        #         truncate_long_seq(processor, example, args.max_seq_len)
        #         for example in dataset
        #     ]

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

    if args["modality"] == "video":
        pass

    msg = f"Modality {args['modality']} is not supported."
    raise ValueError(msg)


def get_output_filepath(args):
    filename = f"{args['model_type']}"
    # if args["img_desc"]:
    #     filename += "-img-desc"
    if args["swap"]:
        filename += "-swap"
    output_filepath = os.path.join(args["output_dir"], f"{filename}.jsonl")
    return output_filepath


def main(args):
    model = args["model_type"]
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")
    model_hf = model_hf_map.get(model, model)

    output_filepath = get_output_filepath(args)
    print(f"Output file: {output_filepath}")

    modality = args["modality"]
    mm_input, dataset = get_multi_modal_input(args)
    data = mm_input["data"]
    questions = mm_input["questions"]

    req_data = model_example_map[model](questions, modality, args)

    engine_args = asdict(req_data.engine_args) | {"seed": args["seed"]}
    engine_args["dtype"] = args["dtype"]
    if "tensor_parallel_size" in args:
        engine_args["tensor_parallel_size"] = args["tensor_parallel_size"]
    llm = LLM(**engine_args)

    # To maintain code compatibility in this script, we add LoRA here.
    # You can also add LoRA using:
    # llm.generate(prompts, lora_request=lora_request,...)
    if req_data.lora_requests:
        for lora_request in req_data.lora_requests:
            llm.llm_engine.add_lora(lora_request=lora_request)

    # Don't want to check the flag multiple times, so just hijack `prompts`.
    prompts = req_data.prompts

    sampling_params = SamplingParams(
        temperature=args.get("temperature", 0.0),
        max_tokens=args.get("max_seq_len") or engine_args.get("max_model_len", 200),
        stop_token_ids=req_data.stop_token_ids,
    )

    assert args["num_prompts"] > 0
    assert len(prompts) == len(data)
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
                    ex["swap"] = args["swap"]
                    ex.pop("Image")
                    f.write(json.dumps(ex) + "\n")
                except AssertionError:
                    print(f"output: {o}")
                    print(f"ID: {ex['ID']}")
                    assertion_err_list.append(start_idx + idx)
                    continue

    print(f"Assertion error list: {assertion_err_list}")


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
