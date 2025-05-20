# Copyright: Meta Platforms, Inc. and affiliates
# Code from https://github.com/facebookresearch/multimodal_rewardbench

import random

random.seed(123)


judge_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"""


def extract_judgment(ex):
    judgment = ex["output"]
    if "[[A]]" in judgment:
        return "A"
    elif "[[B]]" in judgment:
        return "B"
    else:
        return "A" if random.random() < 0.5 else "B"


def prepare_judge_example(ex, swap=False):
    label = "B" if ex["Better"] == "Output2" else "A"
    if swap:
        label = "B" if label == "A" else "A"

    meta = {
        "Text": ex["Text"],
        "Output1": ex["Output1"],
        "Output2": ex["Output2"],
        "Better": ex["Better"],
        "Category": ex["Category"],
    }

    item = {
        "ID": ex["ID"],
        "Text": judge_prompt.format(
            question=ex["Text"],
            answer_a=(ex["Output2"] if swap else ex["Output1"]),
            answer_b=(ex["Output1"] if swap else ex["Output2"]),
        ),
        "Image": ex["Image"],
        "Label": label,
        "Meta": meta,
        "swap": swap,
    }
    return item


def process_multimodal_rewardbench(dataset, load_local_image=True):
    processed_dataset = [prepare_judge_example(ex) for ex in dataset]
    return processed_dataset
