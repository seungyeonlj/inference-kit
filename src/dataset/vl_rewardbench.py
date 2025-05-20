import numpy as np
import random
import re

random.seed(123)


def extract_judgment(ex):
    response = ex["output"]

    pattern = r"(?:Overall Judgment|Therefore)\s*.*\s*-*\s*Answer\s*(\d+)\s*is\s*(?:the\s*)?(?:slightly\s*)?better"
    match = re.search(
        pattern, response.replace("\n", "").replace("*", ""), re.IGNORECASE
    )

    if match:
        answer_number = int(match.group(1))
        return answer_number
    return "doesntMatch"


def prompt(data_obj, random_number):
    if random_number == 0:
        answer1, answer2 = data_obj["Output1"], data_obj["Output2"]
    else:
        answer1, answer2 = data_obj["Output2"], data_obj["Output1"]

    prompt_str = f""" You are a highly capable multimodal AI assistant tasked with evaluating answers to visual questions. Please analyze the following image and question, then determine which of the two provided answers is better.

Question: {data_obj["Text"]}

Answer 1: {answer1}

Answer 2: {answer2}

Please evaluate both answers based on the following criteria:
1. Accuracy: How well does the answer align with the visual information in the image?
2. Completeness: Does the answer fully address all aspects of the question?
3. Clarity: Is the answer easy to understand and well-articulated?
4. Relevance: Does the answer directly relate to the question and the image?

After your evaluation, please:
1. Explain your reasoning for each criterion.
2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2). For example: Overall Judgment: Answer X is better.

Your response should be structured and detailed, demonstrating your understanding of both the visual and textual elements of the task."""
    return prompt_str


def prepare_judge_example(item: dict, swap=False):
    # random_number = random.choice([0, 1])
    # question = prompt(item, random_number)
    number = 1 if swap else 0
    question = prompt(item, number)
    question = question.replace("<image>\n", "")

    label = item["Label"]
    if swap:
        label = 2 if label == 1 else 1

    result = {
        "ID": item["ID"],
        "Text": question,
        "Image": item["Image"],
        "Label": label,
        "Meta": {
            "Text": item["Text"],
            "Output1": item["Output1"],
            "Output2": item["Output2"],
            "Better": item["Better"],
            "random_numer": number,
            "human_ranking": item["human_ranking"],
            "models": item["models"],
            "judge": item["judge"],
            "rationale": item["rationale"],
            "query_source": item["query_source"],
        },
        "swap": swap,
    }

    return result


def unify_key(item: dict, load_local_image=False):
    label = (
        1 if item["human_ranking"][0] == 0 else 2
    )  # human_ranking: rank of the two responses [0, 1] denotes the first one is preferred; [1, 0] denotes the second one is better;

    result = {
        "ID": item["id"],
        "Text": item["query"],
        "Image": f"{item['id']}.jpg" if load_local_image else item["image"],
        "Label": label,
        "Better": f"Output{label}",
        "Output1": item["response"][0],
        "Output2": item["response"][1],
        "human_ranking": item["human_ranking"],
        "models": item["models"],
        "judge": item["judge"],
        "rationale": item["rationale"],
        "query_source": item["query_source"],
    }

    return result


def process_vl_rewardbench(dataset, load_local_image=False):
    processed_dataset = [unify_key(ex, load_local_image) for ex in dataset]
    processed_dataset = [prepare_judge_example(ex) for ex in processed_dataset]
    return processed_dataset
