# Copyright: Meta Platforms, Inc. and affiliates
# A slightly modified version of the original code from https://github.com/facebookresearch/multimodal_rewardbench/blob/main/scripts/1_run_model_as_judge_gpt4o.py

import json
import argparse
import os
import time
from tqdm import tqdm

from multiprocessing import Pool


from dotenv import load_dotenv


# Load environment variables from a .env file
load_dotenv()


class Model:
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwds):
        return self.generate(*args, **kwds)

    def __repr__(self):
        return f"Model({self.model})"

    def generate(self, example):
        # Implement the VLM model prediction logic here
        pass

    def get_model_name(self):
        return self.model


class VisionLanguageModel(Model):
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

    def _local_image_to_data_url(self, image_path):
        import base64
        from mimetypes import guess_type

        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"  # Default MIME type if none is found
        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"


class GPTModel(VisionLanguageModel):
    def __init__(self, model):
        super().__init__(model)

        from openai import OpenAI
        import os

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _get_messages(self, example: list[dict]):
        # Currently, support single turn only. (no assistant message)
        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]
        for ex in example:
            content = [{"type": "text", "text": ex["Text"]}]
            image_path = ex.get("Image", None)
            if image_path:
                image_url = self._local_image_to_data_url(image_path)
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            messages[0]["content"].extend(content)

        return messages

    def _call_client(self, messages, model):
        completion = self.client.chat.completions.create(
            model=model,  # "gpt-4o-2024-08-06",  # "gpt-4o",
            messages=messages,
            max_tokens=1024,
            temperature=1,  # default
            top_p=1,  # default
            frequency_penalty=0,  # default
            presence_penalty=0,  # default
        )
        out = completion.to_dict()["choices"][0]["message"]["content"].strip()
        return out

    def _call_client_wrapper(self, messages, model, max_try=4):
        """Wrapper to call the client with retry logic."""
        count = 1
        while count < max_try:
            try:
                out = self._call_client(messages, model)
                return out
            except Exception as e:
                print("Exception:", e)
                count += 1
                time.sleep(2)
        return "None"

    def generate(self, example: list[dict]):
        """input = [{"Text": "", "Image": "Optional"}, ...]"""
        messages = self._get_messages(example)
        return self._call_client_wrapper(messages, self.model)

    def generate_data(self, examples: list[list[dict]], num_process: int = 1):
        """
        input = [{"Text": "", "Image": "Optional"}, ...]
        examples =[input1, input2, ...]
        """
        with Pool(num_process) as p:
            outputs = list(
                tqdm(
                    p.imap(self.generate, examples),
                    total=len(examples),
                    desc=f"Running model with {num_process} processes",
                )
            )
        assert len(outputs) == len(examples)
        return outputs


class GPTToolModel(Model):
    # TODO: implement the tool model
    def __init__(self, model):
        super().__init__(model)

    def generate(self, example: list[dict]):
        pass


model_example_map = {
    "gpt": GPTModel,
    "o4": GPTModel,
    "o3": GPTModel,
    "o1": GPTModel,
}
