# ref: https://github.com/facebookresearch/multimodal_rewardbench/blob/main/scripts/1_run_model_as_judge_gpt4o.py
# https://platform.openai.com/docs/guides/text?api-mode=responses&lang=python


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
            content = [{"type": "input_text", "text": ex["Text"]}]
            image_path = ex.get("Image", None)
            if image_path:
                image_url = self._local_image_to_data_url(image_path)
                content.append({"type": "input_image", "image_url": image_url})
            messages[0]["content"].extend(content)

        return messages

    def _call_client(self, messages, model, sampling_params):
        if not sampling_params:
            sampling_params = {"temperature": 0.0, "top_p": 1.0}

        response = self.client.responses.create(
            model=model,  # "gpt-4o-2024-08-06",  # "gpt-4o",
            input=messages,
            **sampling_params,
        )
        out = response.output_text
        return out

    def _call_client_wrapper(self, messages, model, max_try, sampling_params):
        """Wrapper to call the client with retry logic."""
        count = 1
        while count < max_try:
            try:
                out = self._call_client(messages, model, sampling_params)
                return out
            except Exception as e:
                print("Exception:", e)
                count += 1
                time.sleep(2)
        return "None"

    def generate(
        self,
        example: list[dict] | list[list[dict]],
        max_try: int = 4,
        sampling_params=None,
    ):
        """example = [{"Text": "", "Image": "Optional"}, ...]"""
        messages = self._get_messages(example)
        return self._call_client_wrapper(
            messages=messages,
            model=self.model,
            max_try=max_try,
            sampling_params=sampling_params,
        )


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
