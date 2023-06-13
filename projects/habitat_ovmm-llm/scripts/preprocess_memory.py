"""A simple script converting raw memory file into trainable data for llama adapter"""

import json

import click
import clip
import torch


@click.command()
@click.option("--infile", required=True)
@click.option("--outfile", default="datadump/processed_memory.json")
def process(infile, outfile):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)
    data = []

    with open(infile, "r") as json_file:
        raw_memory = json.load(json_file)
    full_object_history = []
    full_gt_history = None

    for timestep, context in raw_memory.items():
        features = context["clip_features"]
        objects = context["objects"]
        if not full_gt_history:
            full_gt_history = [False] * len(objects)

        frame_gt = context["is_found"]
        for i, gt in enumerate(frame_gt):
            if gt:
                full_gt_history[i] = True

        text_inputs = torch.cat([clip.tokenize(f"{c}") for c in objects]).to(device)
        # text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in objects]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        for image_features in features:
            image_features = (
                torch.FloatTensor(image_features).to(torch.float16).to(device)
            )
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            full_object_history.append(objects[torch.argmax(similarity).item()])

        for i, obj in enumerate(objects):
            datapoint = {"instruction": "", "input": "", "output": ""}
            datapoint[
                "instruction"
            ] = "What is my next action given the following context?"
            datapoint["input"] = "My task is to find " + obj + " and I have seen"

            for perceived_object in full_object_history:
                datapoint["input"] += " " + perceived_object
            datapoint["output"] = "Go to " + obj if full_gt_history[i] else "Explore"

            data.append(datapoint)

    with open(outfile, "w") as json_file:
        json.dump(data, json_file)


process()
