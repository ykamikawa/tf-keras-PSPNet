# -*- coding: utf-8 -*-
import argparse
import cv2
import json
from keras.preprocessing.image import img_to_array
from utils import createResponse
from PSPNet import PSPNet50


def inference(model, inputs):
    orig_w, orig_h = inputs.shape[0:2]
    inputs = cv2.resize(inputs, (512, 512))
    inputs = img_to_array(inputs)/255
    inputs = inputs.reshape((1, 512, 512, 3))
    outputs = model.predict(inputs)
    outputs = outputs.reshape(512, 512, 20)
    outputs = outputs.argmax(axis=2)

    return outputs


if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="predicter")
    parser.add_argument(
            "--img_path",
            help="target img path")
    parser.add_argument(
            "--weight",
            help="pretrained weight path")
    parser.add_argument(
            "--output_path",
            default="./response.json",
            help="output path")
    args = parser.parse_args()

    # load model
    model = PSPNet50()
    model.load_weights(args.weight)

    # inference
    inputs = cv2.imread(args.img_path)
    outputs = inference(model, inputs)

    # create response
    response = createResponse(outputs, inputs)

    # save response
    with open(args.output_path, "w") as f:
        json.dump(response, f, indent=2)
