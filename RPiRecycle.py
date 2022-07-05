# RPIRecycle.py
# 2021-04-09
# This program handles all of the raspberry pi functions: getting input from GPIO pin,
# taking the picture, loading the tensorflow model, using that model to get a prediction of the image,
# and then uploading the predicion to the webserver
# References:
# Microsoft's lobe.ai model file comes with an example code, several functions were taken
# from said code and edited to fit the needs of the project

import RPi.GPIO as GPIO
import argparse
import json
import os
import requests
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from picamera import PiCamera
from time import sleep
camera = PiCamera()

def takePhoto():
    # displays cameras view on monitor for 2 seconds
    camera.start_preview()
    sleep(2)
    # saves image temporarily
    camera.capture('/tmp/picture.jpg')
    camera.stop_preview()
    return Image.open('/tmp/picture.jpg')

def upload(itemtag):
    url = 'http://pices.ece.queensu.ca/incrementItem.php'
    myobj = {'itemid': itemtag}
    # creates post requset
    x = requests.post(url, data = myobj)
    print(x.text)

def get_model_and_sig(model_dir):
    with open(os.path.join(model_dir, model_dir+"/signature.json"), "r") as f:
        signature = json.load(f)
    model_file =model_dir+"/"+ signature.get("filename")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model file does not exist"+" - "+model_file)
    return model_file, signature


def load_model(model_file):
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    return interpreter


def get_prediction(image, interpreter, signature):
    # Combine the information about the inputs and outputs from the signature.json file with the Interpreter runtime
    signature_inputs = signature.get("inputs")
    input_details = {detail.get("name"): detail for detail in interpreter.get_input_details()}
    model_inputs = {key: {**sig, **input_details.get(sig.get("name"))} for key, sig in signature_inputs.items()}
    signature_outputs = signature.get("outputs")
    output_details = {detail.get("name"): detail for detail in interpreter.get_output_details()}
    model_outputs = {key: {**sig, **output_details.get(sig.get("name"))} for key, sig in signature_outputs.items()}

    if "Image" not in model_inputs:
        raise ValueError("Tensorflow Lite model doesn't have 'Image' input! Check signature.json, and please report issue to Lobe.")

    # process image to be compatible with the model
    input_data = process_image(image, model_inputs.get("Image").get("shape"))

    # set the input to run
    interpreter.set_tensor(model_inputs.get("Image").get("index"), input_data)
    interpreter.invoke()

    # grab our desired outputs from the interpreter!
    # un-batch since we ran an image with batch size of 1, and convert to normal python types with tolist()
    outputs = {key: interpreter.get_tensor(value.get("index")).tolist()[0] for key, value in model_outputs.items()}
    # postprocessing! convert any byte strings to normal strings with .decode()
    for key, val in outputs.items():
        if isinstance(val, bytes):
            outputs[key] = val.decode()

    return outputs


def process_image(image, input_shape):
    width, height = image.size
    # convert jpg to RGB
    image = image.convert("RGB")
    # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
    if width != height:
        square_size = min(width, height)
        left = (width - square_size) / 2
        top = (height - square_size) / 2
        right = (width + square_size) / 2
        bottom = (height + square_size) / 2
        # Crop the center of the image
        image = image.crop((left, top, right, bottom))
    # now the image is square, resize it to be the right shape for the model input
    input_width, input_height = input_shape[1:3]
    if image.width != input_width or image.height != input_height:
        image = image.resize((input_width, input_height))

    # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
    image = np.asarray(image) / 255.0
    # format input as model expects
    return image.reshape(input_shape).astype(np.float32)

def classifyImage(image, model_dir):
    # gets model
    model_file, signature = get_model_and_sig(model_dir)
    # gets interpreter
    interpreter = load_model(model_file)
    # makes prediction
    prediction = get_prediction(image, interpreter, signature)
    # get list of confidences from prediction
    confidences = list(prediction.values())[0]
    # get the label names for the predicted class
    labels = signature.get("classes").get("Label")
    # gets the highest confidence (closest to 1)
    max_confidence = max(confidences)
    # gets label associated with highest confidence
    prediction["Prediction"] = labels[confidences.index(max_confidence)]
    print(prediction)
    print('The label is:', labels[confidences.index(max_confidence)])
    # returns label with the highest confidence
    return labels[confidences.index(max_confidence)]

def buttonCallback(channel):
    #indicates button is pressed
    print('Button Pressed')
    #takes photo
    image = takePhoto()
    # Assume model is in the parent directory for this file
    model_dir = os.getcwd()+"/CRIPModel"
    # gets an integer associated with an object detected
    label = classifyImage(image, model_dir)
    # checks if object was detected
    if label != '0':
        # uploads item id
        upload(label)
    else:
        print('Item is not recognized')

if __name__ == "__main__":
    #Ignores warning 
    GPIO.setwarnings(False)
    #Use physical pin numbering
    GPIO.setmode(GPIO.BOARD)
    #Set pin 10 to be an input pin and set initial value to be pulled low (off)
    GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    #Setup event on pin 10 rising edge 
    GPIO.add_event_detect(10,GPIO.RISING,callback=buttonCallback) 
    #Run until someone presses enter
    message = input("Press enter to quit\n\n")
    #Clean up
    GPIO.cleanup()
