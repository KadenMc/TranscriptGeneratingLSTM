import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

import numpy as np

import sys
import io
import argparse
import yaml


def load_config(f):
    with open(f) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def demo(args):
    # Process arguments
    print("Processing arguments, data, and model...")
    Xy = np.load(args.xy_file)
    max_length = Xy[Xy.files[0]].shape[1]

    chars = Xy[Xy.files[2]]
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    model = load_model(args.model)

    text = open(args.data, 'r').read()
    temperature = args.temperature

    demo_length = args.length
    
    while True:
        text_in = input("Input 'r' for a random prompt from the text, 't' to change the temperature, or anything else as a custom prompt:\n")

        # Create a prompt from a random position in the text
        if text_in == 'r':
            start_index = np.random.randint(0, high=len(text) - max_length - 1)
            prompt = text[start_index: start_index + max_length]

        elif text_in == 't':
            temperature = float(input("Temperature: ").strip())
            continue
        
        # Use an inputted prompt
        else:
            # Crop the prompt to the relevant max_length
            if len(text_in) <= max_length:
                prompt = ' '*(max_length-len(text_in)) + text_in
            else:
                prompt = sentence[-max_length:]

        generated = ''
        print('Generating with prompt:\n' + prompt + '\n')
        sentence = prompt
        for i in range(demo_length):
            x_pred = np.zeros((1, max_length, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print("\n"*2)


def main():
    parser = argparse.ArgumentParser()
    config = load_config("config.yaml")

    # Optional arguments
    parser.add_argument('-d', '--data', type=str, help='Path to the text_file.txt file (otherwise use text_file in config.yaml)',
                        default=config["train"]["text_file"])
    parser.add_argument('-m', '--model', type=str, help='Path to the model.h5 file (otherwise use model_file in config.yaml)',
                        default=config["train"]["model_file"])
    parser.add_argument('-x', '--xy_file', type=str, help='Path to data.npz file to get characters (otherwise use Xy_file in config.yaml)',
                        default=config["train"]["Xy_file"])
    parser.add_argument('-t', '--temperature', help='The temperature, or diversity (otherwise use temperature in config.yaml)',
                        default=config["demo"]["temperature"])
    parser.add_argument('-l', '--length', help='The length of the text generated', default=300)
    
    args = parser.parse_args()

    # Run the demo
    demo(args)

if __name__ == "__main__":
    main()
