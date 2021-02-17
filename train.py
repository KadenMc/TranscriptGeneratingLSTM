import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop

import numpy as np
from matplotlib import pyplot as plt

import yaml
import sys
import os
from shutil import copyfile

from model import Model


def load_config(f):
    with open(f) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def load_data(xy_file):
    Xy = np.load(xy_file)
    X = Xy[Xy.files[0]]
    y = Xy[Xy.files[1]]
    chars = Xy[Xy.files[2]]
    return X, y, chars


def load_text_data(s, max_length, xy_file=None, chars=None):
    if chars is None:
        chars = sorted(list(set(s)))
 
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # Cut text in semi-redundant sequences of max_length characters
    step = 3
    sequences = []
    next_chars = []
    for i in range(0, len(s) - max_length, step):
        sequences.append(s[i: i + max_length])
        next_chars.append(s[i + max_length])
    print('Number of sequences:', len(sequences))

    X = np.zeros((len(sequences), max_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    if xy_file is not None:
        np.savez(xy_file, X, y, np.array(chars))

    return X, y, chars


def get_model(nodes, dropout, recurrent_dropout, max_length, char_length, initial_lr, minimum_lr, decay_lr):
    m = Model(nodes, dropout, recurrent_dropout, max_length, char_length)
    model = m.get_model()
    optimizer = RMSprop(learning_rate=initial_lr, epsilon=minimum_lr, decay=decay_lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # On end of each epoch, print generated text.
    print()
    print('----- Generating text after Epoch: {}'.format(epoch))

    start_index = np.random.randint(0, high=len(text) - max_length - 1)
    generated = ''
    sentence = text[start_index: start_index + max_length]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

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
    print("\n")


def evaluate_history(history):
    s = "Training losses: " + str(history.history['loss'])
    s += "\nValidation losses: " + str(history.history['val_loss'])
    s += "\nTraining accuracies: " + str(history.history['accuracy'])
    s += "\nValidation accuracies: " + str(history.history['val_accuracy'])
    metrics_path = model_file[:-3] + "_metrics.txt"
    f = open(metrics_path, "w")
    f.write(s)
    f.close()

    labels = ["loss", "accuracy"]
    for i in range(2):
        plt.figure(i)
        plt.plot(history.history[labels[i])
        plt.plot(history.history['val_' + labels[i]])
        plt.ylabel(labels[i])
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig(model_file[:-3] + "_" + labels[i] + ".png")


def main():
    config = load_config("config.yaml")

    # Running on GPU
    if config["train"]["GPU"]:
        if tf.test.is_gpu_available():
            print("Running on GPUs.")
            print("Available GPUs:")
            print(tf.config.experimental.list_physical_devices('GPU'))
        else:
            print("No GPUs available. Troubleshoot or switch GPU to False in config.yaml")
    # Running on CPU
    else:
        print("Running on CPU.")
    print("\n")

    # Load filepaths and fix extensions
    global text_file, model_file
    text_file = config["train"]["text_file"]
    if text_file[-4:] != '.txt':
        text_file += '.txt'
    
    xy_file = config["train"]["Xy_file"]
    if xy_file[-4:] != '.npz':
        xy_file += '.npz'
    
    model_file = config["train"]["model_file"]
    if model_file[-3:] != '.h5':
        model_file += '.h5'

    # Load model hyperparameters
    lstm_nodes = config["model"]["lstm_nodes"]
    dropout = config["model"]["dropout"]
    recurrent_dropout = config["model"]["recurrent_dropout"]

    global max_length
    max_length = config["model"]["max_length"]

    # Load training hyperparameters
    epochs = config["train"]["epochs"]
    batch_size = config["train"]["batch_size"]    
    initial_lr = config["train"]["initial_lr"]
    minimum_lr = config["train"]["minimum_lr"]
    decay_lr = config["train"]["decay_lr"]
    validation_split = config["train"]["validation_split"]

    # Training callbacks
    epoch_end_callback = config["train"]["epoch_end_callback"]
    checkpoint_callback = config["train"]["checkpoint_callback"]
    early_stopping_callback = config["train"]["early_stopping_callback"]
    early_stopping_patience = config["train"]["early_stopping_patience"]

    # Load demo parameters for on_epoch_end
    global temperature, demo_length
    temperature = config["demo"]["temperature"]
    demo_length = config["demo"]["length"]

    # Load data, try from .npz files, or if they don't exist, load from text data

    # Load the data  since it's needed for on_epoch_end
    global text, chars
    text = open(text_file, 'r').read()
    
    if os.path.exists(xy_file):
        print("Loading data from {}...".format(xy_file))
        X, y, chars = load_data(xy_file)
    else:
        print("Loading data from {}...".format(text_file))
        X, y, chars = load_text_data(text, max_length, xy_file=xy_file)

    global char_indices, indices_char, char_length
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    char_length = len(chars)

    # Load model
    print("Creating and compiling model...")
    global model
    model = get_model(lstm_nodes, dropout, recurrent_dropout, max_length, chars, initial_lr, minimum_lr, decay_lr)

    # Initialize callbacks
    callbacks = []
    if epoch_end_callback:
        epoch_end = LambdaCallback(on_epoch_end=on_epoch_end)
        callbacks.append(epoch_end)
    
    if checkpoint_callback:
        checkpoint_path = model_file[:-3] + "_checkpoint.h5"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        callbacks.append(checkpoint)

    if early_stopping_callback:
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                       patience=early_stopping_patience)
        callbacks.append(early_stopping)

    # Train the model
    print("Training...")
    history = model.fit(X, y,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2)

    # Save the model, config, final metrics, and loss/accuracy plots for future reference
    evaluate_history(history)
    print("Model saved to {}".format(model_file))
    model.save(model_file)
    copyfile("config.yaml", model_file[:-3] + "_config.yaml")

if __name__ == "__main__":
    main()
