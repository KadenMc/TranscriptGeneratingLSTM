# TranscriptGeneratingLSTM
Extract YouTube transcripts and reproduce them using a character-level text generation LSTM.

In many ways, the goal of this repository is to make it as easy as possible to take a YouTube channel - or personality - and reproduce their content in the form of a transcript. Personally, I wanted to reproduce Bob Ross' cathartic painting content!

## Character-Level LSTM
The character-level text generation LSTM is implemented in Tensorflow's Keras, and consists simply of an `LSTM` layer with `dropout` and `recurrent_dropout`, as well as a `Dense` output layer with a `softmax` activation, where each node represents a character.

## Training

There are no command line arguments for running `train.py`, instead the `config.yaml` file houses the relevant paths and model hyperparameters.

There are multiple convienent features implemented such as:
 * Automatic processing of train/validation data from a .txt file
 * Optional text generation for analyzing progress at the end of each epoch - demo as you go!
 * Optional checkpoint saving upon achieving a new smallest validation loss.
 * Optional early stopping with patience parameter.
 * Automatic saving of the model, metrics, plots of metrics, and config upon completing training.

## Demo

The `demo.py` file allows the testing of a model with prompts from the data or from user input. Temperature (diversity) may be changed whilst running the demo and the length of generated text may be changed (see command line arguments).

## YouTube Transcript Extraction

Automatically extract transcripts from YouTube videos from their links using `extractYouTubeTranscripts.py`! There is the option of extracting specified links in a CSV, or extracting all video transcripts from a given channel. See the command line arguments for `extractYouTubeTranscripts.py` for more information on this.

### `links_example.csv`

| title                                                      | link        |
|------------------------------------------------------------|-------------|
| Bob Ross - Wilderness Day (Season 31   Episode 13)         | nJGCVFn57U8 |
| Bob Ross - In the Midst of Winter (Season   31 Episode 12) | qx2IsmrCs3c |
| Bob Ross - Lake at the Ridge (Season 31   Episode 11)      | 8QWvzEQ69Kw |
| Bob Ross - Balmy Beach (Season 31 Episode   10)            | kMgd6r6c4vE |

The links in the CSV may also be raw, copied links e.g. `https://www.youtube.com/watch?v=nJGCVFn57U8`.

**Note:** When extracting transcripts, it's highly encouraged to pre-process the output before training a model with it. Even just converting the text to lowercase and getting rid of all non alpha-numeric-puncuation characters can greatly improve model performance - though all data is unique.

## Command line arguments

### `demo.py`
 * `--data`: Path to the text_file.txt file (otherwise use text_file in config.yaml)'
 * `--model`: Path to the model.h5 file (otherwise use model_file in config.yaml)'
 * `--xy_file`: Path to data.npz file to get characters (otherwise use Xy_file in config.yaml)'
 * `--temperature`: The temperature, or diversity (otherwise use temperature in config.yaml)
 * `--length`: The length of the text generated


### `extractYouTubeTranscripts.py`
* `data`: Path to save all tanscript .txt data
* `--links`: Path for csv file containing video titles and links
* `--channel`: Download all video transcripts from a given channel ID (Requires API key)
* `--separate`: Path to save transcripts as separate .txt files


The below table represents the first few links in a CSV automatically generated from channel ID UCxcnsr1R5Ge_fbTu5ajt8DQ (Bob Ross' channel).



## API Key

An API key is necessary to get a channel's videos. You can explore getting one here: https://developers.google.com/youtube/v3.


Notes:
 * Getting any API key is not necessary if simply specifying links in a CSV.

 * There is a limit to the number of requests, and you may get a `Forbidden` error if making too many requests too quickly (there is an optional parameter to create a delay).


## Common Errors:

 * *Error: WARNING:tensorflow:Model was constructed with shape (None, 100, 42) for input Tensor("lstm_input:0", shape=(None, 100, 42), dtype=float32), but it was called on an input with incompatible shape (None, 80, 42).*

    Potential problem: You changed the `max_length` parameter in the `config.yaml`. In this example, from 80 to 100.

    Fix 1: Delete/rename the `Xy.npz` file (see current name in config under `Xy_file`), which will automatically reload it from current text data.

    Fix 2: Revert back to the original `max_length` value


## Download Pretrained Models

Coming soon!

## Download cleaned Bob Ross training data

Coming soon!

## Current Shortcomings & Future Improvements
 * Upload a pre-trained model file
 * Upload cleaned Bob Ross training data
 * A basic, optional pre-processing file with (more or less) univerisally applicable steps for easy transcript pre-processing
 * Ability to continue training an existing model
