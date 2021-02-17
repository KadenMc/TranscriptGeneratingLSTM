from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

class Model():
    def __init__(self, nodes, dropout, recurrent_dropout, max_length, chars):
        self.nodes = nodes
        self.max_length = max_length
        self.char_length = len(chars)

        # Network definition
        self.model = Sequential()
        self.model.add(LSTM(self.nodes, input_shape=(self.max_length, self.char_length),
                            dropout=dropout, recurrent_dropout=recurrent_dropout))
        self.model.add(Dense(self.char_length, activation='softmax'))

    def get_model(self):
        return self.model
