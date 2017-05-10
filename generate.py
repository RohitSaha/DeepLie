import sys
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


# load ascii text and covert to lowercase
filename = "TRUMP_speeches.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

filename = "Weights/weights-improvement=68-0.244175.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#pick a random seed
start = numpy.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
print ("Seed : ")
generated = ''.join([int_to_char[value] for value in pattern])
print (generated)

def sample(preds, temperature):
    # helper function to sample an index from a probability array
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)



for i in range(500):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)[0]
    index = numpy.argmax(prediction)
    #index = sample(prediction, 0.1)
    result = int_to_char[index]
    #generated += result
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print ("Done")


epochs = [1, 5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 68]
loss = [2.699306, 2.279430, 1.836922, 1.413346, 1.051214, 0.772619, 0.582175, 0.381016, 0.334254, 0.302160, 0.262853, 0.244175]



'''
01-2.699306
05-2.279430  
10-1.836922
15-1.413346
20-1.051214  
25-0.772619
30-0.582175
40-0.381016  
45-0.334254
50-0.302160
60-0.262853
68-0.244175
'''
