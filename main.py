import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 
import string
import re
from random import randrange

def encode_char(c): 
    n_bits = 5
    encoded = []
    try:
        index = string.ascii_lowercase.index(c)
        for i in range(n_bits):
            encoded.append((index >> i) & 1)
        encoded.reverse()
    except Exception:
        print(c)
        raise Exception()
    return encoded

def encode_word(s):
    encoded = []
    for c in s:
        for n in encode_char(c):
            encoded.append(n)
        
    return encoded

print(encode_word("abcdefg"))

def decode_char(xs):
    xs.reverse()
    index = 0
    l = len(xs)
    for i in range(l):
        index += 2**i * xs[i]
    print(xs)
    print(index)
    return string.ascii_lowercase[index]

def decode_word(ss):
    decoded = ""
    for i in range(int(len(ss) / 5)):
        s = ss[(i*5):(i*5+5)]
        decoded += decode_char(s)
    return decoded

print(decode_word(encode_word("abcdefg")))


def load_real_words(path = "./words.txt"):
    s = ""
    with open(path, "r") as f:
        s = f.read()
    s = s.lower()
    s = s.rstrip().replace("\n", "")
    s = s.rstrip().replace("\r", "")
    s = s.rstrip().replace("\t", "")
    s = s.rstrip().replace("-", "")
    s = s.rstrip().replace("'", "")
    s = s.rstrip().replace("/", "")
    s = s.rstrip().replace("&", "")
    s = s.rstrip().replace(".", "")
    s = s.rstrip().replace("1", "")
    s = s.rstrip().replace("2", "")
    s = s.rstrip().replace("3", "")
    s = s.rstrip().replace("4", "")
    s = s.rstrip().replace("5", "")
    s = s.rstrip().replace("6", "")
    s = s.rstrip().replace("7", "")
    s = s.rstrip().replace("8", "")
    s = s.rstrip().replace("9", "")
    s = s.rstrip().replace("0", "")
    s = s.rstrip().replace("!", "")

    ret = re.sub(r"[^\x00-\x19]","", s)   # <<< This is where the magic happens
    s = s[0 : len(s) - (len(s) % 5)]
    return [s[i*5 : i*5+5] for i in range(int(len(s) / 5))]

def make_fake_data(l):
    out = []
    for j in range(l):
        s = ""
        for i in range(5):
            s += string.ascii_lowercase[randrange(24)]
        out.append(s)
    return out

def shuffle(xs, ys):
    for i in range(len(xs)):
        j = randrange(len(xs))

        a = xs[i]
        xs[i] = xs[j]
        xs[j] = a

        a = ys[i]
        ys[i] = ys[j]
        ys[j] = a
    return xs, ys


real_words = load_real_words()
fake_words = make_fake_data(len(real_words))

x, y = shuffle([encode_word(w) for w in (real_words + fake_words)], [[1] for i in range(len(real_words))] + [[0] for i in range(len(fake_words))])


model = Sequential()
model.add(Dense(8, input_dim=5*5))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(x, y, show_accuracy=True, batch_size=1, nb_epoch=1000)
print(model.predict_proba(x))

