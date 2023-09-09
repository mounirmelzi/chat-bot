from collections import namedtuple
import random
import os
import json
import numpy as np
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

TrainingData = namedtuple("TrainingData", ["x_train", "y_train"])
Data = namedtuple("Data", ["tags", "tags_map", "all_words"])

training_data_pickle_file_path = "./files/training_data.pickle"
data_pickle_file_path = "./files/data.pickle"

# create the files folder if it does not exist
if not os.path.exists("./files"):
    os.mkdir("./files")


def load_json():
    file_path = "intents.json"
    with open(file_path) as file:
        file_content = json.load(file)

    return file_content["intents"]


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(stemmed_tokenized_sentence, words):
    bag = np.zeros(len(words), dtype=np.float32)
    for i, w in enumerate(words):
        if w in stemmed_tokenized_sentence:
            bag[i] = 1

    return bag


def generate_training_data():
    # process data
    intents = load_json()
    ignore_words = ["?", ".", "!"]

    all_words = []
    tags = []
    xy = []

    for intent in intents:
        tag = intent["tag"]
        tags.append(tag)

        for pattern in intent["patterns"]:
            words = tokenize(pattern)
            words = [stem(word) for word in words if word not in ignore_words]
            all_words.extend(words)
            xy.append((words, tag))

    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    random.shuffle(xy)

    # generate training data
    x_train = []
    y_train = []

    for pattern, tag in xy:
        x_train.append(bag_of_words(pattern, all_words))

        tag_index = tags.index(tag)
        y_train.append([1 if index == tag_index else 0 for index in range(len(tags))])

    save_data(intents, tags, all_words)
    return TrainingData(x_train=np.array(x_train), y_train=np.array(y_train))


def get_training_data() -> TrainingData:
    training_data = TrainingData(x_train=np.array([]), y_train=np.array([]))

    try:
        with open(training_data_pickle_file_path, "rb") as f:
            training_data = pickle.load(f)
    except:
        training_data = generate_training_data()
        with open(training_data_pickle_file_path, "wb") as f:
            pickle.dump(training_data, f)

    return training_data


def save_data(intents, tags, all_words):
    tags_dict = dict()

    for intent in intents:
        tag = intent["tag"]
        responses = intent["responses"]
        tags_dict[tag] = responses

    data = Data(tags, tags_dict, all_words)

    with open(data_pickle_file_path, "wb") as f:
        pickle.dump(data, f)


def load_data() -> Data:
    try:
        with open(data_pickle_file_path, "rb") as f:
            data = pickle.load(f)

        return data
    except:
        print("[ERROR] pickle file not found: can't load the data")
        return None
