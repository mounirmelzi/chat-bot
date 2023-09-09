import data
import tflearn
import numpy as np
import random

MODEL_PATH = "./files/model.tflearn"


class Model:
    def __init__(self) -> None:
        self.training_data = data.get_training_data()
        self.data = data.load_data()

        self.input_size = len(self.training_data.x_train[0])
        self.output_size = len(self.training_data.y_train[0])
        self.hidden_size = 8

        net = tflearn.input_data(shape=[None, self.input_size])
        net = tflearn.fully_connected(net, self.hidden_size)
        net = tflearn.fully_connected(net, self.hidden_size)
        net = tflearn.fully_connected(net, self.output_size, activation="softmax")
        net = tflearn.regression(net)

        self.model = tflearn.DNN(net)

    def train(self):
        self.model.fit(
            self.training_data.x_train,
            self.training_data.y_train,
            n_epoch=1000,
            batch_size=8,
            show_metric=True,
        )
        self.model.save(MODEL_PATH)

    def load(self):
        self.model.load(MODEL_PATH)

    def predict(self, prompt):
        prompt = data.tokenize(prompt)
        prompt = [data.stem(word) for word in prompt]
        prompt = data.bag_of_words(prompt, self.data.all_words)

        results = self.model.predict([prompt])[0]
        index = np.argmax(results)
        accuracy = results[index]

        if accuracy < 0.7:
            return "I didn't get that, try again."

        tag = self.data.tags[index]
        return random.choice(self.data.tags_map[tag])
