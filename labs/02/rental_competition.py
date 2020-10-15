#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.metrics

import numpy as np

class Dataset:
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")
parser.add_argument("--feature_path", default="rental_competition.features", type=str, help="Features path")
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)

        # Create a random generator with a given seed
        generator = np.random.RandomState(args.seed)

        train = Dataset()
        data = train.data
        target = train.target


        # making features
        bool_train = np.all(data.astype(int) == data, axis=0)
        int_train = [i for i, b in enumerate(bool_train) if b]
        real_train = [i for i, b in enumerate(bool_train) if not b]

        i = "tr"
        transformers = []
        for arr, encoder in zip([int_train, real_train],
                                [sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                 sklearn.preprocessing.StandardScaler()]):

            if len(arr) > 0:
                transformers.append((i, encoder, arr))
                i += 'a'

        ct = sklearn.compose.ColumnTransformer(transformers=transformers)
        poly = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)

        pipeline = sklearn.pipeline.Pipeline([ ('ct', ct), ('pl', poly) ])
        data = pipeline.fit_transform(data)
        
        # TODO: Train a model on the given dataset and store it in `model`.
        # adding ones


        data = np.c_[data, np.ones(data.shape[0])]

        # Generate initial linear regression weights
        weights = generator.uniform(size=data.shape[1])

        for epoch in range(args.epochs):
            permutation = generator.permutation(data.shape[0])

            i = 0
            n_batches = data.shape[0] // args.batch_size
            for batch in range(n_batches):
                gradient_sum = np.zeros(shape=weights.shape)

                for sample in range(args.batch_size):
                    index = permutation[i + sample]
                    predictions = np.transpose(data[index]) @ weights
                    gradient_sum += (predictions - target[index]) * data[index]

                average_gradient = gradient_sum / args.batch_size

                # SGD update
                weights = weights - args.learning_rate * average_gradient

                i = i + args.batch_size

        model = weights

        # Serialize features
        with lzma.open(args.feature_path, "wb") as feature_file:
            pickle.dump(pipeline, feature_file)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)


    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        with lzma.open(args.feature_path, "rb") as feature_file:
            pipeline = pickle.load(feature_file)

        test_d = pipeline.transform(test.data)
        test_d = np.c_[test_d, np.ones(test_d.shape[0])]

        # TODO: Generate `predictions` with the test set predictions.
        predictions = test_d @ model

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
