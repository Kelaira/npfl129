#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.compose
import sklearn.linear_model
import sklearn

import numpy as np

class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
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
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--iterations", default=50, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")

def main(args):
    if args.predict is None:
        # Create a random generator with a given seed
        generator = np.random.RandomState(args.seed)

        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        data = train.data
        target = train.target

        # DATA PREPROCESSING

        # Append a constant feature with value 1 to the end of every input data
        data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

        # transform real values with `sklearn.preprocessing.StandardScaler`
        # transformers = [('stscaler', sklearn.preprocessing.StandardScaler(), slice(15,21))]
        # column_trans = sklearn.compose.ColumnTransformer(transformers=transformers)

        minmax_scaler = sklearn.preprocessing.MinMaxScaler()
        # Generate polynomial features of order 2 from the current features.
        poly_features = sklearn.preprocessing.PolynomialFeatures(2)

        # logistic regression
        log_reg = sklearn.linear_model.LogisticRegression()

        pipeline = sklearn.pipeline.Pipeline([('mm', minmax_scaler),('pf', poly_features), ('lr', log_reg)])

        params = {'pf__degree': [1, 2],
                  'lr__C': [0.01, 0.1, 1, 10, 50, 100, 500],
                  'lr__max_iter': [500],
                  'lr__solver': ['lbfgs', 'sag', 'liblinear'],
                  'lr__random_state': [args.seed]}

        model = sklearn.model_selection.GridSearchCV(estimator=pipeline,
                                                     param_grid=params,
                                                     cv=sklearn.model_selection.StratifiedKFold(10))

        model.fit(data, target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        test_d = np.append(test.data, np.ones((test.data.shape[0], 1)), axis=1)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test_d)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
