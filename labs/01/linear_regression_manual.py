#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # The input data are in dataset.data, targets are in dataset.target.
    
    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # TODO: Append a new feature to all input data, with value "1"
    dataset.data = np.c_[dataset.data, np.ones(506)]
    #print(dataset.data)
    #print()
    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size = args.test_size, random_state = args.seed)
    #print("train")
    #print(train_data)
    #print(train_target)
    #print("test")
    #print(test_data)
    #print(test_target)
    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    weigths = np.linalg.inv( np.transpose(train_data) @ train_data ) @ np.transpose(train_data) @ train_target
    #print("weigths")
    #print(weigths)
    # TODO: Predict target values on the test set
    predicted = test_data @ weigths
    #print("predicted targets")
    #print(predicted)
    # TODO: Compute root mean square error on the test set predictions
    rmse = sklearn.metrics.mean_squared_error(test_target, predicted, squared=False)

    return rmse

if __name__ == "__main__":
    args = parser.parse_args()
    rmse = main(args)
    print("{:.2f}".format(rmse))
