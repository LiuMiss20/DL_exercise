# coding:utf-8

## python 3.x

import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt


# Load data  set 
cifar10_dir = 'datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# subsample
num_training = 5000  # 从 50000 中 抽取 500 
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500  # 从 10000 中 抽取 50 
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# reshape
X_train = np.reshape(X_train, (X_train.shape[0],-1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))



# Create a kNN classifier instance
from classifiers.k_nearest_neighbor import KNearestNeighbor # 需  pip install future
classifier = KNearestNeighbor()



# Cross-validation
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = []
    
# 找出 最好的 k-value    
for k in k_choices: 
    for i in range(num_folds):
        X_train_cv = np.vstack(X_train_folds[:i] + X_train_folds[i+1:])
        X_test_cv = X_train_folds[i]
        
        y_train_cv = np.hstack(y_train_folds[:i] + y_train_folds[i+1:])
        y_test_cv = y_train_folds[i]
        
        classifier.train(X_train_cv, y_train_cv)
        dists_cv = classifier.compute_distances_no_loops(X_test_cv)
        
        y_test_pred = classifier.predict_labels(dists_cv, k)
        num_correct = np.sum(y_test_pred == y_test_cv)
        accuracy = float(num_correct)/y_test_cv.shape[0]
        
        k_to_accuracies[k].append(accuracy)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))


# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()