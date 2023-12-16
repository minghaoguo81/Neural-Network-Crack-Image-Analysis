import os
import csv
import numpy as np
import pandas as pd
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt
import sys
import cv2
import imageio.v3 as iio
from skimage.transform import resize
import NeuralNetMLP as nnm
import analyze as an
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import binarize, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import calibration_curve
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

train_data_dir = "../concerete_crack_images/training/"
test_data_dir = "../concerete_crack_images/test/"

train_imgs = os.listdir(train_data_dir)
test_imgs = os.listdir(test_data_dir)

print(len(train_imgs))
print(len(test_imgs))


# Prepare x and y
num_px = 64

def extrcatFeaturesAndLabels(dir, impg_dataset):
    X = np.zeros((len(impg_dataset), num_px * num_px * 3))
    y = np.zeros((len(impg_dataset)))
    for i in range(0, len(impg_dataset)):
        # Read an image from a file as an array.
        # The different colour bands/channels are stored
        # in the third dimension, such that a
        # grey-image is MxN, an RGB-image MxNx3
        #image = np.array(iio.imread(dir + impg_dataset[i]))
        image = cv2.imread(os.path.join(dir, impg_dataset[i]))
        image = cv2.resize(image, (num_px, num_px))
        image = image.reshape((1, num_px * num_px * 3)).T
        image = image / 255.0
        image = image.reshape((num_px * num_px * 3, 1))
        for j in range(0, num_px * num_px * 3):
            X[i][j] = image[j][0]

        if 'pos' in impg_dataset[i]:
            y[i] = 1
        else:
            y[i] = 0

    y = y.astype(int)

    return (X, y)


def get_acuuracy(model, X, y):
    y_pred = model.predict(X)

    if sys.version_info < (3, 0):
        accuracy = ((np.sum(y == y_pred, axis=0)).astype('float') /
                    X.shape[0])
    else:
        accuracy = np.sum(y == y_pred, axis=0) / X.shape[0]

    # print('Accuracy: %.2f%%' % (accuracy * 100))

    return (accuracy * 100)

# Training Data Set
X_train, y_train = extrcatFeaturesAndLabels(train_data_dir, train_imgs)
X_test, y_test = extrcatFeaturesAndLabels(test_data_dir, test_imgs)

logreg = LogisticRegression()
an.model_training(logreg, X_train, y_train)

training_accuracy=get_acuuracy(logreg, X_train, y_train)
print('Training accuracy: %.2f%%' %training_accuracy)

test_accuracy=get_acuuracy(logreg, X_test, y_test)
print('Test accuracy: %.2f%%' %test_accuracy)

an.plot_auc_curve(logreg, X_test, y_test)

an.plot_auc_curve(logreg, X_train, y_train)

rf = RandomForestClassifier(n_estimators=1000,
                            criterion='gini',
                            max_features='sqrt',
                            n_jobs=-1)

an.model_training(rf, X_train, y_train)

an.plot_auc_curve(rf, X_test, y_test)
an.Find_Optimal_Cutoff(rf, X_test, y_test)

training_accuracy=get_acuuracy(rf, X_train, y_train)
print('Training accuracy: %.2f%%' %training_accuracy)

test_accuracy=get_acuuracy(rf, X_test, y_test)
print('Test accuracy: %.2f%%' %test_accuracy)

y_test_pred = rf.predict(X_test)

miscl_img = X_test[y_test != y_test_pred][:20]
correct_lab = y_test[y_test != y_test_pred][:20]
miscl_lab= y_test_pred[y_test != y_test_pred][:20]

fig, ax = plt.subplots(nrows=4, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(20):
    img = miscl_img[i].reshape(num_px, num_px, 3)
    ax[i].imshow(img)
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

adaBoost = AdaBoostClassifier(n_estimators=150)
an.model_training(adaBoost, X_train, y_train)

an.plot_auc_curve(adaBoost, X_test, y_test)
#an.Find_Optimal_Cutoff(adaBoost, X_test, y_test)
#an.print_accurcay_metrics(adaBoost, X_test, y_test, 0.5)

an.Find_Optimal_Cutoff(adaBoost, X_test, y_test)

y_test_pred = adaBoost.predict(X_test)

miscl_img = X_test[y_test != y_test_pred][:10]
correct_lab = y_test[y_test != y_test_pred][:10]
miscl_lab= y_test_pred[y_test != y_test_pred][:10]

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = miscl_img[i].reshape(num_px, num_px, 3)
    ax[i].imshow(img)
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

NN = MLPClassifier(solver='adam', alpha=1e-3,
                   hidden_layer_sizes=(50,40,30), random_state=1, max_iter=1000, activation='logistic')

an.model_training(NN, X_train, y_train)
training_accuracy=get_acuuracy(NN, X_train, y_train)
print('Training accuracy: %.2f%%' %training_accuracy)

test_accuracy=get_acuuracy(NN, X_test, y_test)
print('Test accuracy: %.2f%%' %test_accuracy)