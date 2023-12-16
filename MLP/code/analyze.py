import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import binarize, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")



def encoder(dataset, catFeatures, qtyFeatures):
    dataset = dataset[catFeatures + qtyFeatures]
    dataset_encoded = pd.get_dummies(dataset,
                                     columns=catFeatures,
                                     drop_first=True)

    return (dataset_encoded)


def plot_auc_curve(model, X, y):
    try:
        y_pred_prob = model.predict_proba(X)[:, 1]
    except:
        d = model.decision_function(X)
        y_pred_prob = np.exp(d) / (1 + np.exp(d))

    auc = roc_auc_score(y, y_pred_prob)
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)

    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC Curve\n AUC={auc}'.format(auc=auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)


def model_training(model, X, y):
    model.fit(X, y)

    return (model)


def print_accurcay_metrics(model, X, y, threshold):
    try:
        y_pred_prob = model.predict_proba(X)[:, 1]
    except:
        d = model.decision_function(X)
        y_pred_prob = np.exp(d) / (1 + np.exp(d))
    y_pred_class = binarize([y_pred_prob], threshold)[0]

    print("Accurcay:", accuracy_score(y, y_pred_class))
    print("AUC:", roc_auc_score(y, y_pred_prob))
    print("Log Loss:", log_loss(y, y_pred_prob))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred_class))
    print("Recall:", recall_score(y, y_pred_class))
    print("Precision:", precision_score(y, y_pred_class))


def Find_Optimal_Cutoff(model, X, y):
    try:
        y_pred_prob = model.predict_proba(X)[:, 1]
    except:
        d = model.decision_function(X)
        y_pred_prob = np.exp(d) / (1 + np.exp(d))

    fpr, tpr, threshold = roc_curve(y, y_pred_prob)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf - 0).abs().argsort()[:1]]

    print("Optimal Cutoff:", roc_t['threshold'].values)
    return (roc_t['threshold'].values)


def feature_importance(model, X):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


# Plot calibration plots
def plot_calibration(y_true, y_prob, n_bins, model_name):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_true, y_prob, n_bins=n_bins)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (model_name,))

    ax2.hist(y_prob, range=(0, 1), bins=10, label=model_name,
             histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper right", ncol=2)

    plt.tight_layout()
    plt.show()