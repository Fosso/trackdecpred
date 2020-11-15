import pandas as pd
import matplotlib as plt
from models.knn.knn import run_knn
from models.decisiontree.dt import run_dt_on_dataset
from models.suppoertvectormachines.svm import run_svm_on_dataset

import numpy as np


def get_metrics(experiment):
    accuracy = []
    f1_score = []

    if experiment == 3:
        #optimal k = 70
        knn_cv_mean, knn_f1_micro = run_knn(70, 3)
        accuracy.append(knn_cv_mean)
        f1_score.append(knn_f1_micro)

        dt_cv_mean, dt_f1_micro = run_dt_on_dataset(3)
        accuracy.append(dt_cv_mean)
        f1_score.append(dt_f1_micro)

        #optimal solution is with Gaussian kernel
        svm_cv_mean, svm_f1_micro = run_svm_on_dataset(3, "g", 0)
        accuracy.append(svm_cv_mean)
        f1_score.append(svm_f1_micro)

    elif experiment == 5:
        # optimal k = 11
        knn_cv_mean, knn_f1_micro = run_knn(11, 5)
        dt_cv_mean, dt_f1_micro = run_dt_on_dataset(5)
        # optimal solution is with polynomial kernel with degree 3
        svm_cv_mean, svm_f1_micro = run_svm_on_dataset(5, "p", 3)
    else:
        "Can only do this for experiment 3 and 5"

    print("hei dette er funksjonen get_metric:", accuracy, f1_score)
    return accuracy, f1_score




"""
    # Metrics from knn
    knn_cv_mean, knn_f1_micro = run_knn(2, 3)
    print("For knn")
    print("CV MEAN:: ", knn_cv_mean)
    print("F1 MICRO:: ", knn_f1_micro)
    print("------------")

    # Metrics from dt
    dt_cv_mean, dt_f1_micro = run_dt_on_dataset(3)
    print("For dt")
    print("CV MEAN:: ", dt_cv_mean)
    print("F1 MICRO:: ", dt_f1_micro)



    # Metrics from svm
    svm_cv_mean, svm_f1_micro = run_svm_on_dataset(5, "l,", 0)
    print("For SVM")
    print("CV MEAN:::::: ", svm_cv_mean)
    print("F1 MICRO:::::: ", svm_f1_micro)

"""


# print("hei")
# print(class_report.keys())
# exp3_scores = []
# knn_accuracy = round(class_report["accuracy"], 4)
# knn_macro_avg = class_report["macro avg"]
# print(round(knn_macro_avg, 4))
# knn_weighted_avg = round(class_report["weighted avg"], 4)
# print("Acc::: ", knn_accuracy, knn_macro_avg, knn_weighted_avg)
# knn

def accuracy_graph(method,accuracy_list_knn, accuracy_list_dt, accuracy_list_svm):
    labels = ['Accuracy', 'F1 score']
    accuracy_list_dt = ['G1', 'G2', 'G3', 'G4', 'G5']
    accuracy_list_knn = []
    accuracy_list_svm = []
    men_means = [20, 34, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width /3, men_means, width, label='KNN')
    rects2 = ax.bar(x + width / 3, women_means, width, label='Decision Tree')
    rects3 = ax.bar(x + width / 3, women_means, width, label='Support Vector Machiene')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage')
    ax.set_title('Experimant', method)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()



get_metrics(3)