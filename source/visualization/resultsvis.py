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
        # optimal k = 70
        knn_cv_mean, knn_f1_micro = run_knn(11, 5)
        accuracy.append(knn_cv_mean)
        f1_score.append(knn_f1_micro)

        dt_cv_mean, dt_f1_micro = run_dt_on_dataset(5)
        accuracy.append(dt_cv_mean)
        f1_score.append(dt_f1_micro)

        # optimal solution is with Gaussian kernel
        svm_cv_mean, svm_f1_micro = run_svm_on_dataset(5, "p", 3)
        accuracy.append(svm_cv_mean)
        f1_score.append(svm_f1_micro)

    else:
        "Can only do this for experiment 3 and 5"

    return accuracy, f1_score

def accuracy_graph(experiment):
    accuracy, f1_score = get_metrics(experiment)
    labels = ['Accuracy', 'F1 score']
    knn = []
    dt = []
    svm = []
    knn.extend((accuracy[0], f1_score[0]))
    dt.extend((accuracy[1], f1_score[1]))
    svm.extend((accuracy[2], f1_score[2]))

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, knn, width, label='KNN')
    rects2 = ax.bar(width / 2, dt, width, label='Decision Tree')
    rects3 = ax.bar(x + width / 2, svm, width, label='Support Vector Machiene')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage')
    ax.set_title('Experiment', experiment)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        #Attach a text label above each bar in *rects*, displaying its height.
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.show()


accuracy_graph(5)

