import pandas as pd
import matplotlib.pyplot as plt
from models.knn.knn import run_knn
from models.decisiontree.dt import run_dt_on_dataset
from models.supportvectormachines.svm import run_svm_on_dataset
import numpy as np


# Runs models with optimal hyperparams and get metrics(accuracy and f1-score), for experiment 3 or 5
# Decide which experiment to show results for in the bottom
def get_metrics(experiment):
    accuracy = []
    f1_score = []

    if experiment == 3:
        # optimal k = 70
        knn_cv_mean, knn_f1 = run_knn(80, 3, False)
        accuracy.append(knn_cv_mean)
        f1_score.append(knn_f1)

        dt_cv_mean, dt_f1 = run_dt_on_dataset(3)
        accuracy.append(dt_cv_mean)
        f1_score.append(dt_f1)

        # optimal solution is with Gaussian kernel
        svm_cv_mean, svm_f1 = run_svm_on_dataset(3, "g", 0)
        accuracy.append(svm_cv_mean)
        f1_score.append(svm_f1)

    elif experiment == 5:
        # optimal k = 70
        knn_cv_mean, knn_f1 = run_knn(11, 5, False)
        accuracy.append(knn_cv_mean)
        f1_score.append(knn_f1)

        dt_cv_mean, dt_f1 = run_dt_on_dataset(5)
        accuracy.append(dt_cv_mean)
        f1_score.append(dt_f1)

        # optimal solution is with Gaussian kernel
        svm_cv_mean, svm_f1 = run_svm_on_dataset(5, "p", 4)
        accuracy.append(svm_cv_mean)
        f1_score.append(svm_f1)

    else:
        "Can only do this for experiment 3 and 5"

    return accuracy, f1_score

# Shows plot that comparises accuracy and f1 score for experiment 3 or 5
def accuracy_graph(experiment):
    accuracy, f1_score = get_metrics(experiment)
    labels = ['Accuracy', 'F1 score']
    knn = []
    dt = []
    svm = []
    knn.extend((accuracy[0].round(3), f1_score[0].round(3)))
    dt.extend((accuracy[1].round(3), f1_score[1].round(3)))
    svm.extend((accuracy[2].round(3), f1_score[2].round(3)))

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, knn, width, label='KNN', color='royalblue')
    rects2 = ax.bar(x + width, dt, width, label='DT', color='deepskyblue')
    rects3 = ax.bar(x + width * 2, svm, width, label='SVM', color='paleturquoise')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage')
    # ax.set_title('Experiment')
    ax.set_xticks(x + width)

    # Add some text for labels, title and custom x-axis tick labels, etc.

    if experiment == 5:
        ax.set_title('Experiment 5')
    else:
        ax.set_title('Experiment 3')

    ax.set_xticklabels(labels)

    ax.legend()

    # for aligning labels in the plot
    def autolabel(rects):
        # Attach a text label above each bar in *rects*, displaying its height.
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 3, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Show plot
    fig.tight_layout()
    # creating files for results
    # plt.savefig("../../results/accuracy_f1_results_exp3.png")
    # plt.savefig("../../results/accuracy_f1_results_exp5.png")

# Decide which experiment to run, 3 or 5
# Optimal run returning plots of accuracy and f1 scores.
# accuracy_graph(3)
# accuracy_graph(5)
