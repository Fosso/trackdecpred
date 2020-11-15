import pandas as pd
import matplotlib as plt
from models.knn.knn import run_knn
from models.decisiontree.dt import run_dt_on_dataset
from models.suppoertvectormachines.svm import run_svm_on_dataset


def get_metrics():

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

"""
    # Metrics from svm
    svm_cv_mean, svm_f1_micro = run_svm_on_dataset(5, "l,", 0)
    print("For SVM")
    print("CV MEAN:::::: ", svm_cv_mean)
    print("F1 MICRO:::::: ", svm_f1_micro)



# print("hei")
# print(class_report.keys())
# exp3_scores = []
# knn_accuracy = round(class_report["accuracy"], 4)
# knn_macro_avg = class_report["macro avg"]
# print(round(knn_macro_avg, 4))
# knn_weighted_avg = round(class_report["weighted avg"], 4)
# print("Acc::: ", knn_accuracy, knn_macro_avg, knn_weighted_avg)
# knn


get_metrics()
