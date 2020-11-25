# TDT4173 Machine Learning Project

## Table Of Content

* [Folder Structure](https://github.com/Fosso/trackdecpred/tree/readme#folder-structure)
* [How To Run](https://github.com/Fosso/trackdecpred/tree/readme#how-to-run)
* [Main Files](https://github.com/Fosso/trackdecpred/tree/readme#main-files)

## Folder Structure 



```bash
│   .gitignore
│   LICENSE
│   main.py
│   README.md
│   requirements.txt
│
│
├───data
│       cleanneddata_exp1.csv
│       cleanneddata_exp2.csv
│       cleanneddata_exp2_Test.csv
│       cleanneddata_exp3.csv
│       cleanneddata_exp4.csv
│       cleanneddata_exp5.csv
│       procdata.csv
│       rawdata.csv
│
├───models
│   ├───decisiontree
│   │   │   dt.py
│   │
│   ├───knn
│   │   │   knn.py
│   │   │   knn_test.py
│   │   │   knn_year.py
│   │   │   newknn.py
│   │  
│   │
│   └───supportvectormachines
│       │   svm.py
│
├───results
│   ├───comparison
│   │       accuracy_f1_results_exp3.png
│   │       accuracy_f1_results_exp5.png
│   │
│   ├───dt
│   │       confusion_matrix_exp3.png
│   │       confusion_matrix_exp5.png
│   │       execution_time_exp3_exp5_dt
│   │       optimal_md_exp3.png
│   │       optimal_md_exp5.png
│   │       tree_md_2_example.png
│   │
│   ├───exploration
│   │       histogram_plot_decades.png
│   │       line_plot_decades.png
│   │       scatter_plot_exp5_acousticness_energy.png
│   │
│   ├───knn
│   │       confusion_matrix_exp3.png
│   │       confusion_matrix_exp5.png
│   │       execution_time_exp3_exp5_knn
│   │       optimal_k_exp1.png
│   │       optimal_k_exp2.png
│   │       optimal_k_exp3.png
│   │       optimal_k_exp4.png
│   │       optimal_k_exp5.png
│   │
│   └───svm
│           confusion_matrix_exp3.png
│           confusion_matrix_exp5_p3.png
│           confusion_matrix_exp5_p4.png
│           execution_time_exp3_exp5_svm
│           exp_3_and_exp_5
│
└───source
    ├───data
    │       generaldataprocess.py
    │
    └───visualization
            procdatavis.py
            resultsvis.py

```


## How To Setup Environment

```bash
conda create --name <environment-name> --file <requirements.txt>
```

```bash
conda activate <environment-name>
```

```bash
conda install
```


## Main Files
### main.py

Example 1, runs knn for experiment 3 with search for optimal k: 
```bash 
main.py -knn -exp 3 -o
```
Example 2, runs svm for experiment 5 with optimal configuration:
```bash 
main.py -svm -exp 5
```

Hyperparameters can be changed by editing a few lines of code in the file.


#### decisiontree.py
Builds dt classifier, fits the model, evaluates the performance of the model. Contains method for searching for optimal max depth by iterative experimentation.

#### knn.py
Builds knn classifier, fits the model, evaluates the performance of the model. Contains method for searching for optimal k by iterative experimentation.


#### svm.py
Builds svm classifier, fits the model, evaluates the performance of the model.


#### knn_year.py
This file is solely for experiment 1(not one of the main experiments)
Builds knn classifier, fits the model, evaluates the performance of the model. Contains method for searching for optimal k by iterative experimentation.

#### knn_test.py
To test the knn classifier for a song that is not in the dataset. Audio features are gathered from the Spotify Web API.

Builds knn classifier, fits the model, evaluates the performance of the model. Contains method for searching for optimal k by iterative experimentation.



### procdatavis.py

Visualization for data exploration. Needs to be run seperately.

### resultvis.py

Visualization for results. Needs to be run seperately.

### generaldataprocess.py

Pre-processing of dataset. Not necessary to run as subsets are created and stored in the repository under the directory "data".