# NB_Classifier.py
使用Bayes中的MultinomialNB分布实现多分类任务
```
MNB支持TF-IDF.如下：
The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.
```
+ 注意：
> 1. 需要建立数据集，训练一个较好的TF_IDF模型保存数据
> 2. 分层模型的训练集，在第一层最好包含所有类的数据
```
第一层分类结果
                    precision    recall  f1-score   support

           0       0.68      0.58      0.63       100
           1       0.75      0.91      0.82       129
           2       0.88      0.84      0.86       127
           3       0.79      0.87      0.83       110
           4       0.68      0.66      0.67       141
           5       0.75      0.69      0.72       121
           6       0.75      0.68      0.71       155
           7       0.79      0.86      0.82       117

    accuracy                           0.76      1000
   macro avg       0.76      0.76      0.76      1000
weighted avg       0.76      0.76      0.76      1000
第二层分类结果
                precision    recall  f1-score   support

           0       0.78      0.54      0.64        13
           1       0.39      0.91      0.55        53
           2       0.60      0.57      0.59        42
           3       0.50      0.14      0.22        36
           4       0.83      0.48      0.61        21
           5       0.55      0.84      0.67        62
           6       0.41      0.60      0.48        62
           7       0.86      0.26      0.40        23
           8       0.50      0.85      0.63        27
           9       0.48      0.44      0.46        25
          10       0.78      0.29      0.42        24
          11       0.26      0.12      0.17        40
          12       1.00      0.08      0.15        24
          13       0.00      0.00      0.00         2
          14       1.00      0.50      0.67        12
          15       0.00      0.00      0.00        27
          16       0.65      0.35      0.45        43
          17       0.56      0.50      0.53        40
          18       0.50      0.12      0.20        24
          19       0.60      0.33      0.43         9
          20       0.00      0.00      0.00        13
          21       0.00      0.00      0.00        10
          22       0.61      0.44      0.51        57
          23       0.00      0.00      0.00         5
          24       0.37      0.44      0.41        36
          25       0.58      0.69      0.63        54
          26       0.36      0.83      0.51        29
          27       0.00      0.00      0.00         9
          28       0.00      0.00      0.00         3
          29       0.00      0.00      0.00         5
          30       0.00      0.00      0.00        21
          31       0.00      0.00      0.00         9
          32       0.52      0.93      0.67        61
          33       0.36      0.63      0.46        65
          34       0.00      0.00      0.00        11
          35       0.00      0.00      0.00         3

    accuracy                           0.48      1000
   macro avg       0.39      0.33      0.32      1000
weighted avg       0.47      0.48      0.43      1000

```