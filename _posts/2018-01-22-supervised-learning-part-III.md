---
layout:     post
title:      Supervised Learning&#58; Part III
date:       2018-01-22 02:10:29
summary:    Learn about the various factors that infuence your model.
categories: machine learning, supervised learning, hyperparameter tuning
---
## Feature Scaling & Mean Normalization

Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. For example, the majority of classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

**Feature scaling** is a method used to standardize the range of independent variables or features of data.

The simplest method is rescaling the range of features to scale the range in [0, 1] or [−1, 1]. Selecting the target range depends on the nature of the data. The general formula is given as:

![fscaling.png](https://djinit-ai.github.io/images/fscaling1.png)

where **x** is an original value, **x'** is the normalized value. For example, suppose that we have the students' weight data, and the students' weights span [160 pounds, 200 pounds]. To rescale this data, we first subtract 160 from each student's weight and divide the result by 40 (the difference between the maximum and minimum weights).

**Mean normalization** is used to make features have approximate zero mean.

![fscaling.png](https://djinit-ai.github.io/images/fscaling2.png)

Another reason why feature scaling and mean normalization are applied is that gradient descent converges much faster with feature scaling than without it.

## Bias & Variance - Problem of underfitting and overfitting

In supervised machine learning an algorithm learns a model from training data. The goal of any supervised machine learning algorithm is to best estimate the mapping function (f) for the output variable (Y) given the input data (X). The mapping function is often called the target function because it is the function that a given supervised machine learning algorithm aims to approximate.


The prediction error for any machine learning algorithm can be broken down into three parts:

 - **Bias Error:** Bias are the simplifying assumptions made by a model to make the target function easier to learn.
 - **Variance Error:** Variance is the amount that the estimate of the target function will change if different training data was used.
 - **Irreducible Error:** It cannot be reduced regardless of what algorithm is used. It is the error introduced from the chosen framing of the problem and may be caused by factors like unknown variables that influence the mapping of the input variables to the output variable.
 
### Underfitting / High Bias
         A statistical model or a machine learning algorithm is said to have underfitting when it cannot capture the underlying trend of the data. Its occurrence simply means that our model or the algorithm does not fit the data well enough. It usually happens when we have less data to build an accurate model and also when we try to build a linear model with a non-linear data.  Underfitting can be avoided by using more data and also reducing the features by feature selection.
 
### Overfitting/ High Variance
         A statistical model is said to be overfitted, when we train it with a lot of data. When a model gets trained with so much of data, it starts learning from the noise and inaccurate data entries in our data set. Then the model does not categorize the data correctly, because of too much of details and noise. The causes of overfitting are the non-parametric and non-linear methods because these types of machine learning algorithms have more freedom in building the model based on the dataset and therefore they can really build unrealistic models. 


![underover.png](https://djinit-ai.github.io/images/underover1.png)
<center>**Underfitting and Overfitting in Linear Regression**</center>
 
![underover.png](https://djinit-ai.github.io/images/underover2.png)
<center>**Underfitting and Overfitting in Logistic Regression**</center>


### How to avoid underfitting:
         - Adding more number of features
         - Adding polynomial features



### How to avoid overfitting:
         - Get more data
         - Reduce number of features by feature selection
         - Regularization
                 https://www.analyticsvidhya.com/blog/2015/02/avoid-over-fitting-regularization/
                 

        

## Regularization

 - To prevent overfitting, You must minimize the values of the parameters theta. To do so, YOu introduce another parameter lambda into the fray. Regularization is a very effective mechanism as it also provides a simpler hypothesis and smoother curves.
 - The regularization parameter lambda will control the trade-off between keeping the parameters small aand fitting the data well. If you set the lambda value to be very large, You'll end up penalizing all the parameters such that all the values of theta will be close to 0 which results in underfitting. If the value of lambda is very small, It won't be much effective.
 - To implement regularization, you add another term to the cost function as shown in the image.
 
 ![](https://cdn-images-1.medium.com/max/1600/1*xmpCVSV1goZzQUwZ8CQKIg.png)
 
 - This form of regularization is also called as L1-regularization. L1-regularization is one of the many techniques for regularization.
         The different regularization techniques are as follows:
        1. L2-regularization / Weight-Decay
![](https://sebastianraschka.com/images/faq/regularized-logistic-regression-performance/l2-term.png)
        2. Dropout regularization
In this method, At each training stage, individual nodes are either "dropped out" of the net with probability 1-p or kept with probability p, so that a reduced network is left; incoming and outgoing edges to a dropped-out node are also removed. Only the reduced network is trained on the data in that stage. The removed nodes are then reinserted into the network with their original weights. The following image depicts how dropout works. 
![](https://www.researchgate.net/profile/Giorgio_Roffo/publication/317277576/figure/fig23/AS:500357438869504@1496305917227/Figure-7-9-An-illustration-of-the-dropout-mechanism-within-the-proposed-CNN-a-Shows-a.png)
By avoiding training all nodes on all training data, dropout decreases overfitting. The method also significantly improves training speed.
        3. Data Augmentation
 - Sometimes, collection of training data is often expensive and laborious. Data augmentation overcomes this issue by artificially inflating the training set with label preserving transformations. 
 - For example, if you have an image dataset, you could add more examples to the dataset by changing the orientation of an existing image in the dataset and add it as an independent image in the dataset. Similarly, you can add random distortions and rotations as independent units in the dataset.
        4. Early stopping
 - Gradient Descent is used to update the learner so as to make it better fit the training data with each iteration.
 - Up to a point, this improves the learner's performance on data outside of the training set.
 - Past that point, however, improving the learner's fit to the training data comes at the expense of increased generalization error.
 - Early stopping rules provide guidance as to how many iterations can be run before the learner begins to over-fit.
![](http://fouryears.eu/wp-content/uploads/2017/12/early_stopping.png)

# Learning curves

 - Learning curve is a plot between the training set size and the error obtained from the cost function.
 - Learning curves can be used to determine whether the model has underfit or overfit.
 - Once you know if there is high bias or high variance, you can either add more features or more examples/data as required.
 - **[An in-depth explanation of Learning curves is given here](http://mlwiki.org/index.php/Learning_Curves)**

# Dataset split
 - Usually, the entire dataset is split into a training dataset, a cross-validation dataset and a test dataset.
 - The training dataset should contain the bulk of the examples, with a few examples in the CV set and test set respectively.
 - Conventionally, the dataset split is 60/20/20 percentage wise. But, this percentage split is subjective. For example, if there are a million examples in the dataset, the CV set and test set will contain 2 million examples each which is not required. So, the split should be done considering the size of the dataset.
 - When there is a big difference between the error in the training set(training error)and the error in the test set (test error), It is said to have a high variance.
 - When the training error itself is high, the model is said to have underfit or is said to have high bias.
 - **[More insights here](https://machinelearningmastery.com/how-to-choose-the-right-test-options-when-evaluating-machine-learning-algorithms/)**

# Initialization methods
There are two methods of initialization: Zero Initialization and Random Initialization.
 - **Zero Initialization** : The parameters are initialized to zero.
 - **Random Initialization** : The parameters are initialized with random values.
 - There is a problem with Zero initialization called as Symmetry breaking. If all of the weights are the same (i.e 0), they will all have the same error and the model will not learn anything - there is no source of asymmetry between the neurons.
 - What we could do, instead, is to keep the weights very close to zero but make them different by initializing them to small, non-zero numbers.It has the same advantage of all-zero initialization in that it is close to the 'best guess' expectation value but the symmetry has also been broken enough for the algorithm to work.
 - Hence, it is generally advised to go with random initialization of parameters / weights.

# Performance Measures
- We already have seen Accuracy as a performance measure. There are three more performance measures which are as follows:
- **Precision** - It is the ratio of correctly predicted positive observations to the total predicted positive observations.
- **Recall** - Ratio of correctly predicted positive observations to all the correctly predicted observations in the class.
- **F-Score** - Harmonic mean / Weighted average of Precision and Recall.
- **[Check this blog for a better understanding of these measures](http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)**
![](https://qph.ec.quoracdn.net/main-qimg-7341ad39241489ab063e1c65ae4ac3b1-c)


# Principal Component Analysis (PCA)
 - PCA is one of the most commonly used algorithms for dimensionality reduction.
 - PCA speeds up supervised learning; it is also used to prevent overfitting which is a bad use of the algorithm.
 - **Problem Formulation** - Trying to find a dimensional surface such that the projectional error is minimized.
 - One should make sure that the features are scaled and mean normalized before applying PCA.
 - **[More insights on PCA here](https://www.dezyre.com/data-science-in-python-tutorial/principal-component-analysis-tutorial)**

# Ensembling Models

Not every Machine Learning algorithm is suitable for all types of problems. SVM may work well with one dataset but it may lack in some other problem. Now let's consider a problem in classification, where we need to classify the dataset into 2 classes say, 0 and 1. Consider the following situation:

Let's use 2 algorithms viz **SVM and Logistic Regression**, and we build 2 different models and use it for classification. Now when we apply our model on the test data, we see that SVM is able to correctly classify the data belonging to Class 0 (i.e data belonging to class 0 is correctly classified as Class 0), whereas it doesn't work well for the data belonging to Class 1 (i.e data belonging to class 1 is wrongly classified as Class 0 ). Similarly Logistic Regression works very well for Class 1 data but not for Class 0 data. 
Now if we combine both of these models (SVM and Logistic Regression) and create a hybrid model, then don't you think the hybrid model will work well with data belonging to both the classes?? Such a hybrid model is known as an **Ensemble Model.**

Ensembling is a good way to increase or improve the accuracy or performance of a model. In simple words, it is the combination of various simple models to create a single powerful model. But there is no guarantee that Ensembling will improve the accuracy of a model. However it does a good stable model as compared to simple models.

Ensembling can be done in ways like:

 - Voting Classifier

 - Bagging

 - Boosting.

## Voting Classifier

It is the simplest way of combining predictions from many different simple machine learning models. It gives an average prediction result based on the prediction of all the submodels. The submodels or the basemodels are all of diiferent types.



```python
import pandas as pd
import numpy as np
from sklearn import svm #support vector Machine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
```


```python
diab=pd.read_csv('diabetes.csv')
```


```python
diab.head(3)
```




<div class="table-wrapper" markdown="block">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
outcome=diab['Outcome']
data=diab[diab.columns[:8]]
train,test=train_test_split(diab,test_size=0.25,random_state=0,stratify=diab['Outcome'])# stratify the outcome
train_X=train[train.columns[:8]]
test_X=test[test.columns[:8]]
train_Y=train['Outcome']
test_Y=test['Outcome']
```

Lets create 2 simple models and check the accuracy

### SVM


```python
SVM=svm.SVC(probability=True)
SVM.fit(train_X,train_Y)
prediction=SVM.predict(test_X)
print('Accuracy for SVM kernel is',metrics.accuracy_score(prediction,test_Y))
```

    Accuracy for SVM kernel is 0.651041666667


### Logistic Regression


```python
LR=LogisticRegression()
LR.fit(train_X,train_Y)
prediction=LR.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y))
```

    The accuracy of the Logistic Regression is 0.776041666667


### Voting Classifier


```python
from sklearn.ensemble import VotingClassifier #for Voting Classifier
ensemble_lin_rbf=VotingClassifier(estimators=[('svm', SVM), ('LR', LR)], 
                       voting='soft',weights=[1,2]).fit(train_X,train_Y)
print('The accuracy for Ensembled Model is:',ensemble_lin_rbf.score(test_X,test_Y))

```

    The accuracy for Ensembled Model is: 0.78125


You can see clearly that the accuracy for the Voting Classifier is higher as compared to the simple models.


## Bagging
Bagging is a general ensemble method. It works by applying similar classifiers on small partitions of the dataset and then taking the average of all the predictions. Due to the averaging,there is reduction in variance. Unlike Voting Classifier, Bagging makes use of similar classifiers.

#### Bagged KNN

Bagging works best with models with high variance. An example for this can be Decision Tree or Random Forests. We can use KNN with small value of n_neighbours, as small value of n_neighbours.


```python
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,test_Y))
```

    The accuracy for bagged KNN is: 0.744791666667


## Boosting
Boosting is an ensembling technique which uses sequential learning of classifiers. It is a step by step enhancement of a weak model.Boosting works as follows:

A model is first trained on the complete dataset. Now the model will get some instances right while some wrong. Now in the next iteration, the learner will focus more on the wrongly predicted instances or give more weight to it. Thus it will try to predict the wrong instance correctly. Now this iterative process continous, and new classifers are added to the model until the limit is reached on the accuracy.

### AdaBoost(Adaptive Boosting)
The weak learner or estimator in this case is a Decsion Tree. But we can change the dafault base_estimator to any algorithm of our choice.

Now for AdaBoost, we will directly run the Cross Validation Test, i.e we will run the algorithm on the entire dataset and check the mean accuracy for the ensemble model.


```python
from sklearn.ensemble import AdaBoostClassifier
X=diab[diab.columns[:7]]
Y=diab['Outcome']
ada=AdaBoostClassifier(n_estimators=300,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())
```

    The cross validated score for AdaBoost is: 0.763004101162


Looking at the above results, you might be thinking that Voting Classifier might always give the highest accuracy. But as we have discussed earlier, **not every algorithm is for every problem**. Many other factors like the hyperparameters, class imbalance, etc affect the efficiency of the model. In the above cases, if we make changes in some hyperparameters, it might be possible that AdaBoost or Bagging would give a better results. Thus we must always try out every method available.

#### Further Readings:

**[Ensembling Theory](https://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning/)**

**[Ensembling Theory and Implementation](https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/)**

**[XgBoost](https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/)**


```python

```
We hope this post was helpful. Feel free to comment in case of doubts and do let us know your feedback. Stay tuned for more!
