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
![image.png](attachment:image.png)
where **x** is an original value, **x'** is the normalized value. For example, suppose that we have the students' weight data, and the students' weights span [160 pounds, 200 pounds]. To rescale this data, we first subtract 160 from each student's weight and divide the result by 40 (the difference between the maximum and minimum weights).

**Mean normalization** is used to make features have approximate zero mean.
![image.png](attachment:image.png)

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


![image.png](attachment:image.png)
<center>**Underfitting and Overfitting in Linear Regression**</center>
 
![image.png](attachment:image.png)
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


```python

```
