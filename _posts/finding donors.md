---
layout:     post
title:      Supervised Learning&#58; Part II
date:       2018-03-22 01:19:29
summary:    Continue your quest for ML with Classification.
categories: machine learning, supervised learning, classification
---
**Supervised learning** is the **machine learning** task
of **learning** a function that maps an input to an output based on
example input-output pairs. It infers a function from
labeled **training** data consisting of a set of **training** examples.

We will be implementing Supervised learning concepts using a project.

In this project, you will employ several supervised algorithms of your
choice to accurately model individuals' income using data collected from
the 1994 U.S. Census. You will then choose the best candidate algorithm
from preliminary results and further optimize this algorithm to best
model the data. Your goal with this implementation is to construct a
model that accurately predicts whether an individual makes more than
$50,000. This sort of task can arise in a non-profit setting, where
organizations survive on donations. Understanding an individual's income
can help a non-profit better understand how large of a donation to
request, or whether or not they should reach out to begin with. While it
can be difficult to determine an individual's general income bracket
directly from public sources, we can (as we will see) infer this value
from other publically available features.

*\# Import libraries necessary for this project*

**import** **numpy** **as** **np**

**import** **pandas** **as** **pd**

**from** **time** **import** time

**from** **IPython.display** **import** display *\# Allows the use of
display() for DataFrames*

*\# Import supplementary visualization code visuals.py*

**import** **visuals** **as** **vs**

*\# Pretty display for notebooks*

%matplotlib inline

*\# Load the Census dataset*

data = pd.read\_csv("census.csv")

*\# Success - Display the first record*

display(data.head(n=1))<embed src="media/image1.tif" width="642" height="89" />

**Exploring and Understanding data**

Now to explore the data we calculate the following to get familiar with
the trends.

• The total number of records, 'n\_records'

• The number of individuals making more than \\$50,000
annually, 'n\_greater\_50k'.

• The number of individuals making at most \\$50,000
annually, 'n\_at\_most\_50k'.

• The percentage of individuals making more than \\$50,000
annually, 'greater\_percent'.

*\# Total number of records*

n\_records = data\['age'\].count()

*\#Number of records where individual's income is more than $50,000*

n\_greater\_50k = data\[data.income=="&gt;50K"\].income.count()

*\#data\[data\['income'\]=="&gt;50K"\].count()*

*\#Number of records where individual's income is at most $50,000*

n\_at\_most\_50k = data\[data.income=="&lt;=50K"\].income.count()

*\#data\[data\['income'\]=="&lt;=50K"\].count()*

*\#Percentage of individuals whose income is more than $50,000*

greater\_percent = float(n\_greater\_50k)\*100/n\_records

*\# Print the results*

**print** "Total number of records: {}".format(n\_records)

**print** "Individuals making more than $50,000:
{}".format(n\_greater\_50k)

**print** "Individuals making at most $50,000:
{}".format(n\_at\_most\_50k)

**print** "Percentage of individuals making more than $50,000:
{:.2f}%”.format(greater\_percent)

OUTPUT:

Total number of records: 45222

Individuals making more than $50,000: 11208

Individuals making at most $50,000: 34014

Percentage of individuals making more than $50,000: 24.78%

Now, we separate the data into the features that we will use to predict
the target Lakeland the target label itself.

*\# Split the data into features and target label*

income\_raw = data\[‘income'\]

features\_raw = data.drop('income', axis = 1)

A dataset may sometimes contain at least one feature whose values tend
to lie near a single number, but will also have a non-trivial number of
vastly larger or smaller values than that single number. Algorithms can
be sensitive to such distributions of values and can underperform if the
range is not properly normalized. With the census dataset two features
fit this description: 'capital-gain' and ‘capital-loss'.

*\# Visualize skewed continuous features of original data*

vs.distribution(data)

<img src="media/image2.png" width="642" height="306" />

*\# Log-transform the skewed features*

skewed = \['capital-gain', 'capital-loss'\]

features\_raw\[skewed\] = data\[skewed\].apply(**lambda** x: np.log(x +
1))

—-NOTE—— Here we have used x+1 because log of 0 is undefined. If theres
a null value on the x axis, the algorithm will return an error.

*\# Visualize the new log distributions*

vs.distribution(features\_raw, transformed = True)

<img src="media/image3.png" width="642" height="306" />

In addition to performing transformations on features that are highly
skewed, it is often good practice to perform some type of scaling on
numerical features. Applying a scaling to the data does not change the
shape of each feature's distribution (such
as 'capital-gain' or 'capital-loss' above); however, normalization
ensures that each feature is treated equally when applying supervised
learners. Note that once scaling is applied, observing the data in its
raw form will no longer have the same original meaning, as exampled
below.

*\# Import sklearn.preprocessing.StandardScaler*

**from** **sklearn.preprocessing** **import** MinMaxScaler

*\# Initialize a scaler, then apply it to the features*

scaler = MinMaxScaler()

numerical = \['age', 'education-num', 'capital-gain', 'capital-loss',
'hours-per-week'\]

features\_raw\[numerical\] = scaler.fit\_transform(data\[numerical\])

*\# Show an example of a record with scaling applied*

display(features\_raw.head(n = 1))

<embed src="media/image4.tif" width="642" height="89" />

We can see there are several features for each record that are
non-numeric. Typically, learning algorithms expect input to be numeric,
which requires that non-numeric features (called *categorical
variables*) be converted. One popular way to convert categorical
variables is by using the **one-hot encoding** scheme. One-hot encoding
creates a *"dummy"*variable for each possible category of each
non-numeric feature. For example, assume someFeature has three possible
entries: A, B, or C. We then encode this feature
into someFeature\_A, someFeature\_B and someFeature\_C.

Additionally, as with the non-numeric features, we need to convert the
non-numeric target label, 'income' to numerical values for the learning
algorithm to work. Since there are only two possible categories for this
label ("&lt;=50K" and "&gt;50K"), we can avoid using one-hot encoding
and simply encode these two categories as 0 and 1, respectively. In code
cell below, you will need to implement the following:

•
Use [*pandas.get\_dummies()*](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) to
perform one-hot encoding on the 'features\_raw' data.

• Convert the target label 'income\_raw' to numerical entries.

▪ Set records with "&lt;=50K" to 0 and records with "&gt;50K" to 1.

*\# One-hot encode the 'features\_raw' data using pandas.get\_dummies()*

features = pd.get\_dummies(features\_raw)

*\# Encode the 'income\_raw' data to numerical values*

income = income\_raw.apply(**lambda** x: 1 **if** x == "&gt;50K"
**else** 0)

*\# Print the number of features after one-hot encoding*

encoded = list(features.columns)

**print** "{} total features after one-hot
encoding.".format(len(encoded))

OUTPUT:

103 total features after one-hot encoding.

Now all *categorical variables* have been converted into numerical
features, and all numerical features have been normalized. As always, we
will now split the data (both features and their labels) into training
and test sets. 80% of the data will be used for training and 20% for
testing.

*\# Import train\_test\_split*

**from** **sklearn.cross\_validation** **import** train\_test\_split

*\# Split the 'features' and 'income' data into training and testing
sets*

X\_train, X\_test, y\_train, y\_test = train\_test\_split(features,
income, test\_size = 0.2, random\_state = 0)

*\# Show the results of the split*

**print** "Training set has {} samples.".format(X\_train.shape\[0\])

**print** "Testing set has {} samples.”.format(X\_test.shape\[0\])

OUTPUT:

Training set has 36177 samples.

Testing set has 9045 samples.

*CharityML*, equipped with their research, knows individuals that make
more than $50,000 are most likely to donate to their charity. Because of
this, \*CharityML\* is particularly interested in predicting who makes
more than $50,000 accurately. It would seem that using **accuracy** as a
metric for evaluating a particular model's performace would be
appropriate. Additionally, identifying someone that *does not* make more
than $50,000 as someone who does would be detrimental to \*CharityML\*,
since they are looking to find individuals willing to donate. Therefore,
a model's ability to precisely predict those that make more than $50,000
is *more important*than the model's ability to **recall** those
individuals. We can use **F-beta score** as a metric that considers both
precision and recall:

<embed src="media/image5.tif" width="389" height="160" />

In particular, when , more emphasis is placed on precision. This is
called the &lt;strong&gt;F score (or F-score for simplicity).

Looking at the distribution of classes (those who make at most $50,000,
and those who make more), it's clear most individuals do not make more
than $50,000. This can greatly affect **accuracy**, since we could
simply say *"this person does not make more than $50,000"* and generally
be right, without ever looking at the data! Making such a statement
would be called **naive**, since we have not considered any information
to substantiate the claim. It is always important to consider the *naive
prediction* for your data, to help establish a benchmark for whether a
model is performing well.

That been said, using that prediction would be pointless: If we
predicted all people made less than $50,000, *CharityML* would identify
no one as donors.

*\# Calculate accuracy*

**from** **sklearn.metrics** **import** accuracy\_score

**from** **sklearn.metrics** **import** recall\_score

**from** **sklearn.metrics** **import** fbeta\_score

income\_pred=income.apply(**lambda** x:1)

TP=sum(map(**lambda** x,y: 1 **if** x==1 **and** y==1 **else** 0,
income,income\_pred)) *\#True Pos*

FP=sum(map(**lambda** x,y: 1 **if** x==0 **and** y==1 **else** 0,
income,income\_pred)) *\#False Pos*

FN=sum(map(**lambda** x,y: 1 **if** x==1 **and** y==0 **else** 0,
income,income\_pred)) *\#False Neg*

*\# accuracy = TP/(TP+FP)*

accuracy = float(TP)/(TP+FP)

*\# recall = TP/(TP+FN)*

recall=float(TP)/(TP+FN)

*\# The commented code below was used to confirm the recall calculation
was correct*

*\#recal1=recall\_score(income,income\_pred)*

*\#print 'recall comparison',recal1,recall1*

*\# Calculate F-score using the formula above for beta = 0.5*

beta=0.5

fscore = (1+beta\*\*2)\*(accuracy\*recall)/(beta\*\*2\*accuracy+recall)

*\#fscore1=fbeta\_score(income,income\_pred, beta=0.5)*

*\#print 'fscore comparison',fscore,fscore1*

*\# Print the results *

**print** "Naive Predictor: \[Accuracy score: {:.4f}, F-score:
{:.4f}\]".format(accuracy, fscore)

*\# Import two metrics from sklearn - fbeta\_score and accuracy\_score*







### NAIVE BAYES CLASSIFIER
Bayes theorem is mainly used for finding conditional probability i.e. Probability of X provided Y has occurred. For example, if cancer is related to age, then, using Bayes’ theorem, a person’s age can be used to more accurately assess the probability that they have cancer, compared to the assessment of the probability of cancer made without knowledge of the person's age. The formula is stated as: Prob(B given A) = Prob(A and B)/Prob(A).
Naive bayes algorithm uses this theorem for creating a classifier. It is called “Naive” because many assumptions are made which may or may not be true in context with the data.
The working of this classifier is illustrated by the following example:
![alt text](file:///Users/dhvanikansara/Desktop/ml_blog/quest.png)
Find whether a youth student with a medium income and fair credit rating will buy a computer or not.		 	 	 		
Solution:
For: X = (age = youth, income = medium, student = yes, credit rating = fair)		
P(buys computer = yes) = 9/14 = 0.643<br>
P(buys computer = no) = 5/14 = 0.357

P(age = youth | buys computer = yes) = 2/9 = 0.222 <br>
P(age = youth | buys computer = no) = 3/5 = 0.600

P(income = medium | buys computer = yes) = 4/9 = 0.444 <br>
P(income = medium | buys computer = no) = 2/5 = 0.400 <br>

P(student = yes | buys computer = yes) = 6/9 = 0.667<br>
P(student = yes | buys computer = no) = 1/5 = 0.200

P(credit rating = fair | buys computer = yes) = 6/9 = 0.667 <br>
P(credit rating = fair | buys computer = no) = 2/5 = 0.400

In this theorem, it is assumed that the value of a particular feature is independent of the value of any other feature, given the class variable (This assumption is not true in many cases hence, “Naive”).
To find total probability in case of independent events, we multiply the constituent probabilities to get the resultant probability.

So, using above probabilities, we obtain

P(X | buys computer = yes)
= P(age = youth | buys computer = yes) × P(income = medium | buys computer = yes) × P(student = yes | buys computer = yes)× P(credit rating = fair | buys computer = yes) <br>
= 0.222 × 0.444 × 0.667 × 0.667 = 0.044.		

Similarly,
P(X|buys computer = no)<br>
=P(age = youth | buys computer = no) × P(income = medium | buys computer = no) × P(student = yes | buys computer = no)× P(credit rating = fair | buys computer = no) <br>
= 0.600 × 0.400 × 0.200 × 0.400 = 0.019.

Then we compute,<br>
P(X | buys computer = yes) x P(buys computer = yes) = 0.044 × 0.643 = 0.028 <br>
P(X | buys computer = no) x P(buys computer = no) = 0.019 × 0.357 = 0.007

Now we select the one which has a greater probability compared to other, in the above case as **P(X| yes)> P(X| no)**
So, we can say that a youth student with a medium income and fair credit rating **will** buy a computer.

The classifier performance is based factors explained below:

For this we need to know the following definitions:

True positives (TP): These refer to the positive tuples that were correctly labeled by the classifier. Let TP be the number of true positives.

True negatives (TN ): These are the negative tuples that were correctly labeled by the classifier. Let TN be the number of true negatives.

False positives (FP): These are the negative tuples that were incorrectly labeled as positive (e.g., tuples of class buys computer = no for which the classifier predicted buys computer = yes). Let FP be the number of false positives.

False negatives (FN): These are the positive tuples that were mislabeled as neg- ative (e.g., tuples of class buys computer = yes for which the classifier predicted buys computer = no). Let FN be the number of false negatives.

<img src="file:///Users/dhvanikansara/Desktop/ml_blog/table.png" style="width: 300px;"/>

*Accuracy*: The accuracy of a classifier on a given test set is the percentage of test set tuples that are correctly classified by the classifier.

 <img src="file:///Users/dhvanikansara/Desktop/ml_blog/accuracy.png" style="width: 150px;"/>

*Recall*: It is a measure of completeness i.e. The proportion of positive tuples that are correctly identified.

 <img src="file:///Users/dhvanikansara/Desktop/ml_blog/recall.png" style="width: 150px;"/>

*F-score*: Takes into consideration both precision and recall, called as harmonic mean

 <img src="file:///Users/dhvanikansara/Desktop/ml_blog/fscore.png" style="width: 200px;"/>
beta is any non negative real number
###Sklearn Code:

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score

income_pred=income.apply(lambda x:1)

TP=sum(map(lambda x,y: 1 if x==1 and y==1 else 0, income,income_pred)) #True Pos
FP=sum(map(lambda x,y: 1 if x==0 and y==1 else 0, income,income_pred)) #False Pos
FN=sum(map(lambda x,y: 1 if x==1 and y==0 else 0, income,income_pred)) #False Neg

# accuracy = TP/(TP+FP)
accuracy = float(TP)/(TP+FP)

# The commented code below was used to confirm the precision calculation was correct
#accuracy1 = accuracy_score(income,income_pred)
#print 'accuracy comparison',accuracy,accuracy1

# recall = TP/(TP+FN)
recall=float(TP)/(TP+FN)

# The commented code below was used to confirm the recall calculation was correct
#recal1=recall_score(income,income_pred)
#print 'recall comparison',recal1,recall1

# TODO: Calculate F-score using the formula above for beta = 0.5
beta=0.5
fscore = (1+beta**2)*(accuracy*recall)/(beta**2*accuracy+recall)

#fscore1=fbeta_score(income,income_pred, beta=0.5)
#print 'fscore comparison',fscore,fscore1

# Print the results
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)
```

Output:Naive Predictor: [Accuracy score: 0.2478, F-score: 0.2917]

Advantages of naive bayes classifier:<br>

1. Very simple, easy to implement and fast.<br>
2. If the NB conditional independence assumption holds, then it will converge quicker than discriminative models like logistic regression.<br>
3. Even if the NB assumption doesn’t hold, it works great in practice.<br>
4. Need less training data.<br>
5. Highly scalable. <br>
6. It scales linearly with the number of predictors and data points. <br>
7. Can be used for both binary and multi-class classification problems.<br>
8. Handles continuous and discrete data.
9. Not sensitive to irrelevant features.

Disadvantages of naive bayes classifier:

1. It a very strong assumption on the shape of your data distribution, i.e. any two features are independent given the output class. Due to this, the result can be (potentially) very bad. This is not as terrible as people generally think, because the NB classifier can be optimal even if the assumption is violated and its results can be good even in the case of sub-optimality.

2. If a test tuple is not present in the given test dataset, we cannot estimate the required probability, for eg. in the above case if credit rating is “satisfactory”, we do not have any tuple corresponding to that so we cannot estimate.


3. Suppose that for the class buys computer = yes in some training database, D, containing 1000 tuples, we have 0 tuples with income = low, 990 tuples with income = medium, and 10 tuples with income = high. The probabilities of these events, are 0, 0.990 (from 990/1000), and 0.010 (from 10/1000), respectively. Since we have one of them as 0 so during multiplying the effect of all the other probilites will not be seen as it is nullified when multiplied by 0. So we cannot reach a right conclusion.


4. A third problem arises for continuous features. It is common to use a binning procedure to make them discrete, but if you are not careful you can throw away a lot of information. Another possibility is to use Gaussian distributions for the likelihoods.

Naive Bayes classifiers mostly used in text classification (due to better result in independence rule) have higher success rate as compared to other algorithms. As a result, it is widely used in Spam filtering (identify spam e-mail) and Sentiment Analysis (in social media analysis, to identify positive and negative customer sentiments).
It is also used in recommendation system, to find if a given resource will be liked by a user or not based on his past choices.

[read more about Naive Bayes here] (https://docs.oracle.com/cd/B28359_01/datamine.111/b28129/algo_nb.htm#i1005770)
