# Additional Notes: 
# 1. have a categorical variable and a numerical variable, use boxplot
# 2. how to calculate the correlation between two variables: pandas
# import pandas as pd
# df = pd.DataFrame({'A': [1.3 , -1.2	, -0.1	, 0.5 , -0.8 ], 'B': [1.6 , -1.0	, -0.2	, 0.3 , -0.6 ]})
# print(df['A'].corr(df['B']))
# 3. calculate the correlation between numerical variables of a dataframe
# df.corr()
# 4. How can we impute the missing values of a continuous variable : median() and mode() of the variable
# 5. How can we calculate the probability of loan default for test dataset?
#     1 - model.predict_proba(test)[:,1] 
# 6. Ensemble of classifiers may or may not be more accurate than any of its individual model
# 7. Ensemble method works better, if the individual base models have less correlation among predictions
# 8. Benefits of ensemble model: Better performance + Generalized models( More stable model)


# Problem: Predict if a loan will get approved or not.
# make a predictive model with 3 key phases: 
# data exploration : get insights from the data
# data munging: clean and make it better suit statistical modeling
# predictive modeling: run the algorithm

'''
Problem Statement
About Company
Dream Housing Finance company deals in all home loans. They have presence across all urban,
semi urban and rural areas. Customer first apply for home loan after that company validates 
the customer eligibility for loan.

Problem
Company wants to automate the loan eligibility process (real time) based on customer detail 
provided while filling online application form. These details are Gender, Marital Status, 
Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate 
this process, they have given a problem to identify the customers segments, those are eligible 
for loan amount so that they can specifically target these customers. Here they have provided 
a partial data set. 

Evaluation Metrics
accuracy percentage you correctly predict of the loan approval

goal : This project will tell us what steps one should go through to build a robust model.

'''
##################################################################################################################

#  1. open jupyter notebook in terminal and import all libraries

import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
# %matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

# 2. load data and copy data sets to keep original datasets
# train=pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
# test=pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")
train=pd.read_csv("train_file_loan.csv")
test=pd.read_csv("test_file_loan.csv")
train_original=train.copy()
test_original=test.copy()

# 3. look at the structure of train and test datasets , check features and data types
train.columns   # 12 independent varaibles and 1 target varaible
test.columns    # without target varaible (loan status), we will predict it 
train.dtypes
train.shape, test.shape

# target variable type:  categorical variable (check frequency table, percentage distribution and bar plot.)
train['Loan_Status'].value_counts()
train['Loan_Status'].value_counts(normalize=True) # print out proportion 
train['Loan_Status'].value_counts().plot.bar() # to have bar plot

# 4. visualize the data / explore the data  
# univariate analysis
# visualize the categorical features 
# how to draw a figure with 4 pics ? 
plt.figure(1)   # 1 is the number ID, can be used and referred later
plt.subplot(221) # 22 means grid size 2*2 , 1 means put the subplot in the 1 section 
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')

plt.show()

# visualize the ordinal features 
# 1 figure with 3 pics
plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education')

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')

plt.show()


# visualize the Numerical features
# 1 figure with 2 pics 
plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))   # two types of plot for applicantincome 
plt.show()
# check if data is normally distributed .  Algorithms works better if the data is normally distributed.

# combine income and education to plot 
# 1 figure with 2 pics 
train.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("")
plt.Text(0.5,0.98,'')   
plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(df['LoanAmount']);
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))
plt.show()

# Know the data type (ordinal / categorical / numerical)
# categorical : gender, married(yes,no), self_employed(yes,no), credit_history(0.0 , 1.0)
# ordinal : dependent(1,2,3,4), education(gradute,not graduate ), property(rural,urban,semiurban)
# numeral : it can be count : applicant income , loan amount 


# hypotheses: 
# Applicants with high income should have more chances of loan approval.
# Applicants who have repaid their previous debts should have higher chances of loan approval.
# Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high.
# Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval.
# bivariate analysis: explore them again with respect to the target variable.

# categorical independent variables vs target variable 
# this will give us the proportion of approved and unapproved loans.
print(pd.crosstab(train['Gender'],train['Loan_Status']))
Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.xlabel('Gender')
p = plt.ylabel('Percentage')
plt.show()

print(pd.crosstab(train['Married'],train['Loan_Status']))
Married=pd.crosstab(train['Married'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.xlabel('Married')
p = plt.ylabel('Percentage')
plt.show()

print(pd.crosstab(train['Dependents'],train['Loan_Status']))
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Dependents')
p = plt.ylabel('Percentage')
plt.show()

print(pd.crosstab(train['Education'],train['Loan_Status']))
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.xlabel('Education')
p = plt.ylabel('Percentage')
plt.show()

print(pd.crosstab(train['Self_Employed'],train['Loan_Status']))
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.xlabel('Self_Employed')
p = plt.ylabel('Percentage')
plt.show()

print(pd.crosstab(train['Credit_History'],train['Loan_Status']))
Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.xlabel('Credit_History')
p = plt.ylabel('Percentage')
plt.show()

print(pd.crosstab(train['Property_Area'],train['Loan_Status']))
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Property_Area')
p = plt.ylabel('Percentage')
plt.show()

# Numerical Independent Variable vs Target Variable
# ApplicantIncome + CoapplicantIncome vs Target Variable
# income of people for which the loan has been approved vs people with the loan has not been approved.
# ApplicantIncome does not affect the result that much, nor does CoapplicantIncome

# making bins for applicant income variable
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Income_bin']=pd.cut(df['ApplicantIncome'],bins,labels=group)
Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Property_Area')
P = plt.ylabel('Percentage')

# making bins for coapplicant income variable
bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(df['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Property_Area')
P = plt.ylabel('Percentage')


# Making bins for LoanAmount variable
bins=[0,100,200,700]  
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(df['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Property_Area')
P = plt.ylabel('Percentage')

# convert all the categorical variables into numerical variables (replace N with 0 and Y with 1.)
# few models like logistic regression takes only numeric values as input. 

# drop the new variables of bins , replace 3+ with 3 , N with 0 , Y with 1
train = train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin'], axis=1)
train['Dependents'].replace('3+', 3,inplace=True)
test['Dependents'].replace('3+', 3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)

# use heat map to visualize the correlation , darker means more coorelation
matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


# 6. Missing Value 
# list out feature-wise count of missing values.
train.isnull().sum()

# fill in the missing value in both train dataset and test dataset : 
# For numerical variables: imputation using mean or median
# For categorical variables: imputation using mode
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

# replace the missing data of LoanAmount based on self_employed and education
table = train.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
def fage(x):  # Define function to return value of this pivot_table
    return table.loc[x['Self_Employed'],x['Education']] 
train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True) # filling data

train.isnull().sum() # check again 
# similar changes in test data
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test[test['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True) # filling data

# Outlier Treatment
# remove the skewness is by doing the log transformation,it does not affect the smaller 
# values much, but reduces the larger values, so distribution will be similar to normal distribution
train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])


# modeling and feature engineering
# Evaluation Metrics for Classification Problems
# two ways to evaluate a model: plot the results and compare them with the actual values
#                              calculate the distance between the predictions and actual values

# important concept: confusion matrix 
# TP/ FP/ TN /FN
# precision : TP / (TP + FP)  observations labeled as true, how many are actually labeled true.
# recall(sensitivity) : how many true class are labeled correctly
# Specificity: TN/(TN+FP) how many false class are labeled correctly.
# ROC curve
# AUC 

# 7. Model Building
# logistic regression (classification algorithm, predicting binary outcome) ---- 

# drop the unnecessary variable for both train and test data sets 
train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

# sklearn requires target variable in a separate data set, drop it and save it into another dataset
X = train.drop('Loan_Status',1)   # axis = 1    axis = 0 by default
y = train.Loan_Status

# logistic regression only takes numerical variables. dummy variables converts categorical varible into numerical varibles
X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)




####################################################################################
# 8. AUC-ROC curve
# notes : An area of 1 represents a perfect test; an area of .5 represents a worthless test

# validastion to check how robust our model is to unseen data
# common methods for validation
# the validation set approach/Leave one out cross validation (LOOCV)/k-fold cross validation/Stratified k-fold cross validation

# stratified k-fold cross validation: ensure each fold is a good representative of the whole
# avoid bias and variance
# cross validation logistic model with stratified 5 folds and make predictions for test dataset
# split train dataset into train and validation
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.3)

# import LogisticRegression and accuracy_score from sklearn and fit the logistic regression model.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = LogisticRegression(random_state=1)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
pred = model.predict_proba(xvl)[:,1]  # calculate the probability of loan default for test dataset

# The mean validation accuracy: 0.81.     unnecessary step
# Let us visualize the roc curve.
from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(yvl,  pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)   # loc is the legend location number, 4 means lower right
plt.show()

# read in submission file 
submission = pd.read_csv("sample_Submission_Loan.csv")
submission['Loan_Status']=pred_test    # fill predictions in loan_status variable 
submission['Loan_ID']=test_original['Loan_ID'] # fill loan_ICD to the submission file 
submission['Loan_Status'].replace(0, 'N',inplace=True) #convert 0 and 1 to y and n 
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Logistic.csv') #convert to csv file


# 9. Feature Engineering: create 3 new features
# Total Income:Applicant Income and Coapplicant Income , EMI:monthly amount to be paid
# Balance Income: the income left after the EMI has been paid
train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']
train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']
train['Balance Income']=train['Total_Income']-(train['EMI']*1000) # Multiply with 1000 to make the units equal 
test['Balance Income']=test['Total_Income']-(test['EMI']*1000)

# drop the old features: get rid of the high correlation and reduce noises
train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)


# model building after adding 3 new features
# Logistic Regression / Decision Tree / Random Forest / XGBoost
# the following steps are the same as we did before
X = train.drop('Loan_Status',1)
y = train.Loan_Status                # Save target variable in separate dataset


# logistic regression
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = LogisticRegression(random_state=1)    
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
pred=model.predict_proba(xvl)[:,1]
# output: The mean validation accuracy for this model is 0.812
submission['Loan_Status']=pred_test            # filling Loan_Status with predictions
submission['Loan_ID']=test_original['Loan_ID'] # filling Loan_ID with test Loan_ID
submission['Loan_Status'].replace(0, 'N',inplace=True) # replacing 0 and 1 with N and Y
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Log2.csv') # Converting submission file to .csv format


 # decision tree
from sklearn import tree
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = tree.DecisionTreeClassifier(random_state=1)  
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
# output: The mean validation accuracy for this model is 0.69

submission['Loan_Status']=pred_test            # filling Loan_Status with predictions
submission['Loan_ID']=test_original['Loan_ID'] # filling Loan_ID with test Loan_ID
submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Decision Tree.csv')


######################################################################################################
######################################################################################################
# random forest 
from sklearn.ensemble import RandomForestClassifier
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = RandomForestClassifier(random_state=1, max_depth=10)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
# output: The mean validation accuracy for this model is 0.766

######################################################################################################
#  random forest with tuning the hyperparameters
# 10.improve the accuracy by tuning the hyperparameters for this model: gridsearch(select the best hyper parameters)
from sklearn.model_selection import GridSearchCV
paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
grid_search = GridSearchCV(RandomForestClassifier(random_state = 1),paramgrid)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3, random_state=1)
grid_search.fit(x_train,y_train)   # Fit the grid search model

grid_search.best_estimator_     # Estimating the optimized value
# output:  max_depth variable is 3 and for n_estimator is 41
# we will build the model using the optimized values 
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=41)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
# output: The mean validation accuracy for this model is 0.7638 
pred2=model.predict_proba(test)[:,1]

submission['Loan_Status']=pred_test            # filling Loan_Status with predictions
submission['Loan_ID']=test_original['Loan_ID'] # filling Loan_ID with test Loan_ID
submission['Loan_Status'].replace(0, 'N',inplace=True)  # replacing 0 and 1 with N and Y
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Random Forest.csv') # Converting submission file to .csv format

importances=pd.Series(model.feature_importances_, index=X.columns) # see the importance of features
importances.plot(kind='barh', figsize=(12,8))

######################################################################################################
######################################################################################################

# XG boost : only works with numerical variables (max_depth and n_estimator)
import xgboost
from xgboost import XGBClassifier
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = XGBClassifier(n_estimators=50, max_depth=4)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
pred3=model.predict_proba(test)[:,1]
# output: The mean validation accuracy for this model is 0.79

######################################################################################################

# We can train the XGBoost model using grid search to optimize its hyperparameters and improve the accuracy.
from sklearn.model_selection import GridSearchCV
paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
grid_search = GridSearchCV(XGBClassifier(random_state = 1),paramgrid)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3, random_state=1)
grid_search.fit(x_train,y_train)   # Fit the grid search model

grid_search.best_estimator_     # Estimating the optimized value
# output:  max_depth variable is 3 and for n_estimator is 41
# we will build the model using the optimized values 
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = XGBClassifier(random_state=1, max_depth=3, n_estimators=41)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
# output: The mean validation accuracy for this model is 0.7638 
pred2=model.predict_proba(test)[:,1]

submission['Loan_Status']=pred_test            # filling Loan_Status with predictions
submission['Loan_ID']=test_original['Loan_ID'] # filling Loan_ID with test Loan_ID
submission['Loan_Status'].replace(0, 'N',inplace=True)  # replacing 0 and 1 with N and Y
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('XGBoost.csv') # Converting submission file to .csv format

importances=pd.Series(model.feature_importances_, index=X.columns) # see the importance of features
importances.plot(kind='barh', figsize=(12,8))

############################################################################################################################
############################################################################################################################
# future work 
# We can also arrive at the EMI using a better formula which may include interest rates as well.
# We can even try ensemble modeling (combination of different models). 
# We can combine the applicants with 1,2,3 or more dependents and make a new feature as discussed in the EDA part.
# We can also make independent vs independent variable visualizations to discover some more patterns.
