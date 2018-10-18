
Data Science Project 1 : Loan Prediction( based on project from Analytics Vidhya)

Train_file_loan.csv and test_file_loan.csv are the data source files. ( The data has 615 rows and 13 columns) The png files are for data visualization All the rest csv files are the results files.

Synopsis: This project is to predict if a loan will get approved or not. A typial classification problem. A project I did for practicing data science skills and gaining better understanding of machine learning algorithms and concepts. (python)

Libriries used : pandas, numpy , seaborn and matplotlib

Data types for visualization : (1) categorical : gender, married(yes,no), self_employed(yes,no), credit_history(0.0 , 1.0) (2) ordinal : dependent(1,2,3,4), education(gradute,not graduate ), property(rural,urban,semiurban) (3) numeral : it can be count : applicant income , loan amount

Code Example for visulization the categorical features in a single figure

plt.figure(1) # 1 is the number ID, can be used and referred later

plt.subplot(221) # 22 means grid size 2*2 , 1 means put the subplot in the 1 section train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') plt.subplot(222) train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') plt.subplot(223) train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') plt.subplot(224) train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')

plt.show()

Models used: logistic regression , decision tree , random forest and xgboost

Tune the hyperparameter : gridsearch to find the max_depth, n_estimators values

StratifiedKFold to evaluate the models

Results: logistic regression has the highest accuracy 0.812
