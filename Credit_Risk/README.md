CREDIT RISK CLASSIFICATION
++++++++++++++++++++++++++

OVERVIEW:
A logistic regression classifier was trained on historic loan data to assign loans to one
of two categories: healthy (cat = 0) or default (cat = 1). Seven data fields were 
considered to make the classification: loan size, interest rate, borrower income, debt
to income ratio, number of accounts, "derogatory marks" (presumably things like late bill
payment), total debt and loan status).

WORKFLOW:
The work flow of this logistic regression analysis comprised eight steps. 
1. data was read into a pandas dataframe using the command:
	- ' lending_data_df = pd.read_csv("Resources/lending_data.csv") '
2. The data frame was inspected for null values, descriptive statistics, shape, data
types and column header names.
	- 'any_nulls = 'lending_data_df.isnull().values.any()' -> 'print(any_nulls)'
	- 'lending_data_df.describe()'
	- 'lending_data_df.shape'
	- 'lending_data_df.info()'
	- 'lending_data_df.columns'
3. Data was separated in to training and testing datasets
	- ' X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
4. The proportion of training to test data was reported
	- 'train_set_size = len(X_train)'
	- 'data_set_size = lending_data_df.shape[0]'
	- 'print(f"The number of records i data set: {lending_data_df.shape[0]}")'
	- 'fraction_of_data_in_training_set = train_set_size/ data_set_size'
	- 'print(f"Fraction of records in training set: {fraction_of_data_in_training_set}")'
5. A logistic regression model was instantiated using sklearn
	- 'logreg_model = LogisticRegression()'
6. Themodel was trained
	- 'logreg_model.fit(X_train, y_train)'
7. training and test prediction ere generated
	- 'train_predictions = logreg_model.predict(X_train)'
	- 'test_predictions = logreg_model.predict(X_test)'
8. The model's performance was evaluated in two ways
  	8a. with confusion matrices
  		- 'training_confusion_matrix = confusion_matrix(y_train, train_predictions)'
 	 	- 'test_confusion_matrix = confusion_matrix(y_test, test_predictions)'
  	8b. with classification reports
  		- 'training_report = classification_report(y_train, train_predictions)'
  		- 'testing_report = classification_report(y_test, test_predictions)'

RESULTS:

Table 1. Model Performance on Training Data
++++++++++++++++++++++++++++++++++

Loan Status    precision    recall  f1-score   support

           0       1.00      0.99      1.00     56244
           1       0.86      0.94      0.90      1908
           
    accuracy                           0.99     58152

Table 2. Model Performance on Test Data
++++++++++++++++++++++++++++++

Loan Status    precision    recall  f1-score   support

           0       1.00      0.99      1.00     18792
           1       0.85      0.94      0.90       592
           
    accuracy                           0.99     19384
    
Definitions:
	Loan Status: 0 = healthy loan
                 1 = default
	Precision: True Positives/ (True Positives + False Positives)
	Recall: True Positives / (True Positives + False Negatives)
	F1-score: Weighted harmonic mean of precision and recall
	          [ 2 * (Precision * Recall) / (Precision + Recall) ]
	Accuracy: the percentage of total calls correct
	        [  (True Positives + True Negatives) / (
	            ( True Positives + True Negatives + False Positives + false Negatives))]
	            
The tabular data shows that, on training and test data,  the logistic regression 
classifier performed with the following metrics:

							Training Data	Test Data
	+++++++++++++++++++++++++++++++++++++++++++++++++
	- Accuracy				0.99			0.99
	                        -------------------------
	- Precision	(cat 0)		1.00			1.00
				(cat 1)		0.86			0.85
				           	-------------------------
	- F1 score  (cat 0)		1.00			1.00
				(cat 1)		0.90			0.90
							-------------------------
	- Recall	(cat 0)		0.99			0.99
				(cat 1)		0.94			0.94	
	
RECOMMENDATION:
Based on these metrics, we can recommend deployment of this model. Note that the vast 
majority of loans are healthy (~ 98% of total loans), and the model classifies healthy 
loans perfectly. This is desirable;  essentially no profitable, reliable loan applicants 
will be turned away. On the much smalled class of unhealthy loans, the model still 
performs well, correctly identifying 90% of bad loans. 