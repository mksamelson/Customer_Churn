
# Telecom_Customer_Churn
=======
Customer Churn Predictive Model on 3 months of real prepaid telecom data.

This repository has an R file in which 3 months of real prepaid telecom data is munged, analyzed, and a predictive model fit to the data.

Historical, churn, and descriptive information is contained in a single excel file.  One page contains various categorical and continuous factors on customer spending, phone use, SMS use, data use, and some other information. A second page contains a binary churn indicator showing which customers left in September 2013.  A third page has definitions of variables.

After munging and doing preliminary analysis of the historical data set, I chose to utilize a boosted tree model to predict churn.  I utilized the xgbtree method in the Caret package which makes use of XgBoost.  Boosted trees are exceptionally powerful and the XgBoost package is known to be particularly powerful.  I used the Caret package to access XgBoost because I find data pre-processing, using cross validation in model training, and performing a grid search to obtain optimal model parameters to be easier to use in Caret than in XgBoost proper.

The best model had an AUC of approximately .92 and an error of approximately .11 using the accuracy metric.

To address the business problem (i.e., identify and contact customers who will leave in a subsequent month based) I found it best to use an ROC cutoff that balanced specificity and sensitivity and resulted in slightly lower accuracy but yielded a preferable mix of true positives and false positives beneficial to addressing the business problem.

I have also provided a mock powerpoint for EDA that describes the customer base and highlights business opportunity based on the historical data.

For ease of use purposes I split the historical customer data page and the customer churn page in the excel file into their own csv files.

Training the model is time and resource intensive.  The model was trained in an AWS instance with 8G RAM and 30G memory.  Run time in that environment was approximately one hour.  

Since training the model may not be practical in terms of time and resources I have provided the analytics output as comments in the code file.  Plots are in their own separate files in the repository.  .

If you have questions please feel free to contact me at:

Matt Samelson
mksamelson@gmail.com
