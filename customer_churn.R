library(ggplot2)
library(reshape2)
library(data.table)
library(caret)
library(e1071)
library(xgboost)
library(Matrix)
library(DiagrammeR)
library(ROCR)
library(ggplot2)

setwd('C:/Users/Matt/Dropbox/DataScienceBootcamp/Projects/customer churn/data')

######################################

# Loading and Basic Data Analysis

######################################

#Load historical data and churn information files for basic inspection

client.info = read.csv("clientinfo.csv")
churn.info = read.csv("accountinfo.csv")

#This is a fairly large dataset.  I use data.table for most munging

client.info.dt = as.data.table(client.info)

#Basic inspection

str(client.info)
sum(is.na(client.info))
dim(client.info)
summary(client.info)

#The historical data represents three months of information in a stacked (long) format.
#We will need the data to be in wide format for modeling purposes.  The easiest
#(though possibly less elegant) way to do this is to subset the data by month,
#then make each variable month-specific by tagging a unique suffix to the the variable
#name so when merged together each variable is different.  I tag June, July,
#and August variables with "_6", "_7", and "_8".  We also no longer need the 
#year and month variables so we remove them.  Monthly user_lifetime figures 
#(duration of account relationship) are unnecessary so I keep only the one from
#the most recent month (August)

#Subset the data

client.info.jun = client.info.dt[month=="6"]
client.info.jul = client.info.dt[month=="7"]
client.info.aug = client.info.dt[month=="8"]

#Tag the variables

names(client.info.jun)[5:length(client.info.jun)] = paste(names(client.info.jun)[5:length(client.info.jun)],"6",sep="_")
names(client.info.jul)[5:length(client.info.jul)] = paste(names(client.info.jul)[5:length(client.info.jul)],"7",sep="_")
names(client.info.aug)[5:length(client.info.aug)] = paste(names(client.info.aug)[5:length(client.info.aug)],"8",sep="_")

#Eliminate unnecessary variables

client.info.jun = client.info.jun[,`:=`(year=NULL,month=NULL,user_lifetime=NULL)]
client.info.jul = client.info.jul[,`:=`(year=NULL,month=NULL,user_lifetime=NULL)]
client.info.aug = client.info.aug[,`:=`(year=NULL,month=NULL)]

#Merge in wide format

client.info.merged = merge(client.info.jun, client.info.jul, by="user_account_id",all=TRUE)
client.info.merged = merge(client.info.merged, client.info.aug, by="user_account_id",all=TRUE)

#Ensure that we have only complete cases for our model.
#Customers that joined/left service during the three months preceding
#will have 'NA' values for particular months. Drop rows with NAs.  
#This leaves us with only customers who have been with the service 
#for the full past three months

client.info.merged = client.info.merged[complete.cases(client.info.merged), ]

#Drop unnecessary columns

client.info.merged <- client.info.merged[,-c(2,63,125)] #customer_intake 6, 7, 8

#Check datatable shape

dim(client.info.merged)

sum(is.na(client.info.merged))

#Some features have negative values.  None should have them.
#Replace all negative values with 0s

client.info.merged[client.info.merged < 0] <- 0


############################

#Examine Target

############################

#Examine data from churn.csv

str(churn.info)

#Drop first two columns (Year and Month - unnecessary)

churn.info = churn.info[,-c(1,2)]

# Visualize Data

cbind(freq=table(churn.info$churn), percentage=prop.table(table(churn.info$churn))*100)

dim(churn.info)

str(churn.info)

head(churn.info, n=5)

#################################

#Merge Target and  Training Data

#################################

#Merge the binary churn indicator to the account data using 
#the shared column "user_account_id"

client.info.merged = merge(client.info.merged,churn.info,by="user_account_id",all.x = TRUE)

#Set row names to "user_account_id" value and delete that data column

rownames(client.info.merged) = client.info.merged[,user_account_id]
client.info.merged[,user_account_id:=NULL]

# Visualize binary classification distribution 

cbind(freq=table(client.info.merged$churn), percentage=prop.table(table(client.info.merged$churn))*100)

client.info.merged = client.info.merged[,-c(1)]

client.info.merged = as.data.frame(client.info.merged)

#######################################

#Set Factor and Continous Variables

#######################################

#Change factor variables to binary numeric columns. Factors 
#converted are all in binary numeric form (0s and 1s), continuous variables
#defined as numeric, factor variables converted to type numeric now converted to type integer

sapply(client.info.merged, class)


factor.vars = c(4:7,64:67,125:128,181)

client.info.merged[factor.vars] = lapply(client.info.merged[factor.vars], as.factor)

as.numeric.factor = function(x) {as.numeric(levels(x))[x]}

for (col in names(client.info.merged)[factor.vars]){ 
  set(client.info.merged, j=col, value=as.numeric.factor(client.info.merged[[col]]))
}
for (col in names(client.info.merged)[factor.vars]){ 
  set(client.info.merged, j=col, value=as.integer(client.info.merged[[col]]))
}
for (col in names(client.info.merged)[-factor.vars]){ 
  set(client.info.merged, j=col, value=as.numeric(client.info.merged[[col]]))
}


#####################################

#Create Traiing and Validation Sets

#####################################


#Training set will be 80% of the data and Validation Set 20% of the data
#Class distribution preserved 

set.seed(0)
in.train <- createDataPartition(client.info.merged$churn, p=0.8, list=FALSE)
summary(factor(client.info.merged$churn))
ytra <- client.info.merged$churn[in.train]; summary(factor(ytra))
ytst <- client.info.merged$churn[-in.train]; summary(factor(ytst))


train = client.info.merged[in.train]
validation = client.info.merged[-in.train]

train = as.data.frame(train)
validation = as.data.frame(validation)

#Create Training and Validation Sets
#Training set will be 80% of the data and Validation Set 20% of the data
#Class distribution preserved in Training and Validation Sets

set.seed(0)
in.train <- createDataPartition(client.info.merged$churn, p=0.8, list=FALSE)
summary(factor(client.info.merged$churn))
y_train <- client.info.merged$churn[in.train]; summary(factor(y_train))
y_validation <- client.info.merged$churn[-in.train]; summary(factor(y_validation))


train = client.info.merged[in.train,]
validation = client.info.merged[-in.train,]

dim(train)
dim(validation)

train = as.data.frame(train)
validation = as.data.frame(validation)

#####################################

#Pickle  Data for Subse3quent Use

#####################################


save(train,file = "train_new.Rda")
save(validation,file = "validation_new.Rda")
#load("train_new.Rda")
#load("validation_new.Rda")




###########################################

#Prepare Data for Model Training

###########################################


#Split Train into factor and continouse varialbe

factor.vars = c(4:7,64:67,125:128,181)

#client.info.merged[factor.vars] = lapply(client.info.merged[factor.vars], as.factor)

train_cat = train[,factor.vars]
train_num = train[,-factor.vars]


#Transform Continuous Variables

#All numerical features display an eponential distribution
#Good form has us transform these distributions using a log
#transformation to convert to more of a normal distribution


train_num[] <- lapply(train_num, as.numeric)

#for(i in 1:length(train_num)) {  
#  train_num[,i] = log(train_num[,i])
#}


train_standardized = cbind(train_num,train_cat)

#xgbtree in caret requires target column to be a factor variable.

train_standardized$churn = ifelse(train_standardized$churn==0,"No","Yes")
train_standardized$churn = as.factor(train_standardized$churn)



###################################

# Model Training

###################################

#I use the Caret package and the xgbtree model

#I perform a grid search using various values of rounds (number of trees), eta, and tree depth

#I use 5 fold cross validation

# Note:  Model was trained on an AWS instance using 8G CPU and 30 gig RAM.  Run time
# was approximately one hour.  

params <-  expand.grid(nrounds=c(100,500,1000),
                       eta = c(0.01,0.05,0.1),
                       max_depth = c(3,7,10),
                       gamma = 0,
                       colsample_bytree =.3,
                       min_child_weight =1,
                       subsample=0)


clf.train = train(churn~.,
                  data=train_standardized,
                  method = "xgbTree",
                  preProcess = NULL,
                  metric = "Accuracy",
                  #tuneGrid = params,
                  maximize=TRUE,
                  na.action = na.pass,
                  verbose=TRUE,
                  #tuneLength = 1,
                  trControl = trainControl( method = "cv",
                                            number = 5,
                                            classProbs = TRUE, 
                                            savePredictions = TRUE
                                            ))

clf.train

save(clf.train, file='xgb_new_cv.model')
#load('xgb_new_cv.model')

#Function Used Below to nicely format confusion matrix

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Dropped', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Kept', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Dropped', cex=1.2, srt=90)
  text(140, 335, 'Kept', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  


#Calculate AUC for Training Set

preds.train = predict(clf.train,type="prob")
train.actual = train_standardized$churn

pred = prediction(preds.train[2],train.actual)
performance(pred, 'auc')@y.values[[1]]

############################

# Prepare Validation Set

############################

#Running the Model on unseen data give us the best
#approximation of performance in a production environment.

#Accordingly, We preprocess the validation set independent
#variables and make predictions.

#We then assess performance.


factor.vars = c(4:7,64:67,125:128,181)

validation_cat = validation[,factor.vars]
validation_num = validation[,-factor.vars]
validation_num[] <- lapply(validation_num, as.numeric)

validation_standardized = cbind(validation_num,validation_cat)

# Make Preidictions

set.seed(0)
preds.validation = predict(clf.train, newdata= validation_standardized,type="prob")

#Assess accuracy

#check accuracy of model on test set with threhold of .5

validation.actual = validation_standardized$churn

acc <- mean(as.numeric(preds.validation[2] >.5) == validation.actual)
print(paste("Accuracy=", acc))

#Find the Cutoff at which we have maximum accuracy

pred_val = prediction(preds.validation[2],validation.actual)
performance(pred_val, 'auc')@y.values[[1]]

ind = which.max( slot(acc.perf, "y.values")[[1]] )
acc = slot(acc.perf, "y.values")[[1]][ind]
cutoff = slot(acc.perf, "x.values")[[1]][ind]
print(c(accuracy= acc, cutoff = cutoff))

acc.perf = performance(pred_val, measure = "acc")
par(mar=c(5,6,4,2)+0.1)
plot(acc.perf)
points(cutoff,acc,pch=01)
points(0.1203554,.841,pch=01)
text(0.6,0.8,"Maximum Accuracy=.907")
text(0.2,0.7,"Optimal Accuracy=.841")
title(main="Accuracy")

#Confusion Matrix for Max Accuracy

prediction = ifelse(preds.validation[2]>0.4817477 ,1,0)
ct = table(prediction,validation.actual)[2:1, 2:1]
cm = confusionMatrix(ct)

draw_confusion_matrix(cm)

#Plot the ROC Curve 

par(mar=c(5,6,4,2)+0.1)
plot(perf, avg='threshold', spread.estimate ='stddev',
     colorize=FALSE)
abline(0,1,lty=2)
text(0.3,0.7,"AUC=.92")
title(main = "ROC Curve")



# Determine the optimal cutoff balancing specificity and sensitivty


prediction2 = preds.validation[2]

prediction2 = as.numeric(prediction2)
test.actual = as.numeric(validation.actual)

pred <- prediction(prediction2, validation.actual)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 

#Plot the ROC Curve

par(mar=c(5,6,4,2)+0.1)
plot(perf, avg='threshold', spread.estimate ='stddev',
     colorize=FALSE)
abline(0,1,lty=2)
text(0.3,0.7,"AUC=.91")
title(main = "ROC Curve")

#Plot the ROC Curve identifying the optimal cutoff


opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]],
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}
print(opt.cut(perf, pred))


#Plot the ROC Curve identifying the optimal cutoff

par(mar=c(5,6,4,2)+0.1)
plot(perf, avg='threshold', spread.estimate ='stddev',
     colorize=FALSE,print.cutoffs.at=c(0.1203554,.481),text.adj=c(1.5,0.4))
abline(0,1,lty=2)
text(0.45,0.6,"AUC=.91")
text(0.2,0.5,"Maximum Accuracy")
text(0.3,0.8,"Optimal Accuracy")
title(main = "ROC Curve")

#AUC

performance(pred, 'auc')@y.values[[1]]

#Draw Pretty Confusion Matrix

prediction = ifelse(preds.validation[2]>0.1203554 ,1,0)
ct = table(prediction,validation.actual)[2:1, 2:1]
cm = confusionMatrix(ct)

draw_confusion_matrix(cm)

