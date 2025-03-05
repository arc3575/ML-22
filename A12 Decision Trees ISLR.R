#Chapter 8 – Exercises 8.4 – 1, 6, 8 (a-e), 9, 10, 11


# Question 8.1

#Picture of drawing

#Question 8.6

#Provide a detailed explanation of the algorithm that is used to ft a
#regression tree.

#Regression tree algorithm works two major steps
#First step the predictor space, the set of values X1-Xpred is divided into J regions
#This can be seen visually in question #1, in 2 D space. 
#These rectangular regions are exclusive meaning, results can only be in one region
#Also every value must be in at least one
#The second step is that for any value that falls into any region Rj
#The algorithm makes the same prediction, which is the mean response values
#For the training data
#The feature space is divided into J regions based on "top-down greedy" or "recursive binary splitting"
#The algorithm self-optimizes by minimizing residual square error or RSS
#It's able to choose the predictor to split at any given branch based on 
#the lowest RSS

#In the lab, a classification tree was applied to the Carseats data set 
#after converting Sales into a qualitative response variable. 
#Now we will seek to predict Sales using regression trees and related approaches,
#treating the response as a quantitative variable.

# Question 8.8.a
#Split the carseats dataset into train and test data

#First import some libraries
#Gonna need the tree function for regression trees, so install package
library(ISLR)
library(MASS)
install.packages('tree')
install.packages('randomForest')
install.packages('survival')
install.packages('lattice')
install.packages('splines')
install.packages('parallel')
install.packages('gbm')
library(gbm)
library(tree)
attach(Carseats)

#New variable set equal to Carseats

Carseats_df <- Carseats
summary(Carseats_df$Sales)

#Set seed to 1, to make sure reproducibility 
set.seed(1)

#Train variable utilizing sample function

train <- sample(1:nrow(Carseats_df), nrow(Carseats_df)/2) 
Carseats_train <- Carseats[train, ]
Carseats_test <- Carseats[-train,]
#test <- Carseats_df[-train, "Sales"]

# Question 8.8.b

#Use the tree function to create a Carseats tree model
#Establish response variable "Sales", data is Carseats, but subset is train

Carseats_treemodel <- tree(Sales~., data = Carseats_df, subset = train)
Carseats_treemodel <- tree(Sales~., data = Carseats_train)
#Check Carseats tree model Make sure it makes sense
summary(Carseats_treemodel)


#After seeing summary show tree using plot

plot(Carseats_treemodel)

#Add text to the regression tree

text(Carseats_treemodel, pretty = 0, cex = 0.5)

#Make variable "yhat" (predicted values) utilize predict() function

yhat <- predict(Carseats_treemodel,newdata = Carseats_test)

#Calculate mean squared error

#first calculate error

Carseats_treemodel_error <- yhat - Carseats_test$Sales

#Take the mean of the error squared

Carseats_treemodel_MSE <- mean(Carseats_treemodel_error^2)

#Check

Carseats_treemodel_MSE

#MSE is 4.922039

# Question 8.8.c

#Use cross-validation in order to determine the optimal level of tree complexity. 
#Does pruning the tree improve the test MSE?

#Use cv.tree function, cross validation 
cross_validation_carseats <- cv.tree(Carseats_treemodel)

#plot size and dev
plot(cross_validation_carseats$size, cross_validation_carseats$dev, type ='b')

#According to the plot, 16 had the fewest amounts of misclassifications

#Set best equal to 16
prune_Carseats <- prune.tree(Carseats_treemodel, best = 16)

#plot New prune model
plot(prune_Carseats)

#add text
text(prune_Carseats, pretty = 0)


#yhat prune to calculate MSE
yhat_prune <- predict(prune_Carseats, Carseats_test)
mean((yhat_prune - Carseats_test$Sales)^2)

#MSE is 4.903443, slightly more improved than the regression tree

# Question 8.8.d

#Use the bagging approach in order to analyze this data. 
#What test MSE do you obtain? 
#Use the importance() function to determine which variables are most important.

#import random forest library

library(randomForest)
set.seed(1)
bag_car = randomForest(Sales~.,data=Carseats_train,mtry = 10, importance = TRUE)

yhat_bag <- predict(bag_car, newdata = Carseats_test)

mean((yhat_bag - Carseats_test$Sales)^2)

#MSE is 2.605253
#Much lower than the regression tree and the 

#Importance function with bag_car model
importance(bag_car)

#Plot bag_Car

varImpPlot(bag_car)

#The most important variables are Price and ShelveLoc

#According to the dataset description; price is the price that a company charges for car seats at any given site
#ShelveLoc is the quality of the shelving location for car seats at each site

#The bagged regression tree had an MSE of 2.605, which was almsot half that of the optimally pruned regression tree


#Question 8.8.e

# Use random forests to analyze this data. What test MSE do you obtain? 
#Use the importance() function to determine which variables are most important. 
#Describe the efect of m, the number of variables considered at each split, 
#on the error rate obtained.


set.seed(1)

#M is the amount of input variables used in the sampling, so there cannot be more than 10
#since there are only 10 variables
#running M = 1 thru M = 10, you can get an idea of how MSE changes
rf_car = randomForest(Sales~.,data=Carseats_test,mtry = 10, importance = TRUE)
yhat_rf = predict(rf_car,newdata=Carseats_test)
mean((yhat_rf-Carseats_test$Sales)^2)

# M = 1, MSE = 2.08814
# M = 2, MSE = 0.7860879
# M = 3, MSE = 0.5833007
# M = 4, MSE = 0.5062183
# M = 5, MSE = 0.4790012
# M = 6, MSE = 0.4780848
# M = 7, MSE = 0.4520129
# M = 8, MSE = 0.4379401
# M = 9, MSE = 0.4389787
# M = 10, MSE = 0.4324537

# M = 10 had the smallest MSE of 0.4324537, which is extremely small. All, M = 5:10 all had very similar MSE
str(Carseats_test)
importance(rf_car)

varImpPlot(rf_car)

#Price and Shelve Loc are still very important, but ShelveLoc is slightly more important than Price
#The opposite was true in the bagging model

#Question 8.9.a

#This problem involves the OJ data set which is part of the ISLR package.
#Create a training set containing a random sample of 800 observations, 
#and a test set containing the remaining observations.

library(ISLR)
#set seed(1) for reproducibility 
set.seed(1)

#Assign new df to OJ because I like creating new variables
OJ_df <- OJ

#Look at OJ_df structure
str(OJ_df)

#1070 total observations so test will be 270 observations and train will be 800
#18 variables; two factor, 16 num


#Split data up into training and testing
#Dimension training data based on criteria
OJ_t <- sample(dim(OJ_df)[1],800)
OJ_train <- OJ_df[OJ_t,]
OJ_test <- OJ_df[-OJ_t,]

# Question 8.9.b
str(OJ_test)
#Fit a tree to the training data, with Purchase as the response and the other variables 
#as predictors. 
#Use the summary() function to produce summary statistics about the tree, 
#and describe the results obtained. 
#What is the training error rate? 
#How many terminal nodes does the tree have?

#Purchase is factor variable
#Set as response variables
#Other 17 variables are predictors

#Create tree model
OJ_tree_model <- tree(Purchase~., data= OJ_train)

str(OJ_train)
#Use summary to look at tree model stats

summary(OJ_tree_model)
#Residual mean deviance of 0.7391, which is the amount of error in the model after the tree is constructed

#There are 8 terminal nodes, so 8 possible endings in the trees

#Misclassification error rate of 15.8%, 
#these are the amount of training observations that were predicted to fall in the wrong class

#Question 8.9.c

#Type in the name of the tree object in order to get a detailed text output. 
#Pick one of the terminal nodes, 
#and interpret the information displayed.

OJ_tree_model

#I picked the node labelled 9). This is a terminal node as indicated by an asterisk 
# The split criterion is Loyal > 0.0356, the number of observations is 109 in that branch
# There's a deviance of 100.90
# About 17% of the observations in that branch take the value of CH, the remaining 83% take the value of MM

#Question 8.9.d
#Create a plot of the tree, and interpret the results.

plot(OJ_tree_model)
text(OJ_tree_model, pretty = TRUE)

#The most important indicator of the response variable "purchase" seems to be "LoyalCH"
#The first branch represents the intensity of customer brand loyalty to CH. 
#The first three branches correspond to the predictor varaible "LoyalCH"
#The other predictor variables are "Price Diff", "SpecialCH" and "ListPriceDiff"

#Question 8.9.e

#Predict the response on the test data, 
#and produce a confusion matrix 
#comparing the test labels to the predicted test labels. 
#What is the test error rate?


OJ_samp <- sample(dim(OJ_df)[1],800)
OJ_train <- OJ_df[OJ_samp,]
OJ_test <- OJ_df[-OJ_samp,]

tree_prediction <- predict(OJ_tree_model, newdata = OJ_test, type = "class")
table(tree_prediction, OJ_test$Purchase)

#According to the table, the errors are 18 and 32
#So the error rate is

OJ_tree_error <- 18 + 32
OJ_tree_error_rate <- OJ_tree_error/270
OJ_tree_error_rate

#The error rate is 18.5% 

#Question 8.9.f

#Apply the cv.tree() function to the training set in order to determine the optimal tree size.

cv_OJ = cv.tree(OJ_tree_model, FUN = prune.misclass)
cv_OJ

#Question 8.9.g

#Produce a plot with tree size on the x-axis 
#and cross-validated classification error rate on the y-axis.

plot(cv_OJ$size,cv_OJ$dev,type='b', xlab = "Tree size", ylab = "Deviance")

#Question 8.9.h
#Which tree size corresponds to the lowest cross-validated classification error rate?
#We might see that the 5-node tree is the smallest tree 
#with the lowest classification error rate.

#We might see that the tree size with the smallest classification error has 6 nodes

#Question 8.9.i

#Produce a pruned tree corresponding to the optimal tree size 
#obtained using cross-validation. 
#If cross-validation does not lead to selection of a pruned tree, 
#then create a pruned tree with five terminal nodes.

prune_OJ = prune.misclass(OJ_tree_model, best=6)
plot(prune_OJ)
text(prune_OJ,pretty=0)

#Question 8.9.j

#Compare the training error rates between the pruned and unpruned trees. Which is higher?

pruned_tree_pred = predict(prune_OJ, newdata = OJ_test, type = "class")
table(pruned_tree_pred,OJ_test$Purchase)

#The unpruned and pruned tree both have an error rate of about 18%

#Question 8.10.a

#10. We now use boosting to predict Salary in the Hitters data set.
#Remove the observations for whom the salary information is unknown, 
#and then log-transform the salaries.

Hitters_df <- Hitters
Hitters_df = na.omit(Hitters)
Hitters_df$Salary = log(Hitters_df$Salary)

str(Hitters_df)

#Question 8.10.b

#Create a training set consisting of the first 200 observations, 
#and a test set consisting of the remaining observations.

#Split data into train and test data
Htrain = 1:200
Hitters_train = Hitters_df[Htrain,]
Hitters_test = Hitters_df[-Htrain,]
Htrain

#confirm structure of all three datasets before moving forward
str(Hitters_df)
str(Hitters_train)
str(Hitters_test)

#Question 8.10.c

#Perform boosting on the training set with 1,000 trees 
#for a range of values of the shrinkage parameter λ. 
#Produce a plot with different shrinkage values on the x-axis 
#and the corresponding training set MSE on the y-axis.

#set seed for reproducibility 
set.seed(1)

pows <- seq(-10, -0.2, by = 0.1)
lambdas <- 10^pows
train_err <- rep(NA, length(lambdas))
for (i in 1:length(lambdas)) {
  boost_hitters <- gbm(Salary ~ ., data = Hitters_train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[i])
  boost_pred_train <- predict(boost_hitters, Hitters_train, n.trees = 1000)
  train_err[i] = mean((boost_pred_train - Hitters_train$Salary)^2)
}
plot(lambdas, train_err, type = "b", xlab = "Shrinkage values", ylab = "Training MSE")


#Question 8.10.d
#Produce a plot with different shrinkage values on the x-axis 
#and the corresponding test set MSE on the y-axis.

set.seed(1)

#Create variable for test_err
test_err <- rep(NA, length(lambdas))

#Loop that iterates through gbm model and adds output to yhat_hit & test_err
for (i in 1:length(lambdas)) {
  boost_hitters = gbm(Salary ~ ., data = Hitters_train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[i])
  yhat_hit = predict(boost_hitters, Hitters_test, n.trees = 1000)
  test_err[i] = mean((yhat_hit - Hitters_test$Salary)^2)
}
str(test_err)
plot(lambdas, test_err, type = "b", xlab = "Shrinkage values", ylab = "Test MSE")

min(test_err)

lambdas[which.min(test_err)]

#The minimum test error is 0.25 and according to the loop
#This was achieved with a lambda of 0.079

#Question 8.10.e

#Compare the test MSE of boosting to the test MSE 
#that results from applying two of the regression approaches 
#seen in Chapters 3 and 6.

#Going to use "glmnet"
install.packages('glmnet')
library(glmnet)

#First create linear model for Hitters dataset
#Use only training data for linear model
Hitters_Linear_Model <- lm(Salary ~ ., data = Hitters_train)

#Now use testing data, to get predictions
Pred_LM = predict(Hitters_Linear_Model, Hitters_test)

#MSE for linear model
mean((Pred_LM - Hitters_test$Salary)^2)

#MSE is 0.4917

#Generalized linear model
x = model.matrix(Salary ~ ., data = Hitters_train)
x_test = model.matrix(Salary ~ ., data = Hitters_test)
y = Hitters_train$Salary
Hitters_glm <- glmnet(x, y, alpha = 0)
Pred_GLM = predict(Hitters_glm, s = 0.01, newx = x_test)

#MSE is 0.4570
mean((Pred_GLM - Hitters_test$Salary)^2)

#Generalized Boosting Model had a lower MSE than both linear model types
#Standard linear model and generalized linear model

#Question 8.10.f

#Which variables appear to be the most important predictors 
#in the boosted model?

boost_hitters2 <- gbm(Salary ~ ., data = Hitters_train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[which.min(test_err)])
summary(boost_hitters2)

#According to the relative influence, CAtBat has the highest by a large margin
# CRBI is the second most influential, followed by Walks and PutOuts
#For some perspective, CatBat is almost 3 times as influential as Walks and PutOuts

#Question 8.10.g

#Now apply bagging to the training set. 
#What is the test set MSE for this approach?

set.seed(1)

#Bagging approach using Random Forest function
bag_hitters <- randomForest(Salary ~ ., data = Hitters_train, mtry = 19, ntree = 500)

# y-hat variable for bagging predictions
yhat_bag <- predict(bag_hitters, newdata = Hitters_test)

#MSE for bagging
mean((yhat_bag - Hitters_test$Salary)^2)

#MSE is 0.2299

#Bagging had the lowest MSE out of the models performed in this question
#The MSE for Boosting was approximately 0.25, whereas the MSE for bagging was 0.2299
#Both boosting and bagging had lower MSE than the MSEs for the linear models

#Question 8.11.a

#11. This question uses the Caravan data set.
#(a) Create a training set consisting of the frst 1,000 observations,
#and a test set consisting of the remaining observations.

#Split the data into testing and training
Caravan_df <- Caravan

#Index 1 thru 1000 observations
CVantrain = 1:1000

#Change purchase to dummy variable 0 & 1

Caravan_df$Purchase = ifelse(Caravan_df$Purchase == "Yes", 1, 0)

Caravan_train = Caravan_df[CVantrain,]
Caravan_test = Caravan_df[-CVantrain,]


#Look at the three data structures to make sure we're all good

str(Caravan_df)

#Caravan has 5822 observations, so we're only training the first 1000

str(Caravan_train)

#Caravan train has 1000 observations

str(Caravan_test)


#Caravan test has the remaining 4822
#Way more observations in test data

#Question 8.11.b

#Fit a boosting model to the training set with Purchase as the response 
#and the other variables as predictors. 
#Use 1,000 trees, and a shrinkage value of 0.01. 
#Which predictors appear to be the most important?

#First set seed for reproducibility
set.seed(1)

#Create boost model using generalized boosting model function
#1000 treees and shrinkage value of 0.01

boost_caravan = gbm(Purchase ~ ., data = Caravan_train, distribution = "gaussian", n.trees = 1000, shrinkage = 0.01)

#PVRAAUT and AVRAAUT have no variation, so probably not useful for model

#look at summary of boost caravan

summary(boost_caravan)

#According to the boost summary "PPERSAUT" and "MKOOPKLA" are the most influential variables
#A number of variables have no influence or importance at all

#Question 8.11.c

#Use the boosting model to predict the response on the test data. 
#Predict that a person will make a purchase if the estimated prob- ability of purchase 
#is greater than 20 %. Form a confusion matrix. 
#What fraction of the people predicted to make a purchase do in fact make one? 
#How does this compare with the results obtained from applying 
#KNN or logistic regression to this data set?

#First use boost caravan model to predict response on test data

probs_boost_test <- predict(boost_caravan, Caravan_test, n.trees = 1000, type = "response")
pred_test_boost <- ifelse(probs_boost_test > 0.2, 1, 0)
table(Caravan_test$Purchase, pred_test_boost)


#Fraction of people of who predicted to make a purchase that actually made a purchase is
# is **CONFIRM**

#Now compare with logistic regression

logit_caravan <- glm(Purchase ~ ., data = Caravan_train, family = "binomial")

probs_logit_test <- predict(logit_caravan, Caravan_test, type = "response")

pred_test_logit <- ifelse(probs_logit_test > 0.2, 1, 0)
table(Caravan.test$Purchase, pred_test_logit)

#Now the fraction of people who to make a purchase that actually made a purchase is


#Now do a KNN model
library(caret)
library(e1071) 
library(class)
knn_caravan <- knn(Caravan_train, Caravan_test, Caravan_train$Purchase, k = 10)

error_knn <- mean((knn_caravan != Caravan_test$Purchase))

error_knn

#error rate for KNN, K = 10 was about 6%, actually fairly good


pvi<- read.csv("Poverty Vs Incarceration.csv")
head(pvi)

cor(pvi)

pvivd <- read.csv("PovIncDrug.csv")
head(pvivd)

cor(pvivd)
pvivd$State <- NULL
head(pvivd)

cor(pvivd)
