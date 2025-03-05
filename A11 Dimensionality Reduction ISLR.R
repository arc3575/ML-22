#Set csv equal to variable
#Homework 4.14.a 
#(a) Create a binary variable, mpg01, that contains a 1 
#if mpg contains a value above its median, 
#and a 0 if mpg contains a value below its median. 
#You can compute the median using the median() function. 
#Note you may find it helpful to use the data.frame() function to create a single data set 
#containing both mpg01 and the other Auto variables. 
#This csv removed 5 observations because the horsepower values were missing
#This csv also removed the column
auto <- read.csv("Auto_Horsepower.csv")

#First create variable to hold median
medianmpg <- median(auto$mpg)

median(auto$mpg)
#Next set mpg01 equal to the aforementioned formula
#if auto$mpg is above median, assign a 1 
#if auto$mpg is below median, assign a 9'

mpg01 <- (auto$mpg > medianmpg)*1
#use summary to check if mpg01 actually works

summary(mpg01)

str(auto)

#(b) Explore the data graphically 
#in order to investigate the association between mpg01 and the other features. 
#Which of the other features seem most likely to be useful in predicting mpg01? 
#Scatterplots and boxplots may be useful tools to answer this question. 
#Describe your findings.

#Combine mpg01 and auto

auto_mpg01 <- data.frame(mpg01, auto)
dim(auto_mpg01)

#First look at correlations

cor(auto_mpg01[, -10])

#The strongest relationships between mpg01 and the other variables 
#are with cylinders, displacement, horsepower, and weight.
#Other than mpg itself, which is basically the same variable

#Next create pairs based on these closely correlated variables

pairs(~mpg01 + cylinders + displacement + horsepower + weight, auto_mpg01, col = c("orange", "blue", "green", "purple"))

#Change cylinder to factor

cylinders = as.factor(auto_mpg01$cylinders)
summary(cylinders)

#Plots show that there might be a sigmoid fit with mpg01 and displacement, horsepower and weight

#The following are scatter plots of the relevant variables

par(mfrow = c(2,2))
plot(auto_mpg01$cylinders, mpg01, col = "blue", xlab = "cylinders", main = "cylinders vs. mpg01")
plot(auto_mpg01$displacement, mpg01, col = "green", xlab = "displacement", main = "displacement vs. mpg01")
plot(auto_mpg01$horsepower, mpg01, col = "red", xlab = "horsepower", main = "horsepower vs. mpg01")
plot(auto_mpg01$weight, mpg01, col = "purple", xlab = "weight", main = "weight vs. mpg01")

#The following are box plots of the relevant variables

par(mfrow = c(2,2))
boxplot(mpg01 ~ cylinders, data = auto_mpg01, main = "cylinders vs mpg01", xlab = "cylinders", col='blue')
boxplot(mpg01 ~ displacement, data = auto_mpg01, main = "displacement vs mpg01", xlab = "displacement", col='green')
boxplot(mpg01 ~ horsepower, data = auto_mpg01, main = "horsepower vs mpg01", xlab = "horsepower", col='red')
boxplot(mpg01 ~ weight, data = auto_mpg01, main = "weight vs mpg01", xlab = "weight", col='purple')

#It was very hard to garner any relationship or information from the boxplots, but the scatter plots show definite relationships

#(c) Split the data into a training set and a test set

#need to use the function "createDataPartition()
#This function is contained in the "caret" library

library(caret)

#Partition data useing the creatDataPartition() function

set.seed(999)
inTraining <- createDataPartition(auto_mpg01$mpg01, p = 0.8, list = FALSE)

#Assign variables; training and test
training <- auto_mpg01[inTraining, ]
test  <- auto_mpg01[-inTraining, ]


#(d) Perform LDA on the training data in order to predict mpg01
#using the variables that seemed most associated with mpg01 in (b). 
#What is the test error of the model obtained?

# check dimension of training and test data

dim(training)
dim(test)

#The training and test data look good

#Now create linear dimensional analysis model using training
#lda (linear dimensional analysis) fucntion is contained in library "MASS"
#Call MASS library
library(MASS)

#Set lda model
lda_model1 <- lda(mpg01 ~ cylinders + displacement + horsepower + weight, data = training)

lda_model1_prediction <- predict(lda_model1, test)$class

#Create table that looks at prediction

table(lda_model1_prediction, test$mpg01)

#Compute error in the linear dimensional analysis model

error_lda <- mean(lda_model1_prediction != test$mpg01)

#Call error_lda
error_lda

#Value is 0.064
#Translates to 6.4% error

#(e) Perform QDA on the training data in order to predict mpg01
#using the variables that seemed most associated with mpg01 in
#(b). What is the test error of the model obtained?


#Create quadratic linear model
#This is included in MASS

qda_model1 <- qda(mpg01 ~ cylinders + displacement + horsepower + weight, data = training)
qda_prediction <- predict(qda_model1, test)$class
table(qda_prediction, test$mpg01)

#Calculate error for qda

error_qda <- mean(qda_prediction != test$mpg01)

#Display error_qda

error_qda

#This is 0.0512, which is 5.12% 

#(f) Perform logistic regression on the training data 
#in order to predict mpg01 using the variables that seemed most associated with
#mpg01 in (b). 
#What is the test error of the model obtained?

#Create logistic regression model

logRegModel1 <- glm(mpg01 ~ cylinders + displacement + horsepower + weight, data = training, family =binomial)

logRegModel1_prob <- predict(logRegModel1, test, type = "response")

logRegModel1_predict <- ifelse(logRegModel1_prob > 0.5, 1, 0)

table(logRegModel1_predict, test$mpg01)

#Calculate error

error_glm <- mean(logRegModel1_predict != test$mpg01)

#show error for glm

error_glm

#Shown to be 0.1025, which is about 10%



#(h) Perform KNN on the training data, with several values of K, in order to predict mpg01. 
#Use only the variables that seemed most associated with mpg01 in (b). 
#What test errors do you obtain?
#Which value of K seems to perform the best on this data set?

# Loading package 
library(e1071) 
library(class)
#Establish training/testing data for K using cbind() function

training_K <- cbind(training$cylinders, training$displacement, training$horsepower, training$weight)
test_K <- cbind(test$cylinders, test$displacement, test$horsepower, test$weight)

knn_prediction <- knn(training_K, test_K, training$mpg01, k = 10)

#Calculate error for 

error_knn <- mean(knn_prediction != test$mpg01)

#Show error

error_knn

#This was found to be 0.077 or 7.7%

#After trying K 1 thru 10, the error was calculated for each model
# K = 1: 0.1282051, 12%
# K = 2: 0.0897435, 8.97%
# K = 3: 0.0897435, 8.97%
# K = 4: 0.0897435, 8.97%
# K = 5: 0.0512820, 5.12%
# K = 6: 0.0769230, 7.69%
# K = 7: 0.0641025, 6.41%
# K = 8: 0.0512820, 5.12%
# K = 9: 0.0641025, 6.41%
# K = 10: 0.064102, 6.41%

#The best values for K between 1 and 10 are K = 5 and K = 8

#Also remember that I split the test and training data by 20/80
#If I change that split to 25/75 or 30/70, this will also effect the error of the downstream models


#15. This problem involves writing functions.
#(a) Write a function, Power(), that prints out the result of raising 2 to the 3rd power. 
#In other words, your function should compute
#23 and print out the results.
#Hint: Recall that x^a raises x to the power a. Use the print()
#function to output the result

#Write function power that raises 2 to the 3

Power <- function(){print(2^3)}

#Test power function

Power()

#Output is 8 which is equal to 2^3

#(b) Create a new function, Power2(), that allows you to pass any
#two numbers, x and a, and prints out the value of x^a. You can
#do this by beginning your function with the line
#> Power2 <- function(x, a) {
#  You should be able to call your function by entering, for instance,
#  > Power2(3, 8)
#  on the command line. This should output the value of 38, namely,
#  6, 561

#Create Power2 function

Power2 <- function(x, a){print(x^a)}

#Test power2 with 3 and 8

Power2(3, 8)

#This is equal to 6561

#c) Using the Power2() function that you just wrote, compute 10^3,
#8^17, and 131^3.

#Test Power2 function with 10^3
Power2(10,3)

#Answer is 1000, correct

#Test Power2 function with 8^17

Power2(8, 17)

#Answer is 2.2518e+15, correct

#Test Power2 function with 131^3

Power2(131, 3)

#Answer is 2,248,091

#(d) Now create a new function, Power3(), that actually returns the
#result x^a as an R object, rather than simply printing it to the screen. 
#That is, if you store the value x^a in an object called
#result within your function, then you can simply return() this return() result, 
#using the following line:
#return(result)
#The line above should be the last line in your function, before
#the } symbol.

#Create Power3()

Power3 = function(x , a) {
  result <- x^a
  return(result)
  }

#(e) Now using the Power3() function, create a plot of f(x) = x2.
#The x-axis should display a range of integers from 1 to 10, and
#the y-axis should display x2. Label the axes appropriately, and
#use an appropriate title for the fgure. Consider displaying either
#the x-axis, the y-axis, or both on the log-scale. You can do this
#by using log = "x", log = "y", or log = "xy" as arguments to
#the plot() function.

Power3(1, 2)

#Use plot function

plot(1:10, Power3(1:10, 2), xlab = "x", ylab = "f(x)", main = "f(x) = x^2")

#Use log = "xy" to adjust scale
#Can be added in plot function

plot(1:10, Power3(1:10, 2), log ='xy',xlab = "x", ylab = "f(x)",main = "log-scale")

#(f) Create a function, PlotPower(), that allows you to create a plot
#of x against x^a for a fixed a and for a range of values of x. For
#instance, if you call
#> PlotPower(1:10, 3)
#then a plot should be created with an x-axis taking on values
#1, 2,..., 10, and a y-axis taking on values 13, 23,..., 103.

PlotPower = function(x, a){
  plot(x, x^a, main = "x vs x^a")
}

#Test out PlotPower

PlotPower(1:10, 3)
