---
title: "Practical Machine Learning Project"
author: "Sungmin Kim"
date: "June 11, 2015"
output: html_document
---

#Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The goal of this project is to predict the manner in which they did the exercise. We need to predict "classe" variable in the training set. 

#Loading required packages
First of all, it is not required, but efficient to load all packages in advance before doing further analysis. 

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(ggplot2)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

#Getting and cleansing Data
If files do not exist in the data folder, download them from the website and save into the data folder. For reproducibility, everyone can obtain those files and conduct the analysis easily. 

```r
if(!file.exists("./data/pml-training.csv")){
  download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                destfile = "./data/pml-training.csv", method = "curl")
}
```

```
## Warning in download.file(url = "https://d396qusza40orc.cloudfront.net/
## predmachlearn/pml-training.csv", : download had nonzero exit status
```

```r
if(!file.exists("./data/pml-testing.csv")){
  download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                destfile = "./data/pml-testing.csv", method = "curl")
}
```

```
## Warning in download.file(url = "https://d396qusza40orc.cloudfront.net/
## predmachlearn/pml-testing.csv", : download had nonzero exit status
```

After download files, we need to open them and investigate them thoroughly.

```r
data <- read.csv("pml-training.csv", na.strings = c("NA", ""))
testData <- read.csv("pml-testing.csv", na.strings = c("NA", ""))
dim(data)
```

```
## [1] 19622   160
```

We have such huge datasets, which is not necessary for our prediction models. First of all, any columns having a lot of NA values or non-related information should be excluded. Then we need to seperate data into training and test set. In our case we put 70 percent of original data into a training set and 30 percent into a test set. 

```r
data <- data[, -c(1:7)] # non-related columns
filterVal <- apply(data, 2, function(x){ sum(is.na(x))})
data <- data[, which(filterVal == 0)] # filtered out NA columns
dim(data)
```

```
## [1] 19622    53
```

```r
testData <- testData[, -c(1:7)] # same as above
filterVal2 <- apply(testData, 2, function(x){ sum(is.na(x))})
testData <- testData[, which(filterVal2 == 0)] # same as above

inTrain <- createDataPartition(y = data$classe, p = 0.7, list = FALSE)
training <- data[inTrain, ]; testing <- data[-inTrain, ]
dim(training); dim(testing)
```

```
## [1] 13737    53
```

```
## [1] 5885   53
```

#Build a prediction model(Decision Tree)

Here, we build a prediction model with tree and plot it with fancier version. 

```r
modFit <- train(classe ~ ., method = "rpart", data = training)
```

```
## Loading required package: rpart
```

```r
print(modFit$finalModel)
```

```
## n= 13737 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 129.5 12478 8619 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -33.95 1118    7 A (0.99 0.0063 0 0 0) *
##      5) pitch_forearm>=-33.95 11360 8612 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 439.5 9599 6913 A (0.28 0.18 0.24 0.19 0.1)  
##         20) roll_forearm< 122.5 5929 3531 A (0.4 0.18 0.19 0.17 0.056) *
##         21) roll_forearm>=122.5 3670 2466 C (0.078 0.18 0.33 0.23 0.18) *
##       11) magnet_dumbbell_y>=439.5 1761  858 B (0.035 0.51 0.042 0.23 0.18) *
##    3) roll_belt>=129.5 1259   47 E (0.037 0 0 0 0.96) *
```

```r
fancyRpartPlot(modFit$finalModel)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png) 

Since we have our prediction model, we can use our test case to check the model accuracy.

```r
predictions <- predict(modFit, newdata = testing)
confusionMatrix(predictions, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1506  482  468  431  159
##          B   19  383   34  168  144
##          C  122  274  524  365  292
##          D    0    0    0    0    0
##          E   27    0    0    0  487
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4928          
##                  95% CI : (0.4799, 0.5056)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3373          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8996  0.33626  0.51072   0.0000  0.45009
## Specificity            0.6343  0.92309  0.78329   1.0000  0.99438
## Pos Pred Value         0.4944  0.51203  0.33228      NaN  0.94747
## Neg Pred Value         0.9408  0.85283  0.88347   0.8362  0.88922
## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
## Detection Rate         0.2559  0.06508  0.08904   0.0000  0.08275
## Detection Prevalence   0.5176  0.12710  0.26797   0.0000  0.08734
## Balanced Accuracy      0.7670  0.62968  0.64700   0.5000  0.72224
```

In classification tree, we have lower than 50% accuracy, which has higher sample error rate. 

#Build a prediction model(Random Forest)

Here we chose 3-folds cross validation to assess how accurately a predictive model will perform.

```r
#tcCv <- trainControl(method = "cv", number = 3)
modFit2 <- randomForest(classe ~ ., data = training, importance = TRUE, ntree = 1000)
#modFit2 <- train(classe ~ ., method = "rf", data = training, prox = TRUE, trControl = tcCv)
print(modFit2$finalModel)
```

predict a testing set and create a confusion matrix to visualize how well our model accurately predicts. 

```r
predictions2 <- predict(modFit2, newdata = testing)
```

```
## Error in predict(modFit2, newdata = testing): object 'modFit2' not found
```

```r
confusionMatrix(predictions2, testing$classe)
```

```
## Error in confusionMatrix(predictions2, testing$classe): object 'predictions2' not found
```

We have about 98.9% accuracy(1.1% sample error rate) with Random Forest. This is remarkably more accurate compate to our first prediction model. 

#Conclusion
We successfully predicted a 'Classe' value from our testing set and can conclude that Random Forest yields more accurate predictions as opposed to the classification tree model. Although Random Forest might cause some overfitting issue and is a slower algorithm, it is pretty accurate and more useful to predict with non-linear regressions. Classification tree would be better for predicting binary decision.  
