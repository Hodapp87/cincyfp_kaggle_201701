#!/usr/bin/env Rscript

## install.packages("rpart")

library(rpart)

## Read in CSV data, and drop "id" column since it's not needed:
train <- read.table("../data/winequality-data.csv", sep=",", header=TRUE)
train$id <- NULL

## Fit a decision tree, modeling the 'quality' column from every other
## attribute:
tree <- rpart(quality ~ ., data = train)

## Read in "solution" input:
soln <- read.table("../data/winequality-solution-input.csv", sep=",",
                   header=TRUE)

soln_predict <- predict(tree, soln)

## Format the predictions as a submission, and write a CSV:
submission <- data.frame(id = soln$id,
                         quality = soln_predict)
write.table(submission, sep=",", quote=FALSE, row.names = FALSE,
            "winequality-submission.csv")
