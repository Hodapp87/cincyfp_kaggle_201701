#!/usr/bin/env Rscript

## install.packages(c("rpart", "dplyr"))
library(dplyr)
library(rpart)

## Read in CSV data, and drop "id" column since it's not needed:
data <- read.table("../data/winequality-data.csv", sep=",", header=TRUE)
data$id <- NULL

## Split into inputs & outputs, training & testing (1/3 for testing):
split_idx <- round(nrow(data) * 0.33)
train <- slice(data, 1:split_idx)
test  <- slice(data, (split_idx+1):n())

## Fit a decision tree, modeling the 'quality' column from every other
## attribute:
tree <- rpart(quality ~ ., data = train)

## Get training & testing error:
mae <- function(input, model) {
    mean(abs(predict(model, input) - input$quality))
}
print(sprintf("rpart decision tree, training error: %f", mae(train, tree)))
print(sprintf("rpart decision tree, testing error: %f",  mae(test,  tree)))

## Fit a linear model, and again get training & testing error:
lr <- lm(quality ~ ., data = train)
print(sprintf("R linear model, training error: %f", mae(train, lr)))
print(sprintf("R linear model, testing error: %f",  mae(test,  lr)))

## Read in "solution" input:
soln <- read.table("../data/winequality-solution-input.csv", sep=",",
                   header=TRUE)

## Predict for decision tree & linear model:
soln_predict_tree <- predict(tree, soln)
soln_predict_lm <- predict(lr, soln)

## Format the predictions as a submission, and write a CSV:
submission <- data.frame(id = soln$id,
                         quality = soln_predict_tree)
write.table(submission, sep=",", quote=FALSE, row.names = FALSE,
            "winequality-submission-tree.csv")
submission <- data.frame(id = soln$id,
                         quality = soln_predict_lm)
write.table(submission, sep=",", quote=FALSE, row.names = FALSE,
            "winequality-submission-lr.csv")
