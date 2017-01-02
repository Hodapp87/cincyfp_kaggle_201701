# cincyfp_kaggle_201701

This contains some example implementations for the Kaggle competition
done at the CincyFP meeting on January 10th, 2016.
See <https://inclass.kaggle.com/c/uci-wine-quality-dataset> to
download the data files.

## R

See [R/example.R](R/example.R).  This uses
the [rpart](https://cran.r-project.org/web/packages/rpart/index.html)
library to fit a decision tree to the data.

For those not familiar with R, you'll probably need to first run:

```r
install.packages("rpart")
```

## Python

See [python/example.py](python/example.py).  This uses two separate
models from [scikit-learn](http://scikit-learn.org/) (decision trees
and linear regression). You'll need the `sklearn` and `pandas`
packages.
