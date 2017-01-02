# cincyfp_kaggle_201701

This contains some example implementations for the Kaggle competition
done at the CincyFP meeting on January 10th, 2016.
See <https://inclass.kaggle.com/c/uci-wine-quality-dataset> to
download the data files.

## R

See [R/example.R](R/example.R).  This uses
the [rpart](https://cran.r-project.org/web/packages/rpart/index.html)
library to fit a decision tree to the data, and also uses R's built-in
linear models for regression.

For those not familiar with R, you'll probably need to first run:

```r
install.packages(c("rpart", "dplyr"))
```

## Python
 
See [python/example.py](python/example.py).  This uses two separate
models from [scikit-learn](http://scikit-learn.org/) (decision trees
and linear regression). You'll need the `sklearn` and `pandas`
packages.


## Clojure 

See [Clojure/wine-quality](Clojure/wine-quality) project to see an
example using a Weka model.  You will need to
install [Weka](http://www.cs.waikato.ac.nz/ml/weka/) if you want to
interactively explore the different models.
