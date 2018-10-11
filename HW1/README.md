## HW1
This is the folder for homework 1 which implements Fisher Discriminant + Gaussian classifier, Logistic regression and naive Bayes with Gaussian modeling.

## Dependencies
The implementation mainly depend on these libraries
- sklearn: only used to load the data
- matplotlib: for plotting
- scipy: for scientific computing

## main files
The above classifiers are implemented as Classifier classes, and the main function would call the related functions
- main_Q3.py: for question 3, and it would do LDA 1D projection and LDA 2D projection + bivariate Gaussian classification
- main_Q4.py: for question 4, and it would use Logistic regression and naive Bayes classifiers

The interfaces required in the homework description are _LDA1dProjection(), _LDA2dGaussGM() in main_Q3.py and _logisticRegression(), _naiveBayesGaussian() in main_Q4.py
