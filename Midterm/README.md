## Midterm
The midterm covers several topics: bias-variance decomposition, SVM, naive-Bayes, cross-validation, kernel method, gradient descent, boosting and AdaBoost.

## Code
The implementation of AdaBoost with simple binary classifiers as weak learners is in binary_AdaBoost.py. Note that the code is highly for the problem only and is not suitable on other data without many modifications.
The given data in data.csv is as follows
- first column: feature Potty Training Prep (weeks), value: 1, 2, 3
- second column: feature Price, value: discrete sample of a continuous variable
- third column: feature Carpet Damage, value: 0, 1, 2, 3
- fourth column: feature Color, value: 1 (Brown/White) or 2 (Yellow)
- last column: target Adoptable, value: 1 (Yes) or -1 (No)

The commands to run the script
- to print debug messages: python3 -O binary_AdaBoost.py 
- just show the final prediction: python3 binary_AdaBoost.py