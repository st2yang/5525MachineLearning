## HW2
This is the folder for homework 2 which implements the SVM classifiers. The problems and optimization methods for the solvers are (1) SVM dual problem + CVXOPT (2) SVM primal (hinge loss formation) + Pegasos (the paper can be found in the /documents) (3) SVM primal with softplus approximation + SGD

## Dependencies
The implementation mainly depend on these libraries
- numpy: for arrays and matrices
- matplotlib: for plotting
- scipy: for scientific computing
- CVXOPT: used to solve a convex quadratic programming problem in svm_cvx.py

## main files
The commands to run the scripts in the code/ in the terminal
- python3 ./my_dualSVM.py /path/to/dataset.csv c_value, example: python3 ./my_dualSVM.py ../data/MNIST-13.csv 1
- python3 ./my_pegasos.py /path/to/dataset.csv k numruns, example: python3 ./my_pegasos.py ../data/MNIST-13.csv 200 5
- python3 ./my_softplus.py /path/to/dataset.csv k numruns, example: python3 ./my_softplus.py ../data/MNIST-13.csv 200 5

The scripts in the code/ for plotting:
- plot_svm_cvx.py: plot test performance, number of support vectors, geometric margin r.w.t. an array of c values
- plot_primalSVMs.py: plot the loss function values over iterations r.w.t. different SGD batch sizes for Pegasos and Softplus
