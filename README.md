# orGatePerceptron
Perceptron is a elementary neuron without hidden layers like neural networks. It does not exist in the real world, but helps us to understand how machine learning happens. This code tries to show how a perceptron learns to predict an OR gate. It uses gradient descent algorithm to tune the weightages for inputs.

Aim is to make the code readable and understandable by novice programmers. So, it is not optimized for speed in computation.

# Pre-requisites:
Python 3.6.9 or later should be installed in the system.

# How to run:
Open terminal in folder containing the python file in this repo. Run following command in terminal:
python3 orGatePerceptron.py

(Tested working fine in Ubuntu 18.04.5 LTS.)

# How to run in a more complex manner:

Open orGatePerceptron.py in your favorite editor.
For a start, edit the variable under:
1) lr                      -- always give values less than zero
2) MSE_EXPECTED            -- add a few more thousands (1000) in the denominator. Or delete some.
3) ITR_MULT_TO_PRINT_INFO  -- put this as 1. Then 10. Then 100. Then 10000 (ten thousand).

After editing,run in terminal:
python3 orGatePerceptron.py

If that's too simple for you, edit the rest of the file the way you like it.

# Goodies:
To see an excel sheet implementation of the same logic, open the orGatePerceptron.ods in LibreOffice Calc (application in Ubuntu/Linux). They are put in results-sample folder.
