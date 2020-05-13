Objective: 


The code.py file contains broadly 2 parts, one for each task(Task 1 Gibbs Sampling and Classification). In totality the code picks required text and CSV file from the directory of the code file. The path isn’t hard coded, but the files need to be placed in code file directory.
It supports the Python Version 3

PART 0: Import and Storage of CSV files

All the data in the CSV file and text file is converted to ndarry using the function genfromtxt from numpy library. And finally a dictionary is made where keys are file name and values as the ndarry.


PART 1 : Implementation of Code for Task 1 (Gibbs Sampling)

1. Based on the algorithm given, it calculated the inital topic indices, Nwords(totatl number of words),D*K and K*V matrix and K is manual input
2. And then gibbs sampling is performed on each word picked up randomly using the function random permutation.
3.Then Cd and Ct is returned after running this process for about 500 iterations.


Output:
It pops out the 5 most frequent word of each topic and this same output is saved in csv and saved in the same directory of that python file

PART 2 : Implementation of Code for Task 2 (Classification using logistic regression)
1. The first step is to prepare the data for 2 representation, one is topic representation and the other is bag of words representation.
2. Then a functions ds_model performs the learning curve by computing the mean error and sd error for a varying size of the training data


Output:

It pops the graphs of error with the time for two representation.


**Please note that after every task respective graphs pops up, so please close them to get the code executed further.



