# ASSIGNMENT 3 ---- DECISION TREES  
---

|  |  |
| :--- | ---: |
| Name: Mariam Magued Habib Bebawy | [Code Tree Implementation](./A1_1stQuestion.py) |
| Sec.: 2 | [Dataset](./cardio_train.csv) |
| B.N.: 27 | [Manual Tree Implementation](./A3_2ndQuestion.pdf) |  

---
## FIRST REQUIREMENT ----> CODE TREE IMPLEMENTATION  
---

#### Main Function

* start by reading the data into a dataframe, and perform delimiting techniques.
* drop a few columns off the dataframe after some trial and error to find out which one actually do affect the tree.
* define the values and labels of the datafram
* split the aquired data into train and test splits  
_note that "random state = 0" is used for testing to provide the same values every run-time_
* define the implemented decision tree and scikit-learn decision tree to test on both.
* perform data-fitting on the train data for both dTree and skTree.
* perform prediction on the test data for both dTree and skTree.
* calculate the accuracy of the prediction of both models.
* <code>time.time()</code> is used before and after certain function calls to calculate the time needed for that function, all for the sake of comparison.

* in the dataframe, the dropped columns has been tested to see if any change in accuracy would take place.
* accuracy changes were too small in comparison to the time difference in training, compromises were made.
* the maxDepth of the tree has also been tested to see which number would give the highest accuracy with the lowest training-time.

* the code is ready to be run, no need for any adjustments

---
## SECOND REQUIREMENT ----> MANUAL TREE IMPLEMENTATION  
---

* you can find the file attached in the submission form.
* in the file, you can find the tree and sub-trees used to reach the solution, the steps to implement the tree, and the final solution of the tree.