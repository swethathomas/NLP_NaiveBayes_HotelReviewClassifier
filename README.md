# NLP_NaiveBayes_HotelReviewClassifier
In this project I implement a Naive Bayes Classifier to classify Hotel Reviews  as positive or negative and truthful or deceptive. This code only uses python in built packages to build the classifier.

The dataset is excerpted from the Deceptive Opinion Spam Corpus v1.4 and contains hotel reviews that are positive, negative, truthful and deceptive. 80% of the data is used for training and 20% for testing

There are two python scripts nblearn3 and nbclassify3.

nblearn3 takes in the training data as input and learns prior and conditional probabilites for the different classes 

nbclassify3 takes in test data as input and calculates posterior probabilty for each review from the conditinal and prior probability learnt from the training data in nblearn3. It then assigns the class with the maximum posterior probability
