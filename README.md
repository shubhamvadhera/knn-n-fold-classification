# Cosine similarity kNN based classification using 10-fold cross-validation

This is a program to transform text into vectors, and then perform kNN based classification on the resulting vectors.
It uses the badges UCI dataset (https://archive.ics.uci.edu/ml/machine-learning-databases/badges/badges.data).

Given the input parameter c, for each name, a vector is constructed of c-mer terms (usually called k-mers, but here we name them c-mers since the input variable k is being used for kNN), by enumerating all subsequences of length c within the name. For example, if c=3, “naoki abe” becomes < “nao”, “aok”, “oki”, “ki “, “I a”, “ ab”, “abe” >. Finally, we construct sparse term-frequency vectors for each of the objects.

Using the constructed vectors and their associated classes, given the input parameter k, the program performs kNN based classification using 10-fold cross-validation and report the average classification accuracy among all tests. The class of the test sample is chosen by majority vote, with ties broken in favor of the class with the highest average similarity. In the rare case that the test sample does not have any neighbors (no features in common with any training samples), a random value is drawn from a uniform distribution over [0,1) and the test sample is classified as “+” if the value is greater than 0.5 and “-“ otherwise.

Credits:
Inspired heavily from Prof. David C. Anastasiu
