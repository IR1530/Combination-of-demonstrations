The project name is Combination of demonstrations using ICL.

Here, we are implementing an approach by using a pair of examples for prediction aiming to improve few-shot in-context learning by selecting compatible example pairs.

Here's a detailed breakdown of the steps:

1.Neighborhood Selection:

For a given query instance x, find its k-nearest neighbors in the training set.
These neighbors will be denoted as z1, z2, z3, ..., zk.

2.One-shot Prediction:

For each neighbor zi, perform a one-shot prediction using (x, zi) pair.
Compare the prediction (y_hat) with the ground truth label (y).
Assign a binary label: 1 if the prediction is correct, 0 if incorrect.

3.Pair Generation:

Create all possible pairs of neighbors: (zi, zj) where i ≠ j.
The total number of pairs will be k choose 2.

4.Compatibility Labeling:

For each pair (zi, zj), determine their compatibility based on their one-shot prediction and two-shot prediction results.

5.Training Data Creation:

For each query x, create training instances of the form:

Positive example: (x, zi, zj) where (zi, zj) is a compatible pair
Negative examples: (x, zi, zk) where (zi, zk) is an incompatible pair

6.Model Training:

Train a model θ using triplet loss.
The model should take (x, zi, zj) as input and output a similarity score.
The objective is to maximize the score for compatible pairs and minimize it for incompatible pairs.

7.Inference:

For a new query x:
a. Find its k-nearest neighbors.
b. Generate all possible pairs of neighbors.
c. Use the trained model θ to predict compatibility scores for all pairs.
d. Sort the pairs in descending order of their compatibility scores.
e. Select the top-scoring pair(s) to use as examples for two-shot prediction.

8.Evaluation:

Compare the performance of this method against the baseline of using random pairs for two-shot prediction.

Aim : This approach aims to improve ICL by selecting example pairs that are likely to work well together, addressing the issue of "non-cooperation" between examples that may occur in standard few-shot learning.
