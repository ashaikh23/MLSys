# MLSys

Next Steps:

Work on Supervised Learning Approach:
How to collect samples
What kind of classifier to use
Hint:

**Data Preparation:** (Should be able to use Specter2 to setup)
   - Load the dataset containing paper titles, abstracts, and labels indicating whether each paper belongs to the MLSys category or not.
   - Preprocess the text data by tokenizing, removing stopwords, and converting words to numerical representations (e.g., word embeddings).

2. **Positive and Negative Sample Selection:** (All the papers we have are positive samples and the rest are negative)
   - Identify MLSys papers as positive samples and select an equal number of non-MLSys papers as negative samples.
   - Ensure that the negative samples are carefully chosen to avoid using unrelated art papers or papers from irrelevant domains, as they may introduce bias into the classifier.

3. **Balanced Training Dataset:**
   - Combine the positive and negative samples to create a balanced training dataset.
   - Shuffle the dataset to ensure randomness in the order of samples.

4. **Model Definition:**
   - Define a binary classification model architecture using PyTorch, such as a simple feedforward neural network or a convolutional neural network (CNN).
   - Design the input layer to accept the numerical representations of paper titles and abstracts.

5. **Loss Function and Optimizer:**
   - Choose an appropriate loss function for binary classification tasks, such as binary cross-entropy loss.
   - Select an optimizer, such as stochastic gradient descent (SGD) or Adam, to update the model parameters during training.

6. **Training Loop:**
   - Iterate through the training dataset in mini-batches.
   - Forward pass: Input the paper representations into the model and compute the predicted probabilities.
   - Compute the loss between the predicted probabilities and the ground truth labels.
   - Backward pass: Compute the gradients and update the model parameters using the chosen optimizer.
   - Repeat the process for multiple epochs until convergence.

7. **Evaluation:**
   - After training, evaluate the performance of the classifier on a separate validation or test dataset.
   - Calculate metrics such as accuracy, precision, recall, and F1-score to assess the model's performance.
   - Analyze any misclassifications and adjust the model or data preprocessing steps as needed.

8. **Fine-tuning and Hyperparameter Tuning:**
   - Experiment with different model architectures, hyperparameters, and preprocessing techniques to improve performance.
   - Utilize techniques such as cross-validation to tune hyperparameters effectively and avoid overfitting.

9. **Deployment:**
   - Once satisfied with the classifier's performance, deploy it to classify new papers into MLSys and non-MLSys categories.
   - Monitor the classifier's performance over time and update it as needed to maintain accuracy.
