#set text(16pt, font: "New Computer Modern")
#set par(justify: true, leading: 0.58em)
#set enum(numbering: "I.")
#set page(header: [
  #set text(size: 12pt)
  _Valdemar Støvring Storgaard_
  #h(1fr)
  Erhvervsakademi København
])

#set page(numbering: "- 1 -")
#show link: it => underline(text(it, blue))

= ML - Week 1

=== What is Machine Learning?

ML is a branch of AI where computers can learn patterns from data, rather than follwing hardcoded rules. ML systems can improve their perfomance on tasks through experience and examples.

The system identifies patterns in training data and can then use those patterns to make predicitons about new, unseen data.

==== ML Components

- Data (_numpy arrays_): Raw information our model learns from.
- Neural Network (_Sequential Model_): The 'brain' that processes the patterns, layers of neurons that transforms input data.
- Activation Function (_relu, softmax_): Decision-making mechanisms that determine how neurons respond to inputs.
- Optimizer(_Adam_): The learning mechanism that adjusts the model to improve predictions over time.
- Loss Function (_sparse_categorical_crossentropy_): The measure of how wrong our predictions are, guiding the learning process.
- Training (_epochs, batches_): The repetitive learning process where the model sees examples and improves.

=== Political Affilitation Prediction Model

This neural network can predict political affiliation in the US (Democrat, Republican or Independent). It will be based on the following demographic features: age, income, sex and religion.

*Data Encoding*

- _Feature 0:_ Age (normalized, *0-1* scale) 
- _Feature 1:_ Income (normalized, *0-1* scale) 
- _Features 2-3:_ Sex (*one-hot* encoded) 
  - ```[1,0]```  = Male
  - ```[0,1]```  = Female
- _Features 4-6_ Religion (*one-hot* encoded) 
  - ```[0,1,0]```  = Prespytarian
  - ```[1,0,0]```  = Catholic
  - ```[0,0,1]```  = Other
  #v(1cm)
- _Output Classes_ (3 categories)
  - ```[0]``` = Democrat
  - ```[0]``` = Republican
  - ```[0]``` = Independent

*Model Architecture*

```python

model = Sequential()
model.add(Dense(5, input_dim=7, activation='relu'))
model.add(Dense(3, activation='softmax'))

```

Layer 1 (Hidden Layer)
- 5 Neurons.
- ReLU activation (sets negative values to 0, prevents negative ouputs in hidden layers).
- Processes the 7 input features.

Layer 2 (Output layer)
- 3 neurons (one per class)
- Softmax activation (converts outputs to probabilites that sum to 1)

*Training configuration*

Optimizer: Adam with learning rate: 0.01 (controls how much the model adjusts its internal parameters (weights) after each training step)

- Adam tweaks the learning rate on the fly to help the model get better results quicker

Loss Function: Sparse Categorical Crossentropy

- Appropriate for multi-class classification with integer labels
- Measures difference between predicted probabilites and actual class

*Training parameters*

- Epochs: 400 (Number of times the model sees entire dataset)
- Batch size: 4 (Number of samples processed before updating weights)

=== Example

```python
# Training data (5 samples)
X = np.array([
    [0.27, 0.24, 1, 0, 0, 1, 0],  # Democrat
    [0.48, 0.98, 0, 1, 1, 0, 0],  # Republican
    [0.33, 0.44, 0, 1, 0, 0, 1],  # Republican
    [0.30, 0.29, 1, 0, 1, 0, 0],  # Independent
    [0.66, 0.51, 0, 1, 0, 1, 0]   # Democrat
])

y = np.array([0, 1, 1, 2, 0])

# Train the model
model.fit(X, y, epochs=400, batch_size=4, verbose=1)

# Make prediction for new person
# [age=0.38, income=0.51, male, Presbyterian]
prediction = model.predict(np.array([[0.38, 0.51, 0, 1, 0, 1, 0]]))
# Returns probability distribution: [P(Dem), P(Rep), P(Ind)]

# Evaluate model performance
loss, accuracy = model.evaluate(X, y)
```
#pagebreak()

== Key Takeaways

#v(1cm)

1. _One-hot encoding_ transforms categorical variables into numerical format ML models can process.
2. _ReLU activation_ in hidden layers prevents vanishing gradients and speeds up training.
3. _Softmax activation_ in output layer provides interpretable probability distributions.
4. _Small dataset_ (5 samples) means the model may overfit - more data would improve generalizaton.
5. _High epoch count_ (400) ensures the model has sufficient opportunites to learn patterns from limited data.