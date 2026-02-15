#import "config.typ": *
#show: apply-styling

= ML Week 3

== Part 1: Neural Network Theory

=== What is training?

Training adjusts the weights of the network through backpropagation, and the only thing changing during training are weights. 

_Why only weights?_ Weights are the learnable parameters - the architecture (neurons, layers) and data are fixed before training begins.

=== Neuron Architecture

*Structure:*

` x inputs → [Neuron] → 1 output`

A neuron always has:

- Multiple inputs (variable: x₁, x₂, ..., xₙ )
- Exactly 1 output (fixed)

*Why this matters:* Understanding this 1-output rule is fundamental for building network architectures. One neuron's output can feed into many other neuron's inputs.

=== Purpose of Hidden Layers

Hidden Layers enable better predicitons by learning non-linear patterns.

*Without hidden layers:*

- Only linear relationships: y = m x + b
- Cannot learn complex patterns

*With hidden layers*

- Non-linear relationships
- Complex patterns like _"Customer over 60 with low balance AND from Germany churn more"_

=== Activation Functions

*Definition:* Transforms the weighted sum in the neuron's output.

- Calculate the weighted sum:

$ v = #sym.sum ( "input" x "weight" ) + "bias" $

- Apply activation function:

$ "output" = f(v) $

=== Common Functions:

- *ReLU* (used in hidden layers)

$ f(x)  = "max" (0,x) $

- *Sigmoid* (outputs 0 to 1)

$ f(x) = 1/(1+e^(-x)) $

- *Softmax*: Converts oututs to probabilites summing to 1

#infobox[The activation function is NOT just forwarding the data, it is a mathematical transformation of the weighted sum]

== Part 2: Data Preprocessing

=== Converting Text to Numbers

Neural Networks perform mathematical operations, so you can't work with strings. However, in Data we often do have strings, therefore we need to convert these strings into numerical values.

*OneHot Encoding*

If we have Spain, Germany and France, we might be tempted to say they are equal to increasing integers, so 1 = Spain, 2 = Germany and 3 = France.

However, the model might intepret that as France > Spain > Germany which creates a false oridnal relationship.

*The solution:*

```python
Spain:   [1, 0, 0]
France:  [0, 1, 0]  
Germany: [0, 0, 1]

# Now independent - no ordering implied
```

*In our code:*

```python
Geography_Germany  Geography_Spain   Meaning
      0                  0          = France
      1                  0          = Germany
      0                  1          = Spain
```

== Part 3: Loss Functions & Evaluations

=== Choosing Loss Functions

Loss functions measures prediction error.

*Mean Squared Error (MSE)* - Regression:

```python
# For continuous values: age, height, price, temperature
loss = (predicted - actual)²

# Example: Predicting house prices
model.compile(loss='mean_squared_error')
```

*Binary Cross Entropy* - Binary Classification:

```python
# For two categories: spam/ham, churn/stay, pass/fail
model.compile(loss='binary_crossentropy')

# Our Customer Churn project uses this!
```

*Categorical Cross Entropy* - Multi Class:

```python
# For 3+ categories: Democrat/Republican/Independent
model.compile(loss='categorical_crossentropy')
```

=== Model Evaluation

*Training accurasy vs Evaluation accuracy:*

```python
# WRONG: Training accuracy
# High accuracy but may be overfitted
model.fit(X_train, y_train, epochs=100)
# Don't trust this accuracy!

# CORRECT: Evaluation accuracy  
# Tests on NEW, unseen data
loss, accuracy = model.evaluate(X_test, y_test)
# This is the true measure!
```

#infobox[Training accuracy measures memorization. Evaluation accuracy measures generalization.]

== Part 4: Forward & Backward Propagation with ReLU

=== Forward Propagation Explained

*What happens in forward propagation:*

Each neuron performs two steps:

1. Calculate weighted sum (v1)
  - Multiply each input by it's weight, add bias
2. Apply activation function (y)
  - Transform the weighted sum (using ReLU in our case)

*Notation:*

- v = weighted sum (before activation)
- y = output (after activation)
- subscript = which neuron (v1 = neuron 1's weighted sum)

*Example Calculation*

*Neuron 1 in hidden layer:*

$v_1$ = sum of $("inputs" * "weights")$ + bias = 0.17

$y_1$ = ReLU($v_1$) = ReLU(0.17) = 0.17 (ReLU keeps positive values)

*Neuron 2 in hidden layer:*

$v_2$ = sum of $("inputs" * "weights")$ + bias = 0.36

$y_2$ = ReLU($v_2$) = ReLU(0.17) = 0.36

*Output neuron:*

$v_3$ = sum of $(y_1 * "weight" + y_2 * "weight")$ + bias = 0.342

$y_3$ = ReLU($y_3$) = 0.342

*Calculate error:*

$ "error" = "desired output" - "actual output" = 0.9 - 0.342 = 0.558 $

=== Backward Propagation Explained

*Purpose:*

Figure out how much each weight contributed to the error, then adjust weights to reduce further errors.

*Step 1 - Calculate Gradients (#sym.delta - "delta"):*

#sym.delta tells us "how much did this neuron contribute to the error?"

- #sym.delta at output layer = the error itself = 0.558
- #sym.delta at hidden neurons = output layer's #sym.delta x the weight connecting them.
  - #sym.delta for hidden neuron 1 = 0.558 x 0.2 = 0.1116
  - #sym.delta for hidden neuron 2 = 0.558 x 0.3 = 0.1674

*Why multiply by weights?*

The error flows backwards through the network. Larger weights mean the neuron had more influence on the output, so it gets more "blame" for the error.

*Step 2 - Update weights"*

Basic weights update formula:

$ w("'new'") = w("'old'") + #sym.eta * #sym.delta * "input" $

*Where:*

- #sym.eta (eta) = learning rate (how big of a step to take, e.g. 0.25)
- #sym.delta (delta) = gradient (how much this weight contributed to the error)
- Input = the value that came into this weight

*Example*: Update output layer with weight $w_31$

- Old weight = 0.2
- Learning rate = 0.25
- Gradient = 0.558
- Input = 0.17
- New weight = 0.2 + (0.25 × 0.558 × 0.17) = 0.2237

*Advanced: With Momentum:*

$ w("'new'") = w("'old'") + #sym.alpha * w("'previous'") + #sym.eta * #sym.delta * "input" $

*Where:*

- #sym.alpha (alpha) = momentum coefficient (e.g. 0.0001)

*Why momentum?*

- Remembers the previous direction of weight change
- Helps avoid getting stuck in local minima
- Speeds up learning by building "velocity" in consistent directions
- Like a ball rolling downhill - it doesn't stop at every small bump

Momentum adds: $#sym.alpha * "previous weight value" $ (typically tiny, like 0.0001 × 0.2 = 0.00002).

== Part 5: Deployment of ML model

=== Project Structure

```bash
/home/vss/ml/predict/
├── app.py              # Flask web server
├── model.keras         # Trained model
├── scaler.pkl          # Feature scaler
└── templates/
    └── index.html      # Web interface
```

=== Loading the Model 

```python
from tensorflow.keras.models import load_model

model = load_model('model.keras')
```

*Key Point:*

Model was trained in Google Colab - it was saved with `model.save('model.keras')`, then uploaded to the server.

=== Loading the Scaler (Pickle)

```python
import pickle

scaler_data = pickle.load(open('scaler.pkl', 'rb'))
scaler = scaler_data['scaler']
feature_columns = scaler_data['features']
```

*Why do we need a scaler?*

Features have vastly different ranges:

CreditScore:      300 - 850

Age:              18 - 100  

Balance:          0 - 250,000

EstimatedSalary:  0 - 200,000

#infobox[Without scaling, Balance would dominate because it's values are 1000x larger than Age.]

*Before/After scaling:*

```python
# Before scaling
[650, 45, 50000]  # Different magnitudes!

# After scaling  
[0.583, 0.451, 0.200]  # Similar magnitudes
```

*Why save with pickle?*

Must use THE SAME scaker for deployment that was used during training, Pickle preserves the exact scaling parameters.

*Continous (Scaled)*

```python
in1:  CreditScore       # 300-850
in2:  Age               # 18-100
in3:  Tenure            # 0-10 years
in4:  Balance           # 0-250,000
in5:  NumOfProducts     # 1-4
in8:  EstimatedSalary   # 0-200,000
```

*Binary*

```python
in6:  HasCrCard         # 1=Yes, 0=No
in7:  IsActiveMember    # 1=Yes, 0=No
```

*OneHot Encoded:*

```python
in9:  Geography_Germany # 1=Germany, 0=Not
in10: Geography_Spain   # 1=Spain, 0=Not
# France = [0,0] implicit

in11: Gender_Male       # 1=Male, 0=Female
```
=== Prediction Pipeline:

```python
# 1. Collect inputs from form
in1 = request.form.get('in1')  # CreditScore
in2 = request.form.get('in2')  # Age
# ... all 11 inputs

# 2. Check for missing data
if None in [in1, in2, in3, in4, in5, in6, 
            in7, in8, in9, in10, in11]:
    return render_template('index.html', 
                         result=None)

# 3. Create numpy array
arr = np.array([[
    float(in1), float(in2), float(in3), 
    float(in4), float(in5), float(in6),
    float(in7), float(in8), float(in9), 
    float(in10), float(in11)
]])

# 4. Scale the features
arr_scaled = scaler.transform(arr)

# 5. Make prediction
predictions = model.predict(arr_scaled)
probability = predictions[0][0]

# 6. Interpret result
if probability > 0.5:
    prediction_text = "Customer will LEAVE"
else:
    prediction_text = "Customer will NOT leave"

# 7. Return to user
return render_template('index.html',
    result=f"{probability:.6f}",
    prediction_text=prediction_text,
    probability=probability)
```

== Key Takeaways

=== Neural Network Theory:

1. Training adjusts weights only, not architecture or data.
2. Neurons have x inputs and only 1 output, always!
3. Hidden layers enable *non-linear* pattern learning
4. Activation functions transform weighted sums, not just forward data.

=== Data Preprocessing:

1. Use evaluation accuracy on unseen data, not training accuracy.
2. MSE for regression, Binary Cross Entropy for classification.

=== Backpropagation:

=== Deployment:

1. Pickle preserves scaler - criticaæ for consistent preprocessing.
2. Feature scaling prevents large-value features from dominating