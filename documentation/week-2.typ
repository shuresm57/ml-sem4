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

= ML - Week 2

In Week 2 we have focused on building a more realistic ML model, by usong real-world datasets.

*Key Concepts*

- Working with external datasets (CSV file)
  - One by using raw githubusercontent
  - Another by giving access to a Google Drive and then a .xls file
- Train-test splitting to evaluate model performance
- Data preprocessing and standardization
- Binary classification (Yes/No prediction)
- Multi class classification (Predicting multiple categories)
- Making predictions on new and unseen data.

=== 1. Pandas for Data Handling

```python
import pandas as pd
df = pd.read_csv(url)
```

- Pandas is a library for working with tabular data (Kind of like excel for python)
- DataFrame (df) is like a spreadsheet with rows and columns
- We use it to load, view and manipulate datasets.

=== 2. Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

*Why split the data?*
- Training set (80%): Data the model learns from.
- Test set (20%): Data the model has never seen - used to check if its actually learned or just memorized.

Kind of how we humans study practice problems ( = _training_ ) and then do a real exam with different questions
( = _testing_) to see if you actually understand the material.

```random_state=42``` ensures we get the same split every time we run the code, which makes the results
reproducible.

=== 3. StandardScaler (Data Normalization)

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
*What does it do?*
Converts all features to the same scale (mean=0, standard deviation=1)

Mean: The average value of each feature becomes 0. Values above average are positive, below are negative

Standard Deviation: Measures how spread out the values are, setting it to 1 makes all features have the same 'spread'

```
Original ages: [20, 30, 40, 50, 60]
Mean = 40

After scaling: [-1.4, -0.7, 0, 0.7, 1.4]
Mean = 0 (centered around zero)
```

*Why is it important?*

- Features have different units: age in years, salary in dollars, balance in thousands.
- _Without scaling_, features with larger numbers dominate the learning process.
- *Example:* Salary of 50,000 would overwhelm age of 40 without scaling.

*fit_transform vs transform*
- ``` fit_transform(X_train) ```: Learn the scaling parameters from training data AND apply them
- ``` transform(X_test) ```: Apply the same scaling parameters learned from training (don't 
learn new ones from test data)

*Critical rule*: Never let the test data influence the scaler that would be cheating!

=== 4. Binary classification
Predicing one of two outcomes

*Output layer:*

```python
Dense(1, activation="sigmoid")
```

- *1 neuron*: Only need one output
- *Sigmoid activation*: Squashes output to a probability between 0 and 1
- *Binary crossentropy loss*: Measures error for yes/no predictions

*Example*: Will a customer stay or leave? 

=== 5. Validation Split

```python
model.fit(X_train, y_train, epochs=20, validation_split=0.2)
```

- Takes 20% of the training data and sets it aside as "validation set"
- Used to monitor performance during training withouth touching the test set
- Helps detect overfitting early


== Project 1: Wine Quality prediction

This project was done in class.

*Dataset*
Predicts wine quality as either good or bad, based on chemical properties.

*Features*: Chemical measurements like acidity, sugar content, alcohol perc, etc

*Target*: Binary quality rating (0 = bad, 1 = good)
#pagebreak()

*Model Architecture*

```python
model = keras.Sequential([
    Dense(64, activation="tanh", input_shape=(X_train.shape[1],)),
    Dense(32, activation="tanh"),
    Dense(32, activation="tanh"),
    Dense(1, activation="sigmoid")
])
```
*Layer Breakdown*

- *Input layer*: Accepts all chemical features - ``` X_train.shape[1] ```determines this automatically
- *Hidden layers*: 64 #sym.arrow 32 #sym.arrow 32 neurons with tanh activation
- *Output layer*: 1 neuron with sigmoid (Probability between 0-1)

*Tanh activation*: Similar to sigmoid but outputs between -1 and 1 (often works better in hidden layers)

== Project 2: Customer Churn Prediction

*Dataset*

Predicts if a bank customer will leave based on their profile.

*Features*:
- CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
  IsActiveMember, EstimatedSalary

*Target*: Exited (0 = stayed, 1 = left)

*Data Preprocessing*

Dropping Irrelevant Columns:
```python
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
```

These columns don't help predict behavior, they are just identifiers.
#pagebreak()

*One-Hot Encoding*

```python
X = pd.get_dummies(X, drop_first=True)
```

**What is one-hot encoding?**
Converts categorical text data into numerical format

**Example:**
```
Geography: France, Germany, Spain

Becomes:
Geography_Germany: 0 or 1
Geography_Spain: 0 or 1
(France is implied when both are 0)
```

*drop_first=True*: Prevents data redundancy, if someone isn't German or Spanish, they must be French.

*Model Architecture*
```python
model = tf.keras.Sequential([
    Dense(64, activation="tanh", input_shape=(X_train.shape[1],)),
    Dense(32, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(1, activation="sigmoid")
])
```

*Layer progression*: 64 #sym.arrow 32 #sym.arrow 16 #sym.arrow 1

- Gradually narrows down to final prediction
- More complex than wine model (more features to process)

*Making predictions on New Customers*

```python
new_customer = {
    'CreditScore': 600,
    'Geography': "France",
    'Gender': "Male",
    'Age': 40,
    # ... etc
}
```
#pagebreak()
*Process*:

1. Create customer profile as dictionary
2. Convert to DataFrame
3. Apply one-hot encoding (``` pd.get_dummies```)
4. Reindex to match training features (ensures correct column order)
5. Scale using the same scaler from training
6. Predict

*Interpretation:*

```python
if prediction > 0.5:
    print("The customer will leave the bank")
```

- Probability > 0.5 = likely to leave
- Probability < 0.5 = likely to stay

== Project 3: Student Grade Prediction

*Dataset*
Predicts student grades (0-7 scale) based on survey responses

*Target*: GRADE (multi-class, 0,1,2,3,4,5,6,7)

*Features*: 30 survery questions about study habits, attendance, etc.

*Multi-Class Classification*

Key difference from binary classification:

```python
num_classes = 8
Dense(num_classes, activation='softmax')
```

*Output layer*

- *8 neurons*: One for each possible grade
- *Softmax activation*: Converts outputs to probabilities that sum to 1.0
- Each neuron represents the probablitu of that specific grade.

*Loss function*

```python
loss='sparse_categorical_crossentropy'
```
- Used ofr multi-class classification when labels are integers (0, 1, 2, ... 7)
- "Sparse" means we don't need to one-hot encode the target variable

*Getting the Final Prediction*

```python
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
```

**Example output:**
```
predictions = [[0.01, 0.05, 0.10, 0.60, 0.15, 0.05, 0.03, 0.01]]
                 0     1     2     3     4     5     6     7
```

- *np.argmax*: Finds the index with highest Probability
- In this case: 3 (60% probability of grade 3)

*Predicting for a New Student*
```python
new_student = np.array([[22, 3, 3, 1, 2, 2, 1, ...]])  # 30 features
```

*Process*:

1. Create array with all 30 survey responses
2. Convert to DataFrame (to match training format)
3. Scale using the same scaler
4. Predict
5. Use ``` np.argmax ```to get the most likely grade.

#pagebreak()

*Key Differences: Binary vs Multi-Class*

#table(
  columns: 3,
  [*Aspect*], [*Binary Classification*], [*Multi-Class Classification*],
  [*Output neurons*], [1], [Number of classes],
  [*Activation*], [Sigmoid], [Softmax],
  [*Loss function*], [Binary crossentropy], [Sparse categorical crossentropy],
  [*Prediction output*], [Single probability (0-1)], [Array of probabilities],
  [*Final prediction*], [`> 0.5` threshold], [`np.argmax()`],
)

*Common Workflow Summary*

1. Load the data with pandas
2. Separate features (X) and target (y)
3. Preprocess:
  - Drop irrelevant columns
  - One-hot encode categorical variables
  - Split into train/test sets
  - Scale features with StandardScaler
4. Build model:
  - Choose architecture based on problem complexity
  - Use sigmoid for binary, softmax for multiclass
5. Train model with validation split to monitor progress
6. Make predictions on new data (Remember to preprocess the same way!)