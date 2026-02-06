import pandas as pd

url = "https://raw.githubusercontent.com/joneikholmkea/machine-learning/main/spring26/lektion2/customer_staying_or_not.csv"
df = pd.read_csv(url)

df.head()

df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
X = df.drop(columns=["Exited"])

y = df["Exited"]

X = pd.get_dummies(X, drop_first=True)

feature_columns = X.columns

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation

model = tf.keras.Sequential([
    Dense(64, activation="tanh", input_shape = (X_train.shape[1],)),
    Dense(32, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(
    X_train,
    y_train,
    epochs=200,
    validation_split=0.2,
    verbose=1
)

import numpy as np

new_customer = {
    'CreditScore' : 600,
    'Geography' : "France",
    'Gender' : "Male",
    'Age' : 40,
    'Tenure' : 3,
    'Balance' : 60000,
    'NumOfProducts' : 2,
    'HasCrCard' : 1,
    'IsActiveMember' : 1,
    'EstimatedSalary' : 50000
}

new_customer_df = pd.DataFrame([new_customer])

new_customer_encoded = pd.get_dummies(new_customer_df, drop_first=True)

new_customer_encoded = new_customer_encoded.reindex(columns=feature_columns, fill_value=0)

new_customer_scaled = scaler.transform(new_customer_encoded)

prediction = model.predict(new_customer_scaled)[0][0]

if prediction > 0.5:
    print("The customer will leave the bank")
    print(f"Probability: {prediction}")
else:
    print("The customer will not leave the bank")