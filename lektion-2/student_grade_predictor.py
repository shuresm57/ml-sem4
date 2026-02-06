import pandas as pd
from google.colab import drive
drive.mount('/content/gdrive')

df = pd.read_excel('/content/gdrive/MyDrive/student_prediction.xls')

df.head()

X = df.drop(columns=['STUDENTID', 'COURSE ID', 'GRADE'])
y = df['GRADE']

X

y

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
from tensorflow.keras.layers import Dense

num_classes = 8
model = tf.keras.Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

import numpy as np

new_student = np.array([[
    22, 3, 3, 1, 2, 2, 1, 1, 1, 2, 3, 2, 1, 2, 5, 3, 2, 2, 1, 1, 1, 1, 1, 3, 2, 1, 2, 1, 1, 1
]])

new_student_df = pd.DataFrame(new_student, columns=X.columns)
new_student_scaled = scaler.transform(new_student_df)
prediction = model.predict(new_student_scaled)
predicted_grade = np.argmax(prediction, axis=1)[0]

print(f"Prediction: {predicted_grade}")

