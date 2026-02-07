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

Why split the data?
- Training set (80%): Data the model learns from.
- Test set (20%): Data the model has never seen - used to check if its actually learned or just memorized.

Kind of how we humans study practice problems ( = _training_ ) and then do a real exam with different questions
( = _testing_) to see if you actually understand the material.

```random_state=42``` ensures we get the same split every time we run the code, which makes the results
reproducible.

