import numpy as np

from si.data.dataset import Dataset
from si.model_selection.split import stratified_train_test_split
from si.models.logistic_regression import LogisticRegression

data = np.loadtxt("datasets/breast_bin/breast-bin.csv", delimiter=",")

X = data[:, :-1]
y = data[:, -1].astype(int)

dataset = Dataset(X=X, y=y)

train, test = stratified_train_test_split(
    dataset, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(train)

print("Accuracy:", model.score(test))
print("Cost final:", list(model.cost_history.values())[-1])