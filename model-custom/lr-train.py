import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

MODEL_PATH = env_var = os.environ["MODEL_PATH"]

np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

reg =LinearRegression().fit(train_x.reshape(-1,1), train_y.reshape(-1,1))

print("Model trained successfully")

r2 = r2_score(test_y, reg.predict(test_x.reshape(-1,1)))

print("Model Score:", r2)

np.save(MODEL_PATH, mymodel)
