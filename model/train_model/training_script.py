import pandas as pd

from model.train_model.trainer_model. import trainer_model

# ==========================
# Load Working Training Data
# ==========================
training_data = pd.read_csv("data/training_data.csv")

# ==============
# Training Model
# ==============
trainer_model(training_data)
