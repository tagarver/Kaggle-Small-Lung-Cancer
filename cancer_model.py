#cancer_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import numpy as np
import tensorflow as tf

data = pd.read_csv("Features_Train.csv")
data = data.dropna()
features = data.drop(columns = ["Outcome"])
features = pd.get_dummies(features, drop_first=True)
target = data["Outcome"]
global x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

def visualize(data):
	import seaborn as sns
	import matplotlib.pyplot as plt
	
	target = "Outcome"
	features = data.drop(columns = [target])
	labels = data[target]
	
	sns.countplot(x = labels)
	plt.title("Cancer Distributions")
	plt.show()
	
	plt.figure(figsize = (10,8))
	sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
	plt.title("Feature Distribtution")
	plt.show()
	
	
def cancer_model(hidden_units=128, dropout_rate=0.2, learning_rate=0.001, l2_reg=0.01):
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, BatchNormalization, Conv1D, Flatten, Attention, GlobalMaxPooling1D, LeakyReLU
	from tensorflow.keras.optimizers import Nadam, Adamax, Lion, Ftrl, RMSprop, Adagrad
	from sklearn.ensemble import GradientBoostingClassifier
	
	gbm_model = RandomForestClassifier(n_estimator=100, random_state = 42)
	gbm_model.fit(x_train, y_train)
	
	gbm_train_predictions = gbm_model.predict_proba(x_train)[:,1].reshape(-1,1)
	gbm_test_predictions = gbm_model.predict_proba(x_test)[:,1].reshape (-1,1)
	
	x_train_ran = np.hstack((x_train, gbm_train_predictions))
	x_test_ran = np.hstack((x_test, gbm_test_predictions))
	x_train_ran = np.array(x_train_ran, dtype=np.float32)
	x_test_ran = np.array(x_test_ran, dtype=np.float32)


	model = Sequential([
		Dense(hidden_units, activation = 'relu', input_shape = (x_train_ran.shape[1],)),
		Dense(hidden_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
		Dropout(dropout_rate),
		BatchNormalization(),
		Dense(hidden_units // 2),
		LeakyReLU(),
		Dropout(dropout_rate),
		
		Dense(hidden_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
		Dropout(dropout_rate),
		BatchNormalization(),
		Dense(hidden_units // 2),
		LeakyReLU(),
		Dropout(dropout_rate),
		
		Dense(hidden_units, activation='relu'),
		Dropout(dropout_rate),
		BatchNormalization(),
		Dense(hidden_units // 2),
		LeakyReLU(),
		Dropout(dropout_rate),
		
		Dense(hidden_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
		Dense(hidden_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
		Dropout(0.2),
		BatchNormalization(),
		
		Dense(1, activation='sigmoid')
		])

	optimizer = RMSprop(learning_rate = learning_rate)
	model.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
	
	
	return model, x_train_ran, x_test_ran
	
# Defining Optuna objective function
def opjective(trial):
	hidden_units = trial.suggest_int('hidden_units', 32, 256)
	dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
	learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3)
	batch_size = trial.suggest_int('batch_size', 32, 64)
	epochs = trial.suggest_int('epochs', 10, 1000)
	l2_reg = trial.suggest_float('l2_reg', 0.0, 0.2)
	
	model, x_train_ran, x_test_ran = cancer_model(hidden_units=hidden_units, dropout_rate=dropout_rate, learning_rate=learning_rate)

	history = model.fit(x_train_ran, y_train,
		validation_data = (x_test_ran, y_test),
		batch_size=batch_size,
		epochs=epochs,
		verbose = 1)
		
	accuracy = history.history['val_accuracy'][-1]
	return accuracy
	
def optimize_run():
	study = optuna.create_study(direction='maximize')
	study.optimize(opjective, n_trials = 30)
	
	print("Best Hyperparameters:", study.best_params)
	
	best_params = study.best_params
	
	final_model, x_train_ran, x_test_ran = cancer_model(hidden_units = best_params['hidden_units'],
								dropout_rate = best_params['dropout_rate'],
								learning_rate = best_params['learning_rate'])
	
	final_model.fit(x_train_ran, y_train,
			validation_data = (x_test_ran, y_test),
			batch_size = best_params['batch_size'],
			epochs = best_params['epochs'],
			verbose = 1)
			
	test_loss, test_accuracy = final_model.evaluate(x_test, y_test)
	print(f"Test Accuracy: {test_accuracy}")
	
	final_model.save('optimized_lung_cancer_model.keras')
	print("Model saved as optimized_lung_cancer_model.keras")
	
	
	
	
#visualize(data)
optimize_run()
