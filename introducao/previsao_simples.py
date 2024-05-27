import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Dados de exemplo
X = np.array([i for i in range(10)], dtype=float)
Y = np.array([i*2 for i in range(10)], dtype=float)

# Cria um modelo simples
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

# Compila o modelo
model.compile(optimizer='sgd', loss='mean_squared_error')

# Treina o modelo
model.fit(X, Y, epochs=50)

# Suponha que temos dados de teste:
X_test = np.array([i for i in range(15,20)], dtype=float)
Y_test = np.array([i*2 for i in range(15,20)], dtype=float)

# Avalia o modelo nos dados de teste
loss = model.evaluate(X_test, Y_test)
print(f"Perda nos dados de teste: {loss}")

# Usar o modelo para fazer previsões
predictions = model.predict(X_test)
print("Previsões:", predictions.flatten())  # 'flatten' para tornar a saída mais legível




plt.figure(figsize=(10, 5))
plt.scatter(X_test, Y_test, color='blue', label='Valores Reais')
plt.scatter(X_test, predictions.flatten(), color='red', marker='x', label='Previsões')
plt.title('Comparação de Valores Reais e Previsões')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
