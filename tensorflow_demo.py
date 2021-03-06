"""Code extracted from de Jupyter notebook, if you want to check the notes, run dthe notebook instead"""
import tensorflow as tf
tf.config.list_physical_devices('GPU')
#Show the available GPUs on the system
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print(predictions)

print(tf.nn.softmax(predictions).numpy())
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print(loss_fn(y_train[:1], predictions).numpy())
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
print(model.evaluate(x_test,  y_test, verbose=2))

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
print(probability_model(x_test[:5]))