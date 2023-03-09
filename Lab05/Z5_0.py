import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_train.shape[1],
                         X_train.shape[2], 1))

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=32)

print(history.history)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('No. of epoch')
plt.legend(['train', 'validation'])
plt.savefig("Z5_1A.png")

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('No. of epoch')
plt.legend(['train', 'validation'])
plt.savefig("Z5_1L.png")

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)



