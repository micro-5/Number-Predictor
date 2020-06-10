import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
print(y_train[0])
print(x_train[0])

x_train = x_train/255.0
x_test = x_test/255.0

class myCallBack(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, log={}):
    if(log.get('accuracy') > 0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallBack()


model = tf.keras.models.Sequential([

tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation = tf.nn.relu),
tf.keras.layers.Dense(10, activation = tf.nn.softmax)

])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs = 10, callbacks=[callbacks])
model.evaluate(x_test, y_test)
