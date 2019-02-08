"""
# Assignment 2
# Demonstration of Keras and Bokeh
# 
# Submitted by: Neeti Jain
# 

"""

# Importing all the necessary Packages
import tensorflow as tf
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from bokeh.plotting import figure, output_file, show ,gridplot
from sklearn.model_selection import train_test_split
import numpy as np


# Load Training and test data.
dataset_train = np.loadtxt('fashion-mnist_train1.csv', delimiter=',',dtype=int ,skiprows=1)
x_train = dataset_train[:,1:]
y_train = dataset_train[:,0]
dataset_test = np.loadtxt('fashion-mnist_test.csv', delimiter=',',dtype=int ,skiprows=1)
x_test = dataset_test[:,1:]
y_test = dataset_test[:,0]

#  Spliting testing and training dataset using train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape) 
print("x_test shape:", x_test.shape, "y_test shape:", y_test.shape) 
print("x_valid shape:", x_valid.shape, "y_valid shape:",y_valid.shape) 

# Reshaping data as 28*28 pixels of images
print(x_train.shape[0])
x_train=np.reshape(x_train, (x_train.shape[0], 28,28))
x_test=np.reshape(x_test, (x_test.shape[0], 28,28))
x_valid=np.reshape(x_valid, (x_valid.shape[0], 28,28))

# Define the text labels
fashion_mnist_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker","Bag", "Ankle boot"]   # index 9

# Normalize the data dimensions so that they are of approximately the same scale.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# # Converts a class vector (integers) to binary class matrix.
# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')


# ## Models in Keras
# There are two types of built-in models available in Keras: sequential models and models created with the functional API. 
# In our model we will be using Sequential Model.
# Sequential
# Sequential models are created using the keras_model_sequential() function and are composed of a set of linear layers:
model = tf.keras.Sequential()

# first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, 
                                 kernel_size=2, 
                                 padding='same', 
                                 activation='relu', #
                                 input_shape=(28,28,1))) 

model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
# Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.
# The effect is that the network becomes less sensitive to the specific weights of neurons. This in turn results in a network that is capable of better generalization and is less likely to overfit the training data.
model.add(tf.keras.layers.Dropout(0.3))
# ## Activation Function
# - ReLU :: The ReLU is the most used activation function in the world right now.Since, it is used in almost all the convolutional neural networks or deep learning. f(z) is zero when z is less than zero and f(z) is equal to z when z is above or equal to zero.

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
# Maxpooling : Take the Maximum value
#strides: Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Dropout(0.3))
# model.layers is a flattened list of the layers comprising the model.
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dropout(0.5))
# A dense layer represents a matrix vector multiplication. (assuming your batch size is 1) The values in the matrix are the trainable parameters which get updated during backpropagation.
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# model.summary() prints a summary representation of your model. Shortcut for utils.print_summary
model.summary()


# Compile  the Model
# An optimizer ,  A loss function ,  A list of metrics
#We use model.compile() to configure the learning process before training the model. This is where you define the type of loss function, optimizer and the metrics evaluated by the model during training and testing.

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


# # Checkpoint Best Neural Network Model Only
# A simpler check-point strategy is to save the model weights to the same file, if and only if the validation accuracy improves.
# In this case, model weights are written to the file “weights.best.hdf5” only if the classification accuracy of the model on the validation dataset improves over the best seen so far.
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)

# We will train the model with a batch_size of 64 and 10 epochs
epochs_count = 10
history = model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=epochs_count,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])


# Load the weights with the best validation accuracy
model.load_weights('model.weights.best.hdf5')


# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'The Test accuracy of the model is ::  ', score[1])


y_hat = model.predict(x_test)

# Visualising the model and data through bokeh

# Actual Image 
index=14      
a = figure(title='Original Image at Location 14',plot_width=400, plot_height=400)
a.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=14, dh=14, palette="Spectral11")
a.title.background_fill_color = "#eef442"
a.border_fill_color = "whitesmoke"

#Predicted Image 
p  = figure( title='Predicted Image at index 14',plot_width=400, plot_height=400)
p.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=2, dh=2, palette="Spectral11")
p.outline_line_width = 9
p.outline_line_alpha = 0.5
p.title.background_fill_color = "#eef442"
p.border_fill_color = "whitesmoke"

predict_index = np.argmax(y_hat[index])
true_index = np.argmax(y_test[index])
p.background_fill_color = ("#33FFBD" if predict_index == true_index else "#FF5733")

# Actual Image 
index=19
b = figure(title='Original Image at index 19',plot_width=400, plot_height=400)
b.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=14, dh=14, palette="Spectral11")
b.title.background_fill_color = "#eef442"
b.border_fill_color = "whitesmoke"

#Predicted Image 
q  = figure( title='Predicted Image  at index 19',plot_width=400, plot_height=400)
q.image(image=[np.flipud(x_test[index+1])], x=0, y=0, dw=2, dh=2, palette="Spectral11")
q.outline_line_width = 9
q.outline_line_alpha = 0.5
predict_index = np.argmax(y_hat[index])
true_index = np.argmax(y_test[index])
q.background_fill_color = ("#33FFBD" if predict_index == true_index+1 else "#FF5733")
q.title.background_fill_color = "#eef442"
q.border_fill_color = "whitesmoke"


# Actual Image 
index=20
c = figure(title='Original Image at index 20',plot_width=400, plot_height=400)
c.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=14, dh=14, palette="Spectral11")
c.title.background_fill_color = "#eef442"
c.border_fill_color = "whitesmoke"

#Predicted Image 
r  = figure( title='Predicted Image at index 20',plot_width=400, plot_height=400)
r.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=2, dh=2, palette="Spectral11")
r.outline_line_width = 9
r.outline_line_alpha = 0.5
predict_index = np.argmax(y_hat[index])
true_index = np.argmax(y_test[index])
r.background_fill_color = ("#33FFBD" if predict_index == true_index else "#FF5733")
r.title.background_fill_color = "#eef442"
r.border_fill_color = "whitesmoke"

# Actual Image 
index=21
d = figure(title='Original Image at index 21',plot_width=400, plot_height=400)
d.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=2, dh=2,palette="Spectral11")
d.title.background_fill_color = "#eef442"
d.border_fill_color = "whitesmoke"

#Predicted Image 
s  = figure( title='Predicted Image at index 21',plot_width=400, plot_height=400)
s.image(image=[np.flipud(x_test[index+1])], x=0, y=0, dw=2, dh=2, palette="Spectral11")
s.outline_line_width = 9
s.outline_line_alpha = 0.5
predict_index = np.argmax(y_hat[index])
true_index = np.argmax(y_test[index])
s.background_fill_color = ("#33FFBD" if predict_index == true_index+1 else "#FF5733")
s.title.background_fill_color = "#eef442"
s.border_fill_color = "whitesmoke"



# Actual Image 
index=55
e = figure(title='Original Image at index 55',plot_width=400, plot_height=400)
e.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=2, dh=2,palette="Spectral11")
e.title.background_fill_color = "#eef442"
e.border_fill_color = "whitesmoke"

#Predicted Image 
t  = figure( title='Predicted Image at index 55',plot_width=400, plot_height=400)
t.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=2, dh=2, palette="Spectral11")
t.outline_line_width = 9
t.outline_line_alpha = 0.5
predict_index = np.argmax(y_hat[index])
true_index = np.argmax(y_test[index])
t.background_fill_color = ("#33FFBD" if predict_index == true_index else "#FF5733")
t.title.background_fill_color = "#eef442"
t.border_fill_color = "whitesmoke"

# show Accuracy and Loss graph wrt epochs
colors_list = ['green', 'red']
legends_list = ['Test Accuracy', 'Validation Accuracy']

xs=[history.history['acc'], history.history['val_acc']]
ys=[ range(0,epochs_count), range(0,epochs_count)]
acc = figure(plot_width=400, plot_height=400 ,title='Accuracy Plot')
for (colr, leg, x, y ) in zip(colors_list, legends_list, xs, ys):
    my_plot = acc.line(x, y, color= colr, legend= leg)
acc.title.background_fill_color = "#eef442"
acc.border_fill_color = "whitesmoke"
   
legends_list2 = ['Test Loss', 'Validation Loss']
xs_loss=[history.history['loss'], history.history['val_loss']]
ys_loss=[ range(0,epochs_count), range(0,epochs_count)]
val = figure(plot_width=400, plot_height=400 ,title='Loss Plot')
for (colr, leg, x, y ) in zip(colors_list, legends_list2, xs_loss, ys_loss):
    my_plot = val.line(x, y, color= colr, legend= leg)
val.title.background_fill_color = "#eef442"
val.border_fill_color = "whitesmoke"

# Show scatter Plot
s1 = figure(plot_width=400, plot_height=400,title= "Test Data Scatter Plot (200 values) ")
u=list()
for i in range(0,200):
    u.append (np.argmax(y_test[i]))
s1.circle(u, range(0,200), size=2, color="red", alpha=1)
s1.xaxis.ticker = [0,1,2,3,4,5,6,7,8,9]
s1.yaxis.axis_label = "Image Indexes"
s1.xaxis.axis_label = " Different Categories/classes"
s1.title.background_fill_color = "#eef442"
s1.border_fill_color = "whitesmoke"


s2 = figure(plot_width=400, plot_height=400,title= "Validation Data Scatter Plot (200 values) ")
v=list()
for i in range(0,200):
    v.append (np.argmax(y_test[i]))
s2.circle(v, range(0,200), size=2, color="red", alpha=1)
s2.xaxis.ticker = [0,1,2,3,4,5,6,7,8,9]
s2.yaxis.axis_label = "Image Indexes"
s2.xaxis.axis_label = " Different Categories/classes"
s2.title.background_fill_color = "#eef442"
s2.border_fill_color = "whitesmoke"


h = gridplot([[a,p],[b,q],[c,r],[d,s],[e,t], [s1 ,s2 ],[acc, val]] )
output_file('bokeh.html',)
show(h)


