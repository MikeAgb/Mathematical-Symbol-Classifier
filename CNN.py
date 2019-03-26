# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# pip install tensorflow
# pip install --upgrade keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

#Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (45, 45, 1), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 100, activation = 'relu'))
classifier.add(Dense(units = 75, activation = 'softmax'))

# Compiling
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

#apply some image augmentation
train_datagen = ImageDataGenerator(rescale = 1./45,
                                   zoom_range = 0.1
                                   )

#rescale image
test_datagen = ImageDataGenerator(rescale = 1./45)

training_set = train_datagen.flow_from_directory('/path',
                                                 target_size = (45, 45),
                                                 batch_size = 64,
                                                 class_mode = 'categorical',
                                                 shuffle=True,
                                                 color_mode='grayscale')

test_set = test_datagen.flow_from_directory('/path',
                                            target_size = (45, 45),
                                            batch_size = 64,
                                            class_mode = 'categorical',
                                            color_mode='grayscale')

#get the labels for classification
labels = test_set.class_indices
labels = {v: k for k, v in labels.items()} 
x = classifier.fit_generator(training_set,
                         steps_per_epoch = 64,
                         epochs = 64,
                         validation_data = test_set,
                         validation_steps = 200)
                         
                         
