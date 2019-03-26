from keras.models import model_from_json
from keras.models import load_model
import numpy as np


confusion_test = test_datagen.flow_from_directory('/path',
                                            target_size = (45, 45),
                                            batch_size = 1,
                                            class_mode = 'categorical',
                                            color_mode='grayscale',
                                            shuffle=False)
                                            
   
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_num.h5")
print("Loaded model")

loaded_model.save('model_num.hdf5')
loaded_model = load_model('model_num.hdf5')

#replace num_samples by number of sample images
Y_pred = loaded_model.predict_generator(confusion_test, 'num_samples')
answers  = np.argmax(Y_pred , axis=1)

#create classification report
predicted_labels=np.argmax(Y_pred, axis=1)
print(predicted_labels)
true_labels = confusion_test.classes
print(true_labels)
from sklearn.metrics import classification_report
print(classification_report(true_labels, predicted_labels))
