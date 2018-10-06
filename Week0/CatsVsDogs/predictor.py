import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES = ["Dog", "Cat"]



def prepare(filepath):
    IMG_SIZE = 50 
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



print('Loading trained model')
model = tf.keras.models.load_model("Cats_Dogs.model")
print('There are 12501 images in test set')
print ('Select an image between 1-12500')
i=input()
file="C:\\Users/Jidin/Downloads/PetImages/test1/"+ i +".jpg"


img_array=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img_array)


prediction = model.predict([prepare(file)])

print("^PREDICTION^")
print(CATEGORIES[int(prediction[0][0])])



cv2.waitKey(0)& 0xFF ==ord('q')
cv2.destroyAllWindows()
