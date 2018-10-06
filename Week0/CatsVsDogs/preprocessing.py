import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def resizing_training_images():
    count = 0
    for category in Categories:
        path = DataDir+"/"+category
        class_num=Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
                count+=1
                if(count%100==0):
                    print('Resized',count,'images')
            except Exception as e:
                print(e)
        count = 12500


    print("Resized training set and assigned labels to the images")
    print("Shuffling of training data....done")
    import random
    random.shuffle(training_data)



def pickling():          




    x_train=[]
    y_train=[]

    for features,label in training_data:
        x_train.append(features)
        y_train.append(label)
        
    x=np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)



    import pickle
    pickle_out=open("x_train.pickle","wb")
    pickle.dump(x,pickle_out)
    pickle_out.close()

    pickle_out=open("y_train.pickle","wb")
    pickle.dump(y_train,pickle_out)
    pickle_out.close()
    print("Pickling done")






IMG_SIZE=50
training_data=[]
Categories=['Dog','Cat']
DataDir='C:\\Users/Jidin/Downloads/PetImages/train'

print('Resizing Training Images')
resizing_training_images()


print('Pickling training Data')
pickling()


            
            




