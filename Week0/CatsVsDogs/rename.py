import os

print('Renaming files in training set')
Categories=['Dog','Cat']
#Follow the same project hierachy so that everything works straight out of the box
#Run this script from within the train directory to rename files there from 0 to 
DataDir='C:\\Users/Jidin/Downloads/PetImages/train'
path = DataDir+"/"+category
for i in os.listdir():
    old=i
    i=i.split('.')[1]
    a=int(5-len(i))*'0'+i+'.jpg'
    print(a)
    os.rename(old,a)
