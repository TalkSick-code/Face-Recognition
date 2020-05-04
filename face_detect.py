import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0) #Uses Webcam as Camera. Replace with filename or url to capture from a Video

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #HaarCascade Classifier for identifying faces

name = input("Enter Your Name : ") #Captured images will be saved with this name as label to be used for recognition
count = int(input("Enter number of photos : ")) #Number of pictures to be captured
images = [] #Stores reshaped images

while True:
    retval , image = cap.read()

    if retval:
        faces = classifier.detectMultiScale(image)  #detects faces from the video
        cv2.imshow("My Camera", image) #Opens camera and displays image read from video

        if len(faces) > 0:
            sorted_faces = sorted(faces, key=lambda item: item[2]*item[3]) #sorts faces on the basis of area

            x, y, w, h = sorted_faces[-1] #Largest face

            cut = image[y:y+h, x:x+w] #Chops the face out of image
            resized = cv2.resize(cut, (100,100))


            cv2.imshow("Chopped Face" , resized)


    key = cv2.waitKey(1) & 0xFF 

    if key == ord("q"): # 'q' key closes the camera
        break

    elif key == ord("c"): # 'c' key captures the image and stores in images list
        #mean about axis = 2 to convert image to grayscale from RGB (reduces size, faster recognition)
        images.append(resized.mean(axis=2).flatten())  # (1, 10000) new shape instead of (100,100,3)
        print(count)
        count -= 1
        if count == 0:
            break


cap.release()
cv2.destroyAllWindows()

X = np.array(images)
y = np.full((X.shape[0],1), name)

data = np.hstack([y,X]) #Stacks name of image and image data in a single numpy array


#Storing data. If file already exists, overwrites it keeping the previous data
if os.path.exists("faces.npy"):
    old_data = np.load("faces.npy")
    data = np.vstack([old_data, data])

np.save("faces.npy", data)

print(np.array(data).shape) # [0] represents the number of images stored




