import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.load("faces.npy") #loads images data from stored file

X, y = data[:, 1:], data[:, 0] #X stores image's data, y stores name label
model = KNeighborsClassifier()
model.fit(X, y)

cap = cv2.VideoCapture(0) #Uses Webcam as Camera. Replace with filename or url to capture from a Video
font = cv2.FONT_HERSHEY_DUPLEX
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    retval, image = cap.read()

    if retval:
        faces = classifier.detectMultiScale(image) #detects faces from the video

        if len(faces) > 0:
            sorted_faces = sorted(faces, key=lambda item: item[2]*item[3]) #sorts faces on the basis of area

            x, y, w, h = sorted_faces[-1] #Largest face in frame


            cut = image[y:y+h, x:x+w] #Chops the face out of image
            resized = cv2.resize(cut, (100,100))
            y_test = resized.mean(axis=2).flatten() #Face data flattened to compare with faces.npy data

            name = model.predict([y_test])[0] #Predicts the face's name by K-Nearest Neighbors Classification

            cv2.rectangle(image, (x, y), (x + w, y + h), color=(88, 92, 173), thickness=3) #Encloses the face in a rectangle
            cv2.putText(image, str(name), (x, y - 10), font, fontScale=1, color=(255, 255, 255)) #Puts name of the detected face over rectangle

        cv2.imshow("My Camera", image) #Displays the image

    key = cv2.waitKey(1)

    if key == ord("q"): #'q' key closes the camera
        break

cap.release()
cv2.destroyAllWindows()




