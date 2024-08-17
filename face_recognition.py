import cv2
import numpy as np
import os

# KNN CODE #
def distance(v1, v2):
    #Euclidian
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    dist = []
    for i in range (train.shape[0]):
        #Get the vector and Lable
        ix = train[i, :-1]
        iy = train[i, -1]
        #Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
        #######
    #sort based on distance and get the top k
    dist = sorted(dist, key=lambda x:x[0] ) [:k]

    #Retrieve the labels of the top k elements
    labels = np.array([d[1] for d in dist])
    # Perform the majority voting
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
dataset_path = "./face_dataset/"
face_data = []
labels = []
class_id = 0          # har ek file ko label dega
name = {}             # mapping between id and name

# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # Create a mapping between class_id and name
        name[class_id] = fx[:-4]
        data_item = np.load(os.path.join(dataset_path, fx))
        face_data.append(data_item)

        # Create Labels for the class
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)   
        class_id += 1


#Concatenate only if data is empty
if face_data and labels:
     face_dataset = np.concatenate(face_data, axis=0)
     face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
     print(face_labels.shape)
     print(face_dataset.shape)

trainset = np.concatenate((face_data, face_labels), axis=1)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    
    ret, frame = cap.read()
    
    if ret == False:
        continue
    #CONVERT FRAME TO GRAYSCALE
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detect multiple faces in the image from classifier
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    for face in faces:
        x, y, w, h = face
        offset = 5

        #safeguard for out-of-bound error
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(frame.shape[1], x + w + offset), min(frame.shape[0], y + h + offset)
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        #perform KNN classification
        out = knn(trainset, face_section.flatten())
        #Display the prediction on the screen
        predicted_name = name[int(out), "Unknown"]

        #draw rectangle in the original image
        cv2.putText(frame, name[int(out)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    cv2.imshow("Faces", frame)  # Display the frame with detected faces
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()
