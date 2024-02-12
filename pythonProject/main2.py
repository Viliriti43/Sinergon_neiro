from imutils import paths
import face_recognition
import pickle
import cv2
import os


imagePaths = list(paths.list_images('Images'))
knownEncodings = []
knownNames = []
print("ff")
for (i, imagePath) in enumerate(imagePaths):
    print(i,imagePath)


    name = imagePath.split(os.path.sep)[-2]
    print(name)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for en in encodings:
        knownEncodings.append(en)
        knownNames.append(name)
data = {"encodings": knownEncodings, "names": knownNames}
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()
