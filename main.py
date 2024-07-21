import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("elonn.png")
img = cv2.resize(img, (800, 500))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=3)
print(faces)
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.imshow("facedetection", img)
plt.imshow(img)
plt.show()

cv2.waitKey(0)

cv2.destroyAllWindows()