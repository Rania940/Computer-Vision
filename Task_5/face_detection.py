import cv2
from cv2 import CascadeClassifier

def face_detection(img):
    # Create the haar cascade
    faceCascade = CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image by using the facecascade class and use detectMultiscale method
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,  minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE)


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return len(faces),image     

if __name__ == '__main__':
    img='pic\\faces.jpg'
    num_of_faces,image=face_detection(img)
    print("detect: "+str(num_of_faces)+" faces")
    cv2.imshow("Faces_detected", image)
    cv2.waitKey(0)