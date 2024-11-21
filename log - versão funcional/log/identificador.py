import cv2
import numpy as np
from PIL import Image # type: ignore
import os

def desenha_borda(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)

        id, pred = clf.predict(gray_img[y:y+h,x:x+w])
        confidence = int(100*(1-pred/300))

        if confidence>70:
            if id==1:
                cv2.putText(img, "Pessoa 1", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            elif id==2:
                cv2.putText(img, "Pessoa 2", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            elif id==0:
                cv2.putText(img, "Juan", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "Nome não cadastrado", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "DESCONHECIDO", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    return img

#carregando classificador
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classificador.xml")

video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    img = desenha_borda(img, faceCascade, 1.3, 6, (255,255,255), "Face", clf)
    cv2.imshow("face Detection", img)

    if cv2.waitKey(1)==13:
        break
video_capture.release()
cv2.destroyAllWindows()