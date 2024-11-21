#Cria o dataset de imagens para o reconhecimento facial
#Requisitos
#pip install cv2
#pip install pillow
#pip install pymysql

import cv2
import os

def generate_dataset(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.2, 5)
        #scaling_factor = 1.2
        #minimum neighbor = 5
            
        if len(faces) > 1:
            faces = [faces[0]]

        if len(faces) == 0: 
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
            #equalized_face = cv2.equalizeHist(cropped_face)

        return cropped_face
    cap = cv2.VideoCapture(0)
    id = 1
    img_id = 0

    print("Coletando dados...")
    while True:
        frame = cap.read()       
        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = data_path+"/user."+str(id)+"."+str(img_id)+".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Rosto cortado", face)
        if cv2.waitKey(1)==13 or int(img_id)==150: #13 pra caso seja pressionado Enter
            break    
            
    cap.release()
    cv2.destroyAllWindows()
generate_dataset("biometria")