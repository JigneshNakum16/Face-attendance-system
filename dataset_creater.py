import cv2      #OpenCv camera
import numpy as np
import sqlite3


faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)


def insertToUpdate(id,name,age):
    conn = sqlite3.connect("sqlite.db")
    cmd = "select * from students where id = " + str(id)

    cursor = conn.execute(cmd)
    isRecordExits = 0

    for row in cursor:
        isRecordExits = 1
    
    if(isRecordExits):
        conn.execute("update students set name = ? where id = ?",(name,id))
        conn.execute("update students set age = ? where id = ?",(age,id))
    else:
        conn.execute("insert into students (id,name,age) values (?, ?, ?)",(id,name,age))


    conn.commit()
    conn.close()

id = input("Enter the ID : ")
name = input("Enter the name : ")
age  = input("Enter the age : ")

insertToUpdate(id,name,age)


#detect face in web camera coding

sampleNum = 0
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite("/dataset/user."+str(id) + "."+ str(sampleNum) + ".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Face",img)
    cv2.waitKey(1)
    if(sampleNum>20):
        break

cam.release()
cv2.destroyAllWindows()