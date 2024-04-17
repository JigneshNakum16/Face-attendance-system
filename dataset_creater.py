import cv2
import numpy as np
import sqlite3
import os

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

def insertToUpdate(id, name, age):
    try:
        conn = sqlite3.connect("sqlite.db")
        cursor = conn.execute("SELECT * FROM students WHERE id = ?", (id,))
        isRecordExists = cursor.fetchone()

        if isRecordExists:
            conn.execute("UPDATE students SET name = ?, age = ? WHERE id = ?", (name, age, id))
        else:
            conn.execute("INSERT INTO students (id, name, age) VALUES (?, ?, ?)", (id, name, age))

        conn.commit()
    except sqlite3.Error as e:
        print("SQLite error:", e)
    finally:
        conn.close()

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

id = int(input("Enter the ID: "))
name = input("Enter the name: ")
age = int(input("Enter the age: "))

insertToUpdate(id, name, age)

dataset_directory = "dataset"
create_directory(dataset_directory)

sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite(os.path.join(dataset_directory, f"user.{id}.{sampleNum}.jpg"), gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    
    if sampleNum > 20:
        break

cam.release()
cv2.destroyAllWindows()
