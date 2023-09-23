import cv2
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
from datetime import date

name = input("Enter Your Name: ")
id = input("Enter your ID: ")
age = input("Enter your age: ")

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (216, 216), interpolation=cv2.INTER_CUBIC)
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = os.path.join('Images', f'{id}.png')
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': ""
})

ref = db.reference('Patients')

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
current_date = date.today().strftime("%d-%m-%Y")


data = {
    id: {
        "name": name,
        "age": age,
        "checkup" : "Pending",
        "registration_time": current_time,
        "registration_date" : current_date,
        "checkin_time": "null",
        "checkin_date": "null"
    }
}

for key,value in data.items():
    ref.child(key).set(value)
