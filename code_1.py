import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
 

video_capture = cv2.VideoCapture(0) # 0 means the webcam number
#Load know faces
dc_pic=face_recognition.load_image_file("myface\dcpic.jpg")
dc_encoding=face_recognition.face_encodings(dc_pic)[0]      #converts string to binary bits (encoding) and returns a list

as_pic=face_recognition.load_image_file("myface\pic_2.jpg")
as_encoding=face_recognition.face_encodings(as_pic)[1]

know_face_encodings=[dc_encoding,as_encoding]
known_face_names=["Devottam","Ashish"]

# list of students
students=known_face_names.copy()
face_locations=[]   #locations for searching images
face_encodings=[]

# get the current date and time
now=datetime.now()
current_date=now.strftime("%Y-%m-%d")

f=open(f"{current_date}.csv","w+",newline="")   
lnwriter=csv.writer(f)

while True:
    _, frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognise Faces
    face_locations=face_recognition.face_locations(rgb_small_frame)
    face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
    for face_encoding in face_encodings:
        matches=face_recognition.compare_faces(know_face_encodings, face_encoding)
        face_distance=face_recognition.face_distance(know_face_encodings, face_encoding)
        best_match_index=np.argmin(face_distance)

        if(matches[best_match_index]):
            name=known_face_names[best_match_index]
        # add text if person is present
        if name in known_face_names:
            font=cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText=(10,100)
            fontscale=1.5
            fontColor=(255,0,0)
            thickness=3
            linetype=2
            cv2.putText(frame,name+"Present",bottomLeftCornerOfText,font,fontscale,fontColor,thickness,linetype)

            if name in students:
                students.remove(name)
                current_time=now.strftime("%H-%M%S")
                lnwriter.writerow([name,current_time])

    cv2.imshow("Attendance",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()