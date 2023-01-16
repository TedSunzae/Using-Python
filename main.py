import cv2
import numpy as np
import face_recognition

imgSlebew = face_recognition.load_image_file('gambar/Thomasshelby.jpg')
imgSlebew = cv2.cvtColor(imgSlebew,cv2.COLOR_BGR2RGB)
imgTestSlebew = face_recognition.load_image_file('gambar/OIP.jpg')
imgTestSlebew = cv2.cvtColor(imgTestSlebew,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgSlebew)[0]
encodeSlebew = face_recognition.face_encodings(imgSlebew)[0]
cv2.rectangle(imgSlebew,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255,0),2)

facelocTest = face_recognition.face_locations(imgTestSlebew)[0]
encodeTestSlebew = face_recognition.face_encodings(imgTestSlebew)[0]
cv2.rectangle(imgTestSlebew,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255,0),2)

results = face_recognition.compare_faces([encodeSlebew],encodeTestSlebew)
faceDis = face_recognition.face_distance([encodeSlebew],encodeTestSlebew)
print(results,faceDis)
cv2.putText(imgSlebew,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Thomas Shelby',imgSlebew)
cv2.imshow('Thomas Test',imgTestSlebew)
cv2.waitKey(0)