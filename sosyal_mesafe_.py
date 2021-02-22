import cv2
import imutils
import numpy as np 
import argparse 
import math

def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.03)
    
    person = 1
    minSpace = 0
    people= []

    for x, y, w, h in bounding_box_cordinates:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        minSpace += w 
        z=x+(w/2)
        t=y+(h/2)
        people.append((z,t))
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1
    minSpace = 2*(minSpace/person)
    nears=[]

    for i,p1 in enumerate(people):
        for j,p2 in enumerate(people[i+1:]):
            if  math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))<minSpace:
         
                nears.append((i,j+i+1))
    
    for (i,j) in nears:
      
        (x,y,w,h) = bounding_box_cordinates[i]
        cv2.rectangle(frame, (x, y),(x+w,y+h), (0, 0, 255), 2)
        (x,y,w,h) = bounding_box_cordinates[j]
        cv2.rectangle(frame, (x, y),(x+w,y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Social Distance Exceeded', (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0,0), 2) 
        cv2.putText(frame, 'Warning:', (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2) 
        cv2.putText(frame, 'Social Distance Exceeded!!!', (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2) 

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('output', frame)
    return frame


def detectByPathVideo(path):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return
    print('Detecting people...')

    while video.isOpened():
        check, frame = video.read()

        if check:
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            frame = detect(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()


path = "test.mp4"
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

detectByPathVideo(path)
