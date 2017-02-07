import cv2

#import the haarcade and the video
cascade_src = 'haarcascade_cars.xml'
video_src = 'video1.avi'
cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    #get the frames
    ret, img = cap.read()
    #if not more frames then break
    if not ret:
        break
    #convert to gray the frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #using the haarcascade detect if it is a car
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    #draw a rectangle
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    #show the frame
    cv2.imshow('video', img)
    #if you press esc end the while bucle
    if cv2.waitKey(33) == 27:
        break
#end the program
cv2.destroyAllWindows()
