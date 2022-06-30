import cv2

input = cv2.VideoCapture(0)
output = cv2.VideoWriter("recordings/video.avi",cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))

while (input.isOpened()):
    ret, image = input.read()
    if ret == True:
        image = cv2.flip(image, 1)
        cv2.imshow('video', image)
        output.write(image)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    else: 
        break

input.release()
output.release()
cv2.destroyAllWindows()