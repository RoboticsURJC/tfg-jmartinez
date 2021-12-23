from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.resolution = (640, 480)
numPhotos = 8

camera.start_preview()
sleep(5)
print("Empiezo a hacer fotos!")
for i in range(numPhotos):
    filename = "chess_board/"+str(i)+".png"
    camera.capture(filename, format='png', use_video_port=True)
    print("Capturo la foto: "+str(i))
    sleep(1)
camera.stop_preview()