import sys
sys.path.append('..')
from emotionalMesh.EmotionalMesh2 import EmotionalMesh2
import cv2
import glob as gb
import joblib

stream = cv2.VideoCapture(1)
emotionalmesh = EmotionalMesh2(static=False, max_num_faces=1)

# Load the model
model = joblib.load('model/model.pkl')

while(True):
    grabbed, image = stream.read()
    if not grabbed:
        break

    image = cv2.flip(image, 1)

    emotionalmesh.set_image(image)
    emotionalmesh.process()
    if emotionalmesh.face_mesh_detected():
        angles = emotionalmesh.get_angles()
        print(model.predict(angles))


    # Show image
    cv2.imshow("imagen", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cv2.destroyAllWindows()