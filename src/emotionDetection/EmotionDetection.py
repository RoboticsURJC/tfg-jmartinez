import sys
import pickle

sys.path.append('..')
from emotionalMesh.EmotionalMesh2 import EmotionalMesh2

class EmotionDetection:
    def __init__(self):
        # Emotional mesh to extract angles data
        self.emotional_mesh = EmotionalMesh2()

        # Model and PCA to predict
        self.model = None
        self.pca = None
        self.__initialize_model_and_pca()

    def predict(self, image):
        self.emotional_mesh.set_image(image)
        self.emotional_mesh.process()
        self.emotional_mesh.draw_emotional_mesh()
        if self.emotional_mesh.face_mesh_detected():
            angles = self.emotional_mesh.get_angles()
            angles = self.pca.transform(angles)
            pred = self.model.predict(angles)
            return pred
        return 0

    # Private methods ----------------------------------------------------------------------
    def __initialize_model_and_pca(self):
        with open('model/emotionalMesh/model.pkl', 'rb') as modelfile:
            loaded = pickle.load(modelfile)
        self.model = loaded['model']
        self.pca = loaded['pca_fit']