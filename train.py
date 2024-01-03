import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import codecs, json 
import cv2
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

# Assuming load_data() loads your data correctly
X, y, X_test, y_test = load_data()

X = X.reshape(X.shape[0],-1) / X.max()
# X = np.where(X < 127.5, 0, 1)
y = np.where(y < 0.5, 0, 1)
def resize_image(image, target_size=(64, 64)):
    return cv2.resize(image, target_size)

class LaiModel:
    def __init__(self, X) -> None:
        W = np.random.randn(X.shape[1], 1)
        b = np.random.randn(1)
        self.W = W
        self.b = b


    def init_model(self, XX, WW, bb):
        Z = XX.dot(WW) + bb
        A = 1 / (1 + np.exp(-Z))
        return A

    def execute_log_loss(self, A, y):
        epsilon = 1e-15
        log_loss = 1/len(y) * np.sum(-y*np.log(A + epsilon) - (1-y)*np.log(1-A + epsilon))
        return log_loss

    def train_model(self, A, X, y):
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return dW, db

    def update_tanning(self, dW, db, learning_rate):
        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db

    def execute(self, XX):
        self._A_ = self.init_model(XX, self.W, self.b)
        return self._A_ >= 0.5
    def predict_single_image(self, image_path, y):
        # Charger l'image et la prétraiter
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = resize_image(image)
        flattened_image = resized_image.flatten() / 255.0  # Normaliser l'image

        # Appliquer le modèle pour la prédiction
        prediction = self.execute(flattened_image.reshape(1, -1))

        # Imprimer le résultat
        if prediction[0]:
            print(bcolors.OKGREEN+"[OK]" if y==prediction else bcolors.FAIL+"[ERROR]", "Si l'image est un téléphone : [", y,"] et le neurone a trouvé : ", prediction, "sûr à ", self._A_, "%")
        else:
            print(bcolors.OKGREEN+"[OK]" if y==prediction else bcolors.FAIL+"[ERROR]", "Si l'image est un téléphone : [", y,"] et le neurone a trouvé : ", prediction, "sûr à ", self._A_, "%")
        return flattened_image.reshape(1, -1)
    def init_training(self, X, y, learning_rate=0.1, number_of_cycles=10000):
        Loss = []
        acc = []
        TLoss = []
        Tacc = []
        X_t = X_test.reshape(X_test.shape[0], -1) / X_test.max()

        for i in tqdm(range(number_of_cycles)):
            A = self.init_model(X, self.W, self.b)
            # accuracy = accuracy_score(y, self.execute(X))*100
            # print(f"Accuracy: {accuracy}%")
            if i %10 == 0:
                y_pred = self.execute(X)
                acc.append(accuracy_score(y, y_pred))
                Loss.append(self.execute_log_loss(A, y))

                
                A_t = self.init_model(X_t, self.W, self.b)
                y_pred_test = self.execute(X_t)
                Tacc.append(accuracy_score(y_test, y_pred_test))
                TLoss.append(self.execute_log_loss(A_t, y_test))
            dW, db = self.train_model(A, X, y)
            self.update_tanning(dW, db, learning_rate)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(Loss, label='train loss')
        # plt.plot(TLoss, label='test loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(acc, label='train acc')
        plt.plot(Tacc, label='test acc')
        plt.legend()
        plt.show()
        return self.W, self.b

# Create an instance of the model
LaiINSTANCE = LaiModel(X)

# Train the model
W, b = LaiINSTANCE.init_training(X, y, learning_rate=0.01)
input("Pass tests [ENTER] ")
# # Create an instance of the model
print(bcolors.HEADER+"------------------------------------")
print(bcolors.HEADER+"NOT IN DATABASE IMAGES")
print(bcolors.HEADER+"------------------------------------")
image_to_test_path = "C:/Users/lyamz/Downloads/phone.jpeg"  # Remplacez par le chemin de votre image

# Effectuer la prédiction sur l'image sélectionnée
point = LaiINSTANCE.predict_single_image(image_to_test_path, True)

image_to_test_path = "C:/Users/lyamz/Downloads/house.jpg"  # Remplacez par le chemin de votre image
# image_to_test_path = "C:/Users/lyamz/Downloads/house.jpg"  # Remplacez par le chemin de votre image
point = LaiINSTANCE.predict_single_image(image_to_test_path, False)

print(bcolors.HEADER+"------------------------------------")
print(bcolors.HEADER+"IN DATABASE IMAGES")
print(bcolors.HEADER+"------------------------------------")
# # Create an instance of the model
image_to_test_path = "C:/Users/lyamz/Downloads/Houses-Images/eligible/0x0.jpeg"  # Remplacez par le chemin de votre image

# Effectuer la prédiction sur l'image sélectionnée
point = LaiINSTANCE.predict_single_image(image_to_test_path, False)
image_to_test_path = "C:/Users/lyamz/Downloads/archive/Mobile_image/Mobile_image/Datacluster Labs Phone Dataset (9).jpg"  # Remplacez par le chemin de votre image

# Effectuer la prédiction sur l'image sélectionnée
point = LaiINSTANCE.predict_single_image(image_to_test_path, True)

a = LaiINSTANCE.b
b = LaiINSTANCE.W
bP = b.tolist() # nested lists with same data, indices
aP = a.tolist() # nested lists with same data, indices
file_path = "model_trained.json" ## your path variable
f = open(file_path, "w")
f.write(json.dumps([bP, aP]))
f.close()