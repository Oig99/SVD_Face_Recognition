import numpy as np
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Classe responsabile di:
    - Caricare il dataset Olivetti Faces
    - Effettuare il flatten delle immagini
    - Centrare i dati (passaggio fondamentale per SVD)
    - Effettuare lo split training/test
    """
    def __init__(self, test_size=0.25, random_state=42, min_faces_per_person=60, resize=0.4):

        #  Parametri per suddivisione dati
        self.test_size = test_size
        self.random_state = random_state
        self.min_faces_per_person = min_faces_per_person
        self.resize = resize

        # Variabili che verranno popolate successivamente
        self.X = None               # immagini 64x64
        self.X_flat = None          # immagini flatten (400 x 4096)
        self.X_centered = None      # dati centrati
        self.y = None               # etichette (ID persone)
        self.mean_face = None       # volto medio

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def load_olivetti_data(self):
        """
        Carica il dataset Olivetti.
        Restituisce:
        - X: immagini originali
        - X_flat: immagini vettorializzate
        - y: etichette
        """
        olivetti_data = fetch_olivetti_faces(random_state=self.random_state, shuffle=True)
        self.X = olivetti_data.images
        self.y = olivetti_data.target
        self.X_flat = self.X.reshape((self.X.shape[0], -1))
        return self.X, self.X_flat, self.y

    def center_data(self):
        """
        Esegue la centratura dei dati: X_centered = X - media

        Questo è fondamentale per:
        - Garantire che la SVD catturi la varianza reale
        - Ottenere componenti principali corrette
        """
        print("Shape di self.X_flat:", getattr(self.X_flat, "shape", None))
        self.mean_face = np.mean(self.X_flat, axis=0)
        self.X_centered = self.X_flat - self.mean_face
        return self.X_centered

    def dataset_splitting(self, X_reduced):
        """ Divide i dati ridotti in training e test set. """
        if self.y is None:
            raise ValueError("Le etichette non sono state caricate.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_reduced,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def load_lfw_data(self):
        lfw = fetch_lfw_people(
            min_faces_per_person=self.min_faces_per_person,
            resize=self.resize,
            color=False  # grayscale per coerenza con Olivetti
        )

        self.X = lfw.images
        self.y = lfw.target

        # flatten
        self.X_flat = self.X.reshape((self.X.shape[0], -1))

        return self.X, self.X_flat, self.y



