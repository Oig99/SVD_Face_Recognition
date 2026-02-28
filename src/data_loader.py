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
    def __init__(self, test_size=0.25, random_state=42, min_faces_per_person=60, resize=0.4, min_faces_unknow=1, n_unknown_classes=8):

        #  Parametri per suddivisione dati
        self.test_size = test_size
        self.random_state = random_state
        self.min_faces_per_person = min_faces_per_person
        self.resize = resize

        self.min_faces_unknow = min_faces_unknow
        self.n_unknown_classes = n_unknown_classes

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

        # unknow classes
        self.X_ud = None
        self.y_ud = None
        self.X_flat_ud = None


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

    def load_lfw_data(self):
        """
        Carica il dataset LFW.
        Restituisce:
        - X: immagini originali
        - X_flat: immagini vettorializzate
        - y: etichette
        """
        lfw = fetch_lfw_people(
            min_faces_per_person=self.min_faces_per_person,
            resize=self.resize,
            color=False
        )

        self.X = lfw.images
        self.y = lfw.target

        # flatten
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

    def dataset_lfw_unknow_detection(self):
        """
        Carica volti unknown: solo classi NON presenti nel dataset originale (load_lfw_data).
        Dove: min_faces_unknow soglia minima immagini per persona nel dataset esteso
              n_unknown_classes: quante classi unknown selezionare (None = tutte)
        """
        if self.y is None:
            raise ValueError("Carica prima il dataset principale con load_lfw_data().")

        lfw_ud = fetch_lfw_people(
            min_faces_per_person=self.min_faces_unknow,
            resize=self.resize,
            color=False
        )

        self.X_ud = lfw_ud.images
        self.y_ud = lfw_ud.target
        self.X_flat_ud = self.X_ud.reshape((self.X_ud.shape[0], -1))

        # --- Filtra solo classi NON presenti nel dataset originale ---
        known_classes = set(np.unique(self.y))
        all_ud_classes = np.unique(self.y_ud)
        unknown_classes = np.array([c for c in all_ud_classes if c not in known_classes])

        print(f"\n{'=' * 55}")
        print(f"  BENCHMARK UNKNOWN DETECTION DATASET")
        print(f"{'=' * 55}")
        print(f"  Classi nel dataset originale (min={self.min_faces_per_person}) : {len(known_classes)}")
        print(f"  Classi totali nel dataset esteso (min={self.min_faces_unknow}): {len(all_ud_classes)}")
        print(f"  Classi unknown pure (disgiunte): {len(unknown_classes)}")

        # --- Selezione controllata delle classi unknown ---
        rng = np.random.default_rng(self.random_state)
        if self.n_unknown_classes is not None and self.n_unknown_classes < len(unknown_classes):
            unknown_classes = rng.choice(unknown_classes, size=self.n_unknown_classes, replace=False)
            print(f"  Classi unknown selezionate: {self.n_unknown_classes}")
        else:
            print(f"  Classi unknown selezionate: {len(unknown_classes)} (tutte)")

        # --- Filtra campioni ---
        unknown_mask = np.isin(self.y_ud, unknown_classes)
        self.X_ud = self.X_ud[unknown_mask]
        self.y_ud = self.y_ud[unknown_mask]
        self.X_flat_ud = self.X_flat_ud[unknown_mask]

        # Distribuzione campioni per classe
        from collections import Counter
        counts = Counter(self.y_ud)
        cnt_arr = np.array(list(counts.values()))

        print(f"\n  Campioni unknown totali: {len(self.y_ud)}")
        print(f"  Immagini per classe — min: {cnt_arr.min()}\n"
              f"max: {cnt_arr.max()}\n"
              f"media: {cnt_arr.mean():.1f}\n"
              f"std: {cnt_arr.std():.1f}\n"
              )
        print(f"{'=' * 55}\n")

        return self.X_ud, self.X_flat_ud, self.y_ud