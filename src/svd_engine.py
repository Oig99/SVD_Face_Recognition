import numpy as np
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD


class SVDReducerEngine:
    """
    Classe responsabile di:
    - Calcolo SVD completa (analisi energia)
    - Selezione numero componenti ottimale
    - Riduzione dimensionale con TruncatedSVD
    """
    def __init__(self, energy_threshold=0.95, random_state=42):
        # Percentuale di energia da mantenere
        self.energy_threshold = energy_threshold
        self.random_state = random_state

        self.U = None
        self.S = None
        self.VT = None

        self.energy = None
        self.components = None
        self.svd_model = None

    def compute_full_svd(self, X_centred):
        """
        Calcola la SVD completa: X = U SIGMA V^T

        I valori singolari (S) permettono di stimare quanta informazione (energia) contiene ogni componente.
        """
        self.U, self.S, self.VT = svd(X_centred, full_matrices=False)
        self.energy = np.cumsum(self.S**2) / np.sum(self.S**2)
        return self.U, self.S, self.VT, self.energy

    def select_components(self):
        """
        Seleziona il numero minimo di componenti necessarie per raggiungere la soglia di energia.
        """
        self.components = np.argmax(self.energy >= self.energy_threshold) + 1
        print(f"Componenti minimo: {self.components}")
        return self.components

    def fit_transform(self, X_centered):
        """ Applica Truncated SVD per ridurre dimensionalità. Restituisce la proiezione dei dati nello spazio ridotto."""
        self.svd_model = TruncatedSVD(
            n_components=self.components,
            random_state=self.random_state
        )
        return self.svd_model.fit_transform(X_centered)

    def transform(self, X):
        """
        Proietta nuovi dati (es. volto sconosciuto) nello spazio ridotto già appreso.
        """
        return self.svd_model.transform(X)

    def reconstruct_face(self, X_reduced, mean_face):
        """
        Ricostruisce i volti originali a partire dalle componenti ridotte.
        :param X_reduced: Dati ridotti nello spazio TruncatedSVD (n_samples, n_components)
        :param mean_face: Volto medio del dataset (shape: n_features)
        :return: Volti ricostruiti nello spazio originale (n_samples, n_features)
        """
        if self.svd_model is None:
            raise ValueError("Devi prima fit_transform() per calcolare lo spazio ridotto.")

        # Ricostruzione dei dati nello spazio originale
        X_reconstructed = X_reduced @ self.svd_model.components_
        # Aggiungi il volto medio per tornare ai valori originali
        X_reconstructed += mean_face

        return X_reconstructed

