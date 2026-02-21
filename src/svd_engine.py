import numpy as np
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD


class SVDReducerEngine:
    """
    Classe responsabile di:
    - Calcolo della SVD completa (analisi spettro dei valori singolari)
    - Stima dell'energia cumulativa
    - Selezione automatica del numero ottimale di componenti
    - Riduzione dimensionale con TruncatedSVD
    - Proiezione e ricostruzione nello spazio originale
    """
    def __init__(self, energy_threshold=0.95, random_state=42):
        """
        :param energy_threshold: frazione di varianza totale da preservare (es. 0.95 = 95%)
        :param random_state: per riproducibilità della TruncatedSVD
        """
        self.energy_threshold = energy_threshold
        self.random_state = random_state

        # Matrici della SVD completa
        self.U = None
        self.S = None
        self.VT = None

        # Energia cumulativa associata ai valori singolari
        self.energy = None

        # Numero di componenti selezionate
        self.components = None

        # Modello TruncatedSVD per riduzione pratica
        self.svd_model = None

    def compute_full_svd(self, X_centred):
        """
        Calcola la SVD completa: X = U SIGMA V^T

        I valori singolari (S) rappresentano l'importanza di ciascuna direzione principale nello spazio delle feature.

        L'energia cumulativa permette di quantificare quanta varianza del dataset è spiegata dalle prime k componenti.
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
        """
        Applica TruncatedSVD per ridurre la dimensionalità dei dati.

        A differenza della SVD completa (usata per analisi energetica),
        TruncatedSVD calcola direttamente le prime 'k' componenti rendendo il processo computazionalmente più efficiente.

        Returns:
        - Proiezione dei dati nello spazio ridotto (n_samples, n_components)
        """
        if self.components is None:
            raise ValueError("Devi prima chiamare select_components().")
        self.svd_model = TruncatedSVD(
            n_components=self.components,
            random_state=self.random_state
        )
        return self.svd_model.fit_transform(X_centered)

    def transform(self, X):
        """
        Proietta nuovi campioni (es. volto di test o sconosciuto)
        nello spazio latente già appreso.

        Utilizza le componenti calcolate durante fit_transform().
        """
        if self.svd_model is None:
            raise ValueError("Il modello SVD non è stato ancora addestrato.")
        return self.svd_model.transform(X)

    def reconstruct_face(self, X_reduced, mean_face):
        """
        Ricostruisce i volti nello spazio originale delle feature.

        Procedura:
        1. Proiezione inversa tramite moltiplicazione per le componenti principali
        2. Ri-aggiunta del volto medio

        X_original ≈ Z @ V_k + mean_face

        Parameters:
        - X_reduced: rappresentazione nello spazio ridotto (n_samples, k)
        - mean_face: volto medio usato per la centratura

        Returns:
        - Volti ricostruiti nello spazio originale (n_samples, n_features)
        """
        if self.svd_model is None:
            raise ValueError("Devi prima fit_transform() per calcolare lo spazio ridotto.")

        # Ricostruzione dei dati nello spazio originale
        X_reconstructed = X_reduced @ self.svd_model.components_
        # Aggiungi il volto medio per tornare ai valori originali
        X_reconstructed += mean_face

        return X_reconstructed

