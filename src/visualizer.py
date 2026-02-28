import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


class Visualizer:
    """
    Classe responsabile di tutte le visualizzazioni del progetto:
    - Campioni di volti
    - Volto medio
    - Eigenfaces
    - Energia cumulativa
    - Scatter plot 2D
    - Matrice di confusione
    - Distribuzione distanze
    """
    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    # ---------------------------- Visualizzazione dati grezzi ----------------------------

    def plot_sample_faces(self, X, y, n_samples=10):
        """
        Visualizza campioni di volti dal dataset.
        Utile per:
        - Osservare variabilità intra-classe
        - confrontare condizioni controllate
        """
        fig, axes = plt.subplots(1, n_samples, figsize=(15, 2))
        for i, ax in enumerate(axes):
            ax.imshow(X[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f"ID: {y[i]}")
        plt.suptitle("Esempi di volti")
        plt.tight_layout()
        filepath = os.path.join(self.path, 'sample_faces.png')
        plt.savefig(filepath)

    def plot_mean_face(self, mean_face, shape=(64, 64)):
        """
        Visualizza il volto medio del dataset.

        Il mean face rappresenta il centro geometrico nello spazio delle feature.
        Tutte le analisi SVD sono effettuate rispetto a questo punto.
        """
        plt.figure(figsize=(4, 4))
        plt.imshow(mean_face.reshape(shape), cmap='gray')
        plt.title("Volto medio (Mean Face)")
        plt.axis('off')
        filepath = os.path.join(self.path, 'mean_faces.png')
        plt.savefig(filepath)

    def plot_mean_face_lfw(self, mean_face, X):
        """
        Versione generica del mean face che usa le dimensioni reali del dataset.
        """
        plt.figure(figsize=(4, 4))
        h, w = X.shape[1], X.shape[2]  # dimensioni reali dataset
        plt.imshow(mean_face.reshape(h, w), cmap='gray')
        plt.title("Volto medio (Mean Face)")
        plt.axis('off')
        filepath = os.path.join(self.path, 'mean_faces.png')
        plt.savefig(filepath)

    # ---------------------------- Eigenfaces e analisi spettrale ----------------------------

    def plot_eigenfaces(self, VT, n_components=10, shape=(64, 64)):
        """
        Visualizza le prime eigenfaces (componenti principali).

        Ogni eigenface rappresenta una direzione di massima varianza.
        L'ordine riflette l'importanza (valore singolare associato).
        """
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i in range(min(n_components, len(axes))):
            eigenface = VT[i].reshape(shape)
            axes[i].imshow(eigenface, cmap='gray')
            axes[i].set_title(f"Eigenface {i + 1}")
            axes[i].axis('off')

        plt.suptitle("Prime 10 Eigenfaces")
        filepath = os.path.join(self.path, 'eigenfaces.png')
        plt.savefig(filepath)

    def plot_eigenfaces_lfw(self, VT, X, n_components=10):
        """
        Visualizza le prime eigenfaces (componenti principali), adattamento per il dataset LFW.
        """
        h, w = X.shape[1], X.shape[2]

        plt.figure(figsize=(12, 6))

        for i in range(n_components):
            plt.subplot(2, n_components // 2, i + 1)

            eigenface = VT[i].reshape(h, w)
            plt.imshow(eigenface, cmap='gray')
            plt.title(f"Eigenface {i + 1}")
            plt.axis("off")

        plt.suptitle("Prime 10 Eigenfaces")
        filepath = os.path.join(self.path, 'eigenfaces.png')
        plt.savefig(filepath)

    def plot_cumulative_energy(self, energy):
        """
        Mostra l'energia cumulativa spiegata dalle componenti SVD.

        Permette di:
        - valutare il decadimento dello spettro
        - stimare il rango effettivo del dataset
        - selezionare k tale che E_k ≥ soglia (es. 95%)
        """
        plt.figure(figsize=(10, 5))
        plt.plot(energy, linewidth=2)
        plt.axhline(y=0.95, color='r', linestyle='--', label='Soglia 95%')
        plt.xlabel('Numero di componenti')
        plt.ylabel('Energia cumulativa')
        plt.title('Energia cumulativa spiegata dalle componenti SVD')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(self.path, 'cumulative_energy.png')
        plt.savefig(filepath)

    # ---------------------------- Geometria nello spazio ridotto ----------------------------

    def plot_2d_projection(self, X_reduced, y, title="Proiezione 2D delle prime componenti"):
        """
        Proiezione dei campioni sulle prime due componenti principali.

        Utile per osservare:
        - separabilità tra classi
        - eventuale sovrapposizione
        - struttura clusterizzata del dataset
        """
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                              c=y, cmap='tab20', alpha=0.7, edgecolors='k')
        plt.colorbar(scatter, label='ID Persona')
        plt.xlabel('Prima componente')
        plt.ylabel('Seconda componente')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(self.path, 'projection.png')
        plt.savefig(filepath)

    # ---------------------------- Analisi classificazione ----------------------------

    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Matrice di confusione.

        Permette di identificare:
        - classi più frequentemente confuse
        - pattern sistematici di errore
        """
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax = plt.subplots(figsize=(12, 10))
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        plt.title('Matrice di Confusione')
        plt.tight_layout()
        filepath = os.path.join(self.path, 'confusion_matrix.png')
        plt.savefig(filepath)

    def plot_distance_distribution(self, distances_test, threshold, filename='distance_distribution.png'):
        """
        Visualizza la distribuzione delle distanze minime per train e test set.
        Utile per analizzare la soglia di riconoscimento volti sconosciuti.
        Fondamentale per la open-set recognition:
        la soglia separa la regione delle classi note da punti potenzialmente esterni alla distribuzione.
        """
        plt.figure(figsize=(12, 5))

        plt.hist(distances_test, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Soglia = {threshold}')
        plt.xlabel('Distanza minima')
        plt.ylabel('Frequenza')
        plt.title('Distribuzione distanze minime')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.path, filename)
        plt.savefig(filepath)

    # ---------------------------- Ricostruzione ----------------------------

    def plot_original_vs_reconstructed(self, X_original, X_reconstructed, y_true, num_samples=5, shape=(64, 64)):
        """
        Visualizza affiancati il volto originale e quello ricostruito.

        Permette di valutare qualitativamente:
        - perdita di informazione
        - capacità compressiva della SVD
        """
        num_samples = min(num_samples, len(X_original))
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5))

        for i in range(num_samples):
            # Volti originali
            axes[0, i].imshow(X_original[i].reshape(shape), cmap='gray')
            title_orig = f"ID: {y_true[i]}" if y_true is not None else "Originale"
            axes[0, i].set_title(title_orig, fontsize=10)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel("ORIGINALI", fontsize=12, fontweight='bold')

            # Volti ricostruiti
            axes[1, i].imshow(X_reconstructed[i].reshape(shape), cmap='gray')
            axes[1, i].set_title("Ricostruita", fontsize=10)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel("RICOSTRUITI", fontsize=12, fontweight='bold')

        plt.suptitle("Confronto Originali vs Ricostruiti", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.path, 'all_reconstructions.png'), bbox_inches='tight')

    def plot_original_vs_reconstructed_lfw(self, X_original, X_reconstructed, y_true, X, num_samples=5):
        """
        Mostra affiancati il volto originale e quello ricostruito (Adattamento dataset realistici)
        """
        h, w = X.shape[1], X.shape[2]
        num_samples = min(num_samples, len(X_original))

        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5))

        for i in range(num_samples):
            # --- Volti originali ---
            axes[0, i].imshow(X_original[i].reshape(h, w), cmap='gray')
            title_orig = f"ID: {y_true[i]}" if y_true is not None else "Originale"
            axes[0, i].set_title(title_orig, fontsize=10)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel("ORIGINALI", fontsize=12, fontweight='bold')

            # --- Volti ricostruiti ---
            axes[1, i].imshow(X_reconstructed[i].reshape(h, w), cmap='gray')
            axes[1, i].set_title("Ricostruita", fontsize=10)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel("RICOSTRUITI", fontsize=12, fontweight='bold')

        plt.suptitle("Confronto Originali vs Ricostruiti", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.path, 'all_reconstructions.png'), bbox_inches='tight')
        

    # ----------------------------  Unknown detection ----------------------------

    def plot_new_faces(self, unknown_face, X, distance, label, th, filepath='simulation_face.png'):
        """
        Visualizza un volto test e il risultato della open-set decision.

        Nota:
        La soglia deve essere mostrata nel suo valore reale, altrimenti si altera il significato geometrico.
        """
        plt.figure(figsize=(4, 4))

        # Volto simulato
        plt.imshow(unknown_face.reshape(X.shape[1], X.shape[2]), cmap='gray')
        plt.axis("off")

        plt.suptitle(
            f"Min dist: {distance:.3f} | Threshold: {th}\n"
            + ("NON riconosciuto (UNKNOWN)" if label == "UNKNOWN" else f"Riconosciuto come ID: {label}")
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.path, filepath), bbox_inches='tight')
        plt.close()

    def plot_unknown_detection_results(self, X_ud, y_ud, results_ud, X_ref, n_samples=10):
        """
        Visualizza un campione di volti unknown con l'esito della detection.
        Verde = correttamente rifiutato (UNKNOWN)
        Rosso = erroneamente accettato (falso positivo)
        """
        h, w = X_ref.shape[1], X_ref.shape[2]
        n_samples = min(n_samples, len(results_ud))

        fig, axes = plt.subplots(2, n_samples // 2, figsize=(n_samples * 2, 6))
        axes = axes.ravel()

        for i, ax in enumerate(axes):
            r = results_ud[i]
            img = X_ud[r['index']].reshape(h, w)
            true_cls = y_ud[r['index']]

            ax.imshow(img, cmap='gray')
            ax.axis('off')

            color = 'green' if r['correctly_rejected'] else 'red'
            esito = "UNKNOWN" if r['correctly_rejected'] else f"ID:{r['label']}"

            ax.set_title(
                f"True:{true_cls} | {esito}\nd={r['distance']:.2f}",
                color=color,
                fontsize=8
            )

        plt.suptitle("Unknown Detection — Verde=Rifiutato  Rosso=Accettato", fontsize=12)
        plt.tight_layout()
        filepath = os.path.join(self.path, 'unknown_detection_faces.png')
        plt.savefig(filepath)
        plt.show()
        plt.close()

    # ----------------------------  Errore di ricostruzione ----------------------------

    def plot_reconstruction_error(self, mse_per_sample):
        """
        Visualizza l'errore di ricostruzione per ogni campione. Gestisce casi in cui tutti i valori siano uguali.
        Utile per:
        - valutare qualità media della compressione
        - individuare campioni anomali
        """
        std_mse = np.std(mse_per_sample)
        print(f"Deviazione standard MSE: {std_mse:.6e}")

        plt.figure(figsize=(12, 6))

        # Se tutti i valori sono uguali, crea almeno 1 bin
        unique_vals = np.unique(mse_per_sample)
        if len(unique_vals) <= 1:
            bins = 1
        else:
            bins = min(30, len(unique_vals))
        plt.hist(mse_per_sample, bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel("Indice campione")
        plt.ylabel("MSE")
        plt.title("Errore di ricostruzione per campione")
        plt.grid(True, alpha=0.3)
        filepath = os.path.join(self.path, 'reconstruction_error.png')
        plt.savefig(filepath)
        plt.close()

    # ----------------------------  utils ----------------------------

    def save_excel(self, df, filename):
        """Salva excel"""
        df = pd.DataFrame(df).transpose()
        df.to_excel(os.path.join(self.path, filename), index=False)

    @staticmethod
    def classifier(y_test, y_pred, output_dict):
        """Genera il classification report."""
        print(classification_report(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=output_dict)
        return report