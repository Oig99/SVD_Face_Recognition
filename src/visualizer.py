import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

    def plot_sample_faces(self, X, y, n_samples=10):
        """
        Visualizza campioni di volti dal dataset.
        """
        fig, axes = plt.subplots(1, n_samples, figsize=(15, 2))
        for i, ax in enumerate(axes):
            ax.imshow(X[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f"ID: {y[i]}")
        plt.suptitle("Esempi di volti")
        plt.tight_layout()
        filepath = os.path.join(self.path, 'sample_faces_lfw.png')
        plt.savefig(filepath)
        plt.show()
        
        
    def plot_mean_face(self, mean_face, shape=(64, 64)):
        """
        Visualizza il volto medio del dataset.
        """
        plt.figure(figsize=(4, 4))
        plt.imshow(mean_face.reshape(shape), cmap='gray')
        plt.title("Volto medio (Mean Face)")
        plt.axis('off')
        filepath = os.path.join(self.path, 'mean_faces.png')
        plt.savefig(filepath)
        plt.show()

    def plot_mean_face_lfw(self, mean_face, X):
        """
        Visualizza il volto medio del dataset. (è una versione con le dimensioni reali)
        """
        plt.figure(figsize=(4, 4))
        h, w = X.shape[1], X.shape[2]  # dimensioni reali dataset
        plt.imshow(mean_face.reshape(h, w), cmap='gray')
        plt.title("Volto medio (Mean Face)")
        plt.axis('off')
        filepath = os.path.join(self.path, 'mean_faces.png')
        plt.savefig(filepath)
        plt.show()

    
    def plot_eigenfaces(self, VT, n_components=10, shape=(64, 64)):
        """
        Visualizza le prime eigenfaces (componenti principali).
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
        plt.show()

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
        plt.show()

    
    def plot_cumulative_energy(self, energy):
        """
        Visualizza l'energia cumulativa rispetto al numero di componenti.
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
        plt.show()

    
    def plot_2d_projection(self, X_reduced, y, title="Proiezione 2D delle prime componenti"):
        """
        Visualizza la proiezione 2D dei dati ridotti (prime 2 componenti).
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
        plt.show()

    
    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Visualizza la matrice di confusione.
        """
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax = plt.subplots(figsize=(12, 10))
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        plt.title('Matrice di Confusione')
        plt.tight_layout()
        filepath = os.path.join(self.path, 'confusion_matrix.png')
        plt.savefig(filepath)
        plt.show()

    
    def plot_distance_distribution(self, distances_test, threshold):
        """
        Visualizza la distribuzione delle distanze minime per train e test set.
        Utile per analizzare la soglia di riconoscimento volti sconosciuti.
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
        filepath = os.path.join(self.path, 'distance_distribution.png')
        plt.savefig(filepath)
        plt.show()

    def plot_original_vs_reconstructed(self, X_original, X_reconstructed, y_true, num_samples=5, shape=(64, 64)):
        """
        Visualizza affiancati il volto originale e quello ricostruito.
        """
        num_samples = min(num_samples, len(X_original))  # sicurezza

        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5))

        for i in range(num_samples):
            # --- Volti originali ---
            axes[0, i].imshow(X_original[i].reshape(shape), cmap='gray')
            title_orig = f"ID: {y_true[i]}" if y_true is not None else "Originale"
            axes[0, i].set_title(title_orig, fontsize=10)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel("ORIGINALI", fontsize=12, fontweight='bold')

            # --- Volti ricostruiti ---
            axes[1, i].imshow(X_reconstructed[i].reshape(shape), cmap='gray')
            axes[1, i].set_title("Ricostruita", fontsize=10)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel("RICOSTRUITI", fontsize=12, fontweight='bold')


        # Titolo generale
        plt.suptitle("Confronto Originali vs Ricostruiti", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.path, 'all_reconstructions.png'), bbox_inches='tight')
        plt.show()

    def plot_original_vs_reconstructed_lfw(self, X_original, X_reconstructed, y_true, X, num_samples=5):
        """
        Mostra affiancati il volto originale e quello ricostruito.
        Funziona con qualsiasi dataset (Olivetti, LFW, ecc.).
        """
        h, w = X.shape[1], X.shape[2]
        num_samples = min(num_samples, len(X_original))

        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5))

        for i in range(num_samples):
            # --- ORIGINALI ---
            axes[0, i].imshow(X_original[i].reshape(h, w), cmap='gray')
            title_orig = f"ID: {y_true[i]}" if y_true is not None else "Originale"
            axes[0, i].set_title(title_orig, fontsize=10)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel("ORIGINALI", fontsize=12, fontweight='bold')

            # --- RICOSTRUITI ---
            axes[1, i].imshow(X_reconstructed[i].reshape(h, w), cmap='gray')
            axes[1, i].set_title("Ricostruita", fontsize=10)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel("RICOSTRUITI", fontsize=12, fontweight='bold')

        plt.suptitle("Confronto Originali vs Ricostruiti", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.path, 'all_reconstructions.png'), bbox_inches='tight')
        plt.show()


    def save_excel(self, df, filename):
        """Salva excel"""
        df = pd.DataFrame(df).transpose()
        df.to_excel(os.path.join(self.path, filename))

    def plot_new_faces(self, unknown_face, X, distance, label, th):
        """
        Visualizza la ricostruzione di un volto esterno (o sintetico).
        Unknown detection
        """
        # Plot
        plt.figure(figsize=(4, 4))

        # Volto simulato
        plt.imshow(unknown_face.reshape(X.shape[1], X.shape[2]), cmap='gray')
        plt.axis("off")
        th = th/100

        plt.suptitle(
            f"Min dist: {distance:.3f} | Threshold: {th}\n"
            + ("NON riconosciuto (UNKNOWN)" if label == "UNKNOWN" else f"Riconosciuto come ID: {label}")
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.path, 'simulation_face.png'), bbox_inches='tight')
        plt.show()

    def plot_reconstruction_error(self, mse_per_sample):
        """
        Visualizza l'errore di ricostruzione per ogni campione.
        Gestisce casi in cui tutti i valori siano uguali.
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
        plt.show()
        plt.close()

