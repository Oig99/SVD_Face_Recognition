import time
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.data_loader import DataLoader
from src.svd_engine import SVDReducerEngine
from src.recognizer import FaceRecognizer
from src.visualizer import Visualizer

import warnings

warnings.filterwarnings('ignore')


def main():
    viz = Visualizer(path=r"result/lfw")

    # === CARICAMENTO DATASET ===
    dataset = DataLoader()
    dataset.X, dataset.X_flat, dataset.y = dataset.load_lfw_data()

    print(f"Shape immagini: {dataset.X.shape}")
    print(f"Shape flatten: {dataset.X_flat.shape}")
    print(f"Numero totale immagini: {len(dataset.y)}")
    print(f"Numero persone uniche: {len(np.unique(dataset.y))}")

    viz.plot_sample_faces(dataset.X, dataset.y, n_samples=10)

    # === CENTRATURA DATI ===
    dataset.X_centered = dataset.center_data()
    print(f"Shape dati centrati: {dataset.X_centered.shape}")
    print(f"Media dei dati centrati: {np.mean(dataset.X_centered):.10f}")


    varianza_totale = np.var(dataset.X_centered, axis=0).sum()
    print(f"\nVarianza totale dei dati centrati: {varianza_totale:.6f}")

    viz.plot_mean_face_lfw(dataset.mean_face, dataset.X)

    # === SVD COMPLETA ===
    svd_reducer = SVDReducerEngine(energy_threshold=0.95)
    U, S, VT, energy = svd_reducer.compute_full_svd(dataset.X_centered)

    print(f"Shape U: {U.shape}")
    print(f"Shape S: {S.shape}")
    print(f"Shape V^T: {VT.shape}")
    print(f"\nPrimi 5 valori singolari: {S[:5]}")

    viz.plot_eigenfaces_lfw(VT, dataset.X, n_components=10)

    n_components = svd_reducer.select_components()
    print(f"Energia richiesta: {svd_reducer.energy_threshold * 100}%")
    print(f"Componenti selezionate: {n_components}")
    print(f"Energia effettiva: {energy[n_components - 1] * 100:.2f}%")
    print(f"Riduzione dimensionale: {dataset.X_flat.shape[1]} → {n_components}")
    print(f"Fattore di compressione: {dataset.X_flat.shape[1] / n_components:.2f}x")


    # === VARIANZA SPIEGATA DAI COMPONENTI PRINCIPALI ===
    varianza_componenti = (S ** 2) / (dataset.X_centered.shape[0] - 1)
    varianza_totale_svd = varianza_componenti.sum()
    varianza_spiegata = np.sum(varianza_componenti[:n_components]) / varianza_totale_svd * 100
    varianza_residua = varianza_totale_svd - np.sum(varianza_componenti[:n_components])

    print(f"\nVarianza totale (SVD): {varianza_totale_svd:.6f}")
    print(f"Varianza spiegata dai primi {n_components} componenti: {varianza_spiegata:.2f}%")
    print(f"Varianza residua dopo riduzione dimensionale: {varianza_residua:.6f}")


    viz.plot_cumulative_energy(energy)

    # === RIDUZIONE DIMENSIONALE ===
    X_reduced = svd_reducer.fit_transform(dataset.X_centered)
    print(f"Shape dati ridotti: {X_reduced.shape}")

    # Ricostruzione volti
    X_reconstructed = svd_reducer.reconstruct_face(X_reduced, dataset.mean_face)
    viz.plot_original_vs_reconstructed_lfw(dataset.X_flat, X_reconstructed, dataset.y, dataset.X)

    viz.plot_2d_projection(X_reduced, dataset.y)

    # === SPLIT TRAIN/TEST ===
    X_train, X_test, y_train, y_test = dataset.dataset_splitting(X_reduced)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Split ratio: {dataset.test_size * 100:.0f}% test")

    print("\nOttimizzazione iperparametri KNN in corso...")

    # === ANALISI ERRORE DI RICOSTRUZIONE ===
    mse_per_sample = np.mean((dataset.X_flat - X_reconstructed) ** 2, axis=1)
    mean_mse = np.mean(mse_per_sample)
    std_mse = np.std(mse_per_sample)

    print("\n" + "=" * 60)
    print("CONFRONTO KNN vs SVM")
    print("=" * 60)

    # =====================================================
    # KNN
    # =====================================================
    start = time.time()
    recognizer = FaceRecognizer(n_neighbors=1, unknown_threshold=0.5)
    recognizer.train_knn(X_train, y_train)
    y_pred_knn = recognizer.evaluate_knn(X_test, y_test)
    knn_time = time.time() - start
    knn_acc = accuracy_score(y_test, y_pred_knn)

    print(f"\nKNN Accuracy: {knn_acc * 100:.2f}%")
    print(f"KNN Tempo training+predict: {knn_time:.4f}s")

    # =====================================================
    # SVM LINEARE
    # =====================================================
    start = time.time()
    svm_linear = SVC(kernel='linear', C=1)
    svm_linear.fit(X_train, y_train)
    y_pred_svm_lin = svm_linear.predict(X_test)
    lin_time = time.time() - start
    lin_acc = accuracy_score(y_test, y_pred_svm_lin)

    print(f"\nSVM Lineare Accuracy: {lin_acc * 100:.2f}%")
    print(f"SVM Lineare Tempo: {lin_time:.4f}s")

    # =====================================================
    # SVM RBF
    # =====================================================
    start = time.time()
    svm_rbf = SVC(kernel='rbf', C=10, gamma='scale')
    svm_rbf.fit(X_train, y_train)
    y_pred_svm_rbf = svm_rbf.predict(X_test)
    rbf_time = time.time() - start
    rbf_acc = accuracy_score(y_test, y_pred_svm_rbf)

    print(f"\nSVM RBF Accuracy: {rbf_acc * 100:.2f}%")
    print(f"SVM RBF Tempo: {rbf_time:.4f}s")

    print("\nClassification Report (SVM RBF):")
    print(classification_report(y_test, y_pred_svm_rbf))
    report = classification_report(y_test, y_pred_svm_rbf, output_dict=True)
    viz.save_excel(report, "svm_report.xlsx")

    # =====================================================
    # GRID SEARCH SVM
    # =====================================================
    print("\n" + "=" * 60)
    print("GRID SEARCH SVM")
    print("=" * 60)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

    start = time.time()
    grid.fit(X_train, y_train)
    grid_time = time.time() - start

    print("Best parameters:", grid.best_params_)
    print("Best CV accuracy:", grid.best_score_)
    print(f"GridSearch Time: {grid_time:.4f}s")

    best_svm = grid.best_estimator_
    y_pred_best = best_svm.predict(X_test)
    best_acc = accuracy_score(y_test, y_pred_best)

    print(f"Test accuracy con miglior SVM: {best_acc * 100:.2f}%")

    # =====================================================
    # Analisi support vectors
    # =====================================================
    print("\nAnalisi Support Vectors:")
    print("Support vectors per classe:", best_svm.n_support_)
    print("Totale support vectors:", np.sum(best_svm.n_support_))

    # =====================================================
    # CONFRONTO FINALE
    # =====================================================
    print("\n" + "=" * 60)
    print("CONFRONTO FINALE MODELLI")
    print("=" * 60)

    print(f"KNN Accuracy: {knn_acc * 100:.2f}%")
    print(f"SVM Lineare Accuracy: {lin_acc * 100:.2f}%")
    print(f"SVM RBF Accuracy: {rbf_acc * 100:.2f}%")
    print(f"Best SVM (GridSearch) Accuracy: {best_acc * 100:.2f}%")

    if best_acc > knn_acc:
        print("\n SVM performa meglio di KNN")
    else:
        print("\n KNN performa meglio di SVM")

    # Salvataggio risultati
    comparison = {
        "Model": ["KNN", "SVM Linear", "SVM RBF", "SVM Best"],
        "Accuracy": [knn_acc, lin_acc, rbf_acc, best_acc],
        "Time": [knn_time, lin_time, rbf_time, grid_time]
    }

    viz.save_excel(comparison, "model_comparison.xlsx")

    # === ERRORE RICOSTRUZIONE ===
    print("\n=== ANALISI ERRORE RICOSTRUZIONE ===")
    print(f"MSE medio: {mean_mse:.6f}")
    print(f"Deviazione standard MSE: {std_mse:.6f}")
    print(f"MSE minimo: {np.min(mse_per_sample):.6f}")
    print(f"MSE massimo: {np.max(mse_per_sample):.6f}")
    viz.plot_reconstruction_error(mse_per_sample)

    # === MATRICE DI CONFUSIONE ===
    viz.plot_confusion_matrix(y_test, y_pred_best)

    # === DISTANZE ===
    distances_train = recognizer.compute_min_distances(X_train, X_train)
    distances_test = recognizer.compute_min_distances(X_test, X_train)
    print(f"Distanze Training Set:\n  Min: {np.min(distances_train):.4f}, Max: {np.max(distances_train):.4f}, "
          f"Media: {np.mean(distances_train):.4f}, Std: {np.std(distances_train):.4f}")
    print(f"\nDistanze Test Set:\n  Min: {np.min(distances_test):.4f}, Max: {np.max(distances_test):.4f}, "
          f"Media: {np.mean(distances_test):.4f}, Std: {np.std(distances_test):.4f}")

    viz.plot_distance_distribution(distances_test, recognizer.unknown_threshold)

    print("\n" + "=" * 80)
    print("  ANALISI COMPLETATA")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()