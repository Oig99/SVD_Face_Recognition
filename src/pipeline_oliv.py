import time
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

from src.data_loader import DataLoader
from src.svd_engine import SVDReducerEngine
from src.recognizer import FaceRecognizer
from src.visualizer import Visualizer

import warnings

warnings.filterwarnings('ignore')

def main():
    """
    Script principale per Face Recognition con SVD utilizzando il dataset olivetti
    Include tutte le visualizzazioni e analisi presenti nel notebook.
    """
    viz = Visualizer(path=r"result/olivetti")

    # === CARICAMENTO DATASET ===
    dataset = DataLoader()
    dataset.X, dataset.X_flat, dataset.y = dataset.load_olivetti_data()

    print(f"Shape immagini: {dataset.X.shape}")
    print(f"Shape flatten: {dataset.X_flat.shape}")
    print(f"Numero totale immagini: {len(dataset.y)}")
    print(f"Numero persone uniche: {len(np.unique(dataset.y))}")

    # Visualizza campioni di volti
    viz.plot_sample_faces(dataset.X, dataset.y, n_samples=10)

    # === CENTRATURA DATI ===

    dataset.X_centered = dataset.center_data()
    print(f"Shape dati centrati: {dataset.X_centered.shape}")
    print(f"Media dei dati centrati: {np.mean(dataset.X_centered):.10f}")

    varianza_totale = np.var(dataset.X_centered, axis=0).sum()
    print(f"\nVarianza totale dei dati centrati: {varianza_totale:.6f}")

    # Visualizza volto medio
    viz.plot_mean_face(dataset.mean_face)

    # === SVD COMPLETA ===

    svd_reducer = SVDReducerEngine(energy_threshold=0.95)
    U, S, VT, energy = svd_reducer.compute_full_svd(dataset.X_centered)

    print(f"\nShape U: {U.shape}")
    print(f"Shape S: {S.shape}")
    print(f"Shape V^T: {VT.shape}")
    print(f"\nPrimi 5 valori singolari: {S[:5]}")

    # Visualizza eigenfaces
    viz.plot_eigenfaces(VT, n_components=10)

    # === SELEZIONE COMPONENTI ===
    n_components = svd_reducer.select_components()
    print(f"Energia richiesta: {svd_reducer.energy_threshold * 100}%")
    print(f"Componenti selezionate: {n_components}")
    print(f"Energia effettiva: {energy[n_components - 1] * 100:.2f}%")
    print(f"Riduzione dimensionale: {dataset.X_flat.shape[1]} â†’ {n_components}")
    print(f"Fattore di compressione: {dataset.X_flat.shape[1] / n_components:.2f}x")

    # Visualizza energia cumulativa
    viz.plot_cumulative_energy(energy)

    # === VARIANZA SPIEGATA DAI COMPONENTI PRINCIPALI ===
    varianza_componenti = (S ** 2) / (dataset.X_centered.shape[0] - 1)
    varianza_totale_svd = varianza_componenti.sum()
    varianza_spiegata = np.sum(varianza_componenti[:n_components]) / varianza_totale_svd * 100
    varianza_residua = varianza_totale_svd - np.sum(varianza_componenti[:n_components])

    print(f"\nVarianza totale (SVD): {varianza_totale_svd:.6f}")
    print(f"Varianza spiegata dai primi {n_components} componenti: {varianza_spiegata:.2f}%")
    print(f"Varianza residua dopo riduzione dimensionale: {varianza_residua:.6f}")

    # === RIDUZIONE DIMENSIONALE ===
    X_reduced = svd_reducer.fit_transform(dataset.X_centered)

    print(f"\nShape dati ridotti: {X_reduced.shape}")


    # Ricostruzione dei volti dallo spazio ridotto
    X_reconstructed = svd_reducer.reconstruct_face(X_reduced, dataset.mean_face)

    # Visualizza il primo volto originale e ricostruito
    viz.plot_original_vs_reconstructed(dataset.X_flat, X_reconstructed, dataset.y)
    # viz.plot_original_vs_reconstructed_lfw(dataset.X_flat, X_reconstructed, dataset.y, dataset.X)

    # Visualizza proiezione 2D
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
    # GRID SEARCH SVM (versione magistrale)
    # =====================================================
    print("\n" + "=" * 60)
    print("GRID SEARCH SVM")
    print("=" * 60)

    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

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
    # ANALISI SUPPORT VECTORS
    # =====================================================
    print("\nAnalisi Support Vectors:")
    print("Support vectors per classe:", best_svm.n_support_)
    print("Totale support vectors:", np.sum(best_svm.n_support_))

    # =====================================================
    # 6CONFRONTO FINALE
    # =====================================================
    print("\n" + "=" * 60)
    print("CONFRONTO FINALE MODELLI")
    print("=" * 60)

    print(f"KNN Accuracy: {knn_acc * 100:.2f}%")
    print(f"SVM Lineare Accuracy: {lin_acc * 100:.2f}%")
    print(f"SVM RBF Accuracy: {rbf_acc * 100:.2f}%")

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

    print("\n=== ANALISI ERRORE RICOSTRUZIONE ===")
    print(f"MSE medio: {mean_mse:.6f}")
    print(f"Deviazione standard MSE: {std_mse:.6f}")
    print(f"MSE minimo: {np.min(mse_per_sample):.6f}")
    print(f"MSE massimo: {np.max(mse_per_sample):.6f}")

    viz.plot_reconstruction_error(mse_per_sample)

    # === PREDIZIONI CON CONFIDENCE ===
    results_with_confidence = recognizer.predict_with_confidence(X_test)

    # Stampiamo le prime 5 predizioni con confidence
    print("\nPredizioni con confidence:")
    for i, res in enumerate(results_with_confidence[:5]):
        print(f"Sample {i}: Predizione={res['prediction']}, "
              f"Confidence={res['confidence']:.3f}, "
              f"Avg distance={res['avg_distance']:.3f}, "
              f"Neighbor consensus={res['neighbor_consensus']:.3f}")

    # === VALUTAZIONE ===

    y_pred = recognizer.evaluate_knn(X_test, y_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    viz.save_excel(report, "report.xlsx")

    # Visualizza matrice di confusione
    viz.plot_confusion_matrix(y_test, y_pred)

    result = recognizer.cross_validate(X_train, y_train, cv=5)
    viz.save_excel([result], 'result.xlsx')

    error = recognizer.analyze_misclassifications(X_train, y_train, y_pred)

    # Mostra i primi errori
    print("\nPrimi 5 errori di classificazione:")
    for e in error['misclassified_samples'][:5]:
        print(e)


    # Mostrate le coppie più confuse
    print("\nCoppie più confuse:")
    for pair, count in error['most_confused_pairs']:
        print(f"{pair}: {count} volte")


    best_params, best_score = recognizer.optimize_hyperparameters(X_train, y_train)
    print("\nKNN; best_params:", best_params, "best_score:", best_score)


    # === ANALISI DISTANZE ===

    distances_train = recognizer.compute_min_distances(X_train, X_train)
    distances_test = recognizer.compute_min_distances(X_test, X_train)

    print(f"Distanze Training Set:")
    print(f"  Min: {np.min(distances_train):.4f}")
    print(f"  Max: {np.max(distances_train):.4f}")
    print(f"  Media: {np.mean(distances_train):.4f}")
    print(f"  Std: {np.std(distances_train):.4f}")

    print(f"\nDistanze Test Set:")
    print(f"  Min: {np.min(distances_test):.4f}")
    print(f"  Max: {np.max(distances_test):.4f}")
    print(f"  Media: {np.mean(distances_test):.4f}")
    print(f"  Std: {np.std(distances_test):.4f}")

    # Visualizza distribuzione distanze
    viz.plot_distance_distribution(distances_test, recognizer.unknown_threshold)

    # === TEST VOLTO SCONOSCIUTO ===
    is_image = True
    # Prende un' immagine di esempio se settato a True altrimenti prende il rumore
    if is_image:
        # Carica immagine
        img = Image.open(r"image_example.jpg").convert('L')  # converti in grayscale

        # Ridimensiona alla stessa dimensione del dataset Olivetti (64x64). Ottieni dimensioni reali dataset
        h, w = dataset.X.shape[1], dataset.X.shape[2]

        img = img.resize((w, h))  # attenzione: PIL usa (width, height)

        # Converti in array numpy e flatten
        unknown_face = np.array(img).flatten().astype(float)

        # Normalizza se necessario (Olivetti ha valori 0-1)
        unknown_face /= 255.0

        # Aggiungi dimensione batch
        unknown_face = unknown_face.reshape(1, -1)
        face_centered = unknown_face - dataset.mean_face
        svd_reducer.transform(face_centered)
    else:
        np.random.seed(0)
        unknown_face = np.random.rand(1, dataset.X_flat.shape[1])

    print(f"Shape: {unknown_face.shape}")

    label, distance = recognizer.detect_unknown(
        unknown_face,
        dataset.mean_face,
        svd_reducer,
        X_train
    )

    print(f"\nDistanza minima trovata: {distance:.3f}")
    print(f"Soglia impostata: {recognizer.unknown_threshold}")

    if label == "UNKNOWN":
        print("Volto NON riconosciuto (sconosciuto)")
    else:
        print(f"Volto riconosciuto come ID: {label}")

    viz.plot_new_faces(
        unknown_face=unknown_face,  # array flatten 1x4096
        X=dataset.X,  # immagini originali shape (n_samples, h, w)
        distance=distance,  # distanza calcolata
        label=label,  # 'UNKNOWN' o ID predetto
        th=recognizer.unknown_threshold  # soglia utilizzata
    )

    # Aggiorna soglia unknown
    recognizer.optimize_unknown_threshold(X_train, X_test)

    # Test volto sconosciuto con nuova soglia
    label, distance = recognizer.detect_unknown(
        unknown_face,
        dataset.mean_face,
        svd_reducer,
        X_train
    )
    print(f"\nVolto sconosciuto test: {label}, distanza={distance:.3f}")


    # === RIEPILOGO FINALE ===

    accuracy = np.mean(y_pred == y_test)

    print(f"Accuracy complessiva: {accuracy * 100:.2f}%")
    print(f"Componenti mantenute: {n_components}")
    print(f"Energia preservata: {energy[n_components - 1] * 100:.2f}%")
    print(f"Riduzione dimensionale: {dataset.X_flat.shape[1]}; {n_components}")
    print(f"Test volto sconosciuto: {'RIFIUTATO' if label == 'UNKNOWN' else 'ACCETTATO'}")

    print("\n" + "=" * 80)
    print("  ANALISI COMPLETATA")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()