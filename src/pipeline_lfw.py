import numpy as np
from PIL import Image
from src.data_loader import DataLoader
from src.svd_engine import SVDReducerEngine
from src.recognizer import FaceRecognizer
from src.visualizer import Visualizer

import warnings

warnings.filterwarnings('ignore')


def main():
    """
    Script principale per Face Recognition con SVD utilizzando il dataset Labeled Faces in the Wild
    Include tutte le visualizzazioni e analisi presenti nel notebook.
    """
    viz = Visualizer(path=r"result/lfw")
    recognizer = FaceRecognizer(n_neighbors=1, unknown_threshold=0.5)
    dataset = DataLoader()
    svd_reducer = SVDReducerEngine(energy_threshold=0.95)

    # === CARICAMENTO DATASET ===
    dataset.X, dataset.X_flat, dataset.y = dataset.load_lfw_data()

    # Dataset originale (prima SVD)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = dataset.dataset_splitting(
        dataset.X_flat
    )

    results_raw, knn_rep_raw, lin_rep_raw, rbf_rep_raw = recognizer.compare_classifiers(
        X_train_raw,
        y_train_raw,
        X_test_raw,
        y_test_raw
    )
    viz.save_excel(results_raw, "comparison_no_svd.xlsx")
    viz.save_excel(knn_rep_raw, "knn_no_svd.xlsx")
    viz.save_excel(lin_rep_raw, "lin_no_svd.xlsx")
    viz.save_excel(rbf_rep_raw, "rbf_no_svd.xlsx")

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

    print("\nOttimizzazione iperparametri in corso...")

    best_params, best_score = recognizer.optimize_hyperparameters(X_train, y_train)
    print("\nKNN; best_params:", best_params, "best_score:", best_score)

    best_params, best_score, support_total, support_per_classes = recognizer.optimize_svm(X_train, y_train)
    print("\nSVM; best_params:", best_params, "best_score:", best_score)
    print("Support vectors per classe:", support_total)
    print("Totale support vectors:", support_per_classes)

    # === ANALISI ERRORE DI RICOSTRUZIONE ===
    mse_per_sample = np.mean((dataset.X_flat - X_reconstructed) ** 2, axis=1)
    mean_mse = np.mean(mse_per_sample)
    std_mse = np.std(mse_per_sample)


    # =====================================================
    # CONFRONTO FINALE
    # =====================================================
    print("\n" + "=" * 60)
    print("CONFRONTO KNN vs SVM")
    print("=" * 60)

    results, knn_report, lin_report, rbf_report = recognizer.compare_classifiers(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Salvataggio report dettagliati
    viz.save_excel(knn_report, "knn_report.xlsx")
    viz.save_excel(lin_report, "svm_linear_report.xlsx")
    viz.save_excel(rbf_report, "svm_rbf_report.xlsx")

    # Salvataggio confronto sintetico
    viz.save_excel(results, "model_comparison.xlsx")

    # === ERRORE RICOSTRUZIONE ===
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
        print(
            f"[Sample {i:02d}] "
            f"Pred: {int(res['prediction'])} | "
            f"Conf: {res['confidence']:.3f} | "
            f"Dist: {res['avg_distance']:.3f} | "
            f"Consensus: {res['neighbor_consensus']:.2f}"
        )

    # === VALUTAZIONE ===
    y_pred = recognizer.evaluate_svm(X_test)
    report = viz.classifier(y_test, y_pred, output_dict=True)
    viz.save_excel(report, "report.xlsx")

    # === MATRICE DI CONFUSIONE ===
    viz.plot_confusion_matrix(y_test, y_pred)

    result = recognizer.cross_validate_svm(X_train, y_train, cv=5)
    viz.save_excel([result], 'result.xlsx')

    error = recognizer.analyze_misclassifications(X_train, y_train, y_pred)

    # Mostra i primi errori
    print("\nPrimi 5 errori di classificazione:")
    for e in error['misclassified_samples'][:5]:
        print(
            f"[Idx {int(e['index']):03d}] "
            f"True: {int(e['true_label'])} → "
            f"Pred: {int(e['predicted_label'])} | "
            f"Dist: {float(e['nearest_distance']):.4f}"
        )


    # Mostrate le coppie più confuse
    print("\nCoppie più confuse:")
    for pair, count in error['most_confused_pairs']:
        print(f"{pair[0]}: {pair[1]} : {int(count)} volte")

    # === DISTANZE ===
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
    try:
        # Carica immagine reale
        img = Image.open(r"image_example.jpg").convert('L')
        h, w = dataset.X.shape[1], dataset.X.shape[2]
        img = img.resize((w, h))
        unknown_face = np.array(img).flatten().astype(float) / 255.0
        unknown_face = unknown_face.reshape(1, -1)
        print("Immagine caricata correttamente.")

    except FileNotFoundError:
        print("Immagine non trovata, genero volto casuale come placeholder...")
        unknown_face, _, _, _ = recognizer.simulate_unknown_detection(
            dataset.X_flat, X_train, dataset.mean_face, svd_reducer
        )

    print(f"Shape volto: {unknown_face.shape}")

    # --- Riconoscimento ---
    label, distance = recognizer.detect_unknown(
        unknown_face,
        dataset.mean_face,
        svd_reducer,
        X_train
    )

    print(f"\nDistanza minima trovata: {distance:.3f}")
    print(f"Soglia attiva: {recognizer.unknown_threshold:.3f}")

    if label == "UNKNOWN":
        print("Volto NON riconosciuto (sconosciuto)")
    else:
        print(f"Volto riconosciuto come ID: {label}")

    # --- Ottimizzazione soglia ---
    print("\nOttimizzazione soglia unknown detection...")
    recognizer.optimize_unknown_threshold(X_train, X_test)

    # --- Ri-test con soglia aggiornata ---
    label, distance = recognizer.detect_unknown(
        unknown_face,
        dataset.mean_face,
        svd_reducer,
        X_train
    )

    # --- Visualizzazione ---
    viz.plot_new_faces(
        unknown_face=unknown_face,  # array flatten 1x4096
        X=dataset.X,  # immagini originali shape (n_samples, h, w)
        distance=distance,  # distanza calcolata
        label=label,  # 'UNKNOWN' o ID predetto
        th=recognizer.unknown_threshold  # soglia utilizzata
    )

    print(f"\nTest con soglia:")
    print(f"  Distanza: {distance:.3f} | Soglia: {recognizer.unknown_threshold:.3f}")
    if label == "UNKNOWN":
        print("Volto NON riconosciuto (sconosciuto)\n")
    else:
        print(f"Volto riconosciuto come ID: {label}\n")

    # === RIEPILOGO FINALE ===
    accuracy = np.mean(y_pred == y_test)

    print("\n" + "=" * 80)
    print("  ANALISI COMPLETATA")
    print("=" * 80 + "\n")

    print(f"\nAccuracy complessiva: {accuracy * 100:.2f}%")
    print(f"Componenti mantenute: {n_components}")
    print(f"Energia preservata: {energy[n_components - 1] * 100:.2f}%")
    print(f"Riduzione dimensionale: {dataset.X_flat.shape[1]}; {n_components}")


if __name__ == "__main__":
    main()