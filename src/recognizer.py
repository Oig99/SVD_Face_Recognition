import time
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC


class FaceRecognizer:
    """
    Classe responsabile di:
    - Addestramento e valutazione KNN
    - Addestramento e ottimizzazione SVM
    - Analisi degli errori
    - Stima soglia per unknown detection
    - Confronto tra classificatori
    """
    def __init__(self, n_neighbors=1, unknown_threshold=0.5, kernel='rbf', C=0.1, gamma='scale', metric="manhattan", wgs="uniform", singular_values=None):
        self.y_train = None
        self.n_neighbors = n_neighbors # Parametri del classificatore
        self.unknown_threshold = unknown_threshold # Soglia per decidere se un volto è sconosciuto
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=metric, weights=wgs)
        self.svm = SVC(kernel="linear", C=C, gamma=gamma, probability=True)
        self.svm_rbf = SVC(kernel=kernel, C=10, gamma=gamma, probability=True)
        self.singular_values = singular_values

    # ---------------------------- KNN ----------------------------

    def train_knn(self, X_train, y_train):
        """
        Addestra il KNN nello spazio SVD.

        KNN è un metodo non parametrico: memorizza i campioni e basa la decisione sulla prossimità locale.
        """
        self.knn.fit(X_train, y_train)
        self.y_train = y_train

    def evaluate_knn(self, X_test):
        """
        Esegue predizione su test set. Restituisce le etichette predette.
        """
        y_pred = self.knn.predict(X_test)
        return y_pred

    @staticmethod
    def compute_min_distances(X_test, X_train):
        """
        Calcola, per ogni punto di test, la distanza minima dai campioni del training set.
        Utile per analisi soglia unknown.
        Geometricamente misura quanto un punto è "vicino" alla distribuzione delle classi note.
        """
        distances = euclidean_distances(X_test, X_train)
        distances = np.min(distances, axis=1)
        return distances

    # ---------------------------- Unknown detection ----------------------------

    def detect_unknown(self, face, mean_face, svd_reducer, X_train):
        """
        Data una faccia (array flatten 1xN), la centra, la riduce con SVD
        e calcola la distanza minima euclidea dai campioni di training.
        Restituisce il label ('UNKNOWN' o ID) e la distanza minima.
        """
        face_reduced = svd_reducer.transform(face - mean_face)

        # distances = np.linalg.norm(X_train - face_reduced, axis=1)
        # min_dist = np.min(distances)
        distances = self.compute_min_distances(face_reduced, X_train)
        min_dist = distances[0]

        if min_dist > self.unknown_threshold:
            return "UNKNOWN", min_dist
        else:
            predicted_id = self.knn.predict(face_reduced)[0]
            return predicted_id, min_dist

    def simulate_unknown_detection(self, X_flat, X_train, mean_face, svd_reducer, seed=0):
        """
        Sanity check: genera un volto sintetico di puro rumore casuale
        e verifica che venga correttamente rifiutato come sconosciuto.
        Un volto di rumore DEVE sempre restituire 'UNKNOWN'.
        """
        np.random.seed(seed)
        unknown_face = np.random.rand(1, X_flat.shape[1])

        # Riusa detect_unknown per coerenza — stessa logica, stessa metrica
        label, min_dist = self.detect_unknown(unknown_face, mean_face, svd_reducer, X_train)

        print(f"Distanza minima trovata: {min_dist:.3f}")
        print(f"Soglia attiva: {self.unknown_threshold:.3f}")

        if label == "UNKNOWN":
            print("Volto NON riconosciuto (UNKNOWN) — comportamento corretto")
        else:
            print(f"Volto riconosciuto come ID: {label} — soglia probabilmente troppo alta!")

        # Riduzione SVD per eventuale uso esterno (es. Visualizzazione)
        unknown_svd = svd_reducer.transform(unknown_face - mean_face)

        return unknown_face, unknown_svd, min_dist, label

    def optimize_unknown_threshold(self, X_train, X_val):
        """
        Calcola automaticamente la soglia ottimale per unknown detection
        basandosi sulla distribuzione delle distanze minime sul validation set.
        Soglia = media + 2 * deviazione standard (copre ~97.5% dei volti noti).

        Nota: questo metodo calibra la soglia solo sui NOTI.
        Per una calibrazione più precisa che usa anche esempi di sconosciuti reali,
        usa calibrate_threshold_with_unknowns().
        """
        distances = self.compute_min_distances(X_val, X_train)

        mean_d = float(np.mean(distances))
        std_d = float(np.std(distances))
        optimal_threshold = mean_d + 2 * std_d
        self.unknown_threshold = optimal_threshold

        result = {
            'optimal_threshold': optimal_threshold,
            'distance_stats': {
                'mean': mean_d,
                'std': std_d,
                'median': float(np.median(distances)),
                'p95': float(np.percentile(distances, 95))
            }
        }

        print(f"\nSoglia (mean+2σ): {optimal_threshold:.3f}")
        print(
            f"   mean={mean_d:.3f} | std={std_d:.3f} | "
            f"median={result['distance_stats']['median']:.3f} | "
            f"p95={result['distance_stats']['p95']:.3f}"
        )

        return result

    # ---------------------------- Cross Validation e Analisi Errori ----------------------------

    def cross_validate(self, X, y, cv=5):
        """ Esegue una valutazione della robustezza del modello tramite Stratified K-Fold cross-validation."""

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.knn, X, y, cv=skf, scoring='accuracy')

        result = {
            'mean_accuracy': float(scores.mean()),
            'std_accuracy': float(scores.std()),
            'confidence_interval inf': float(scores.mean() - 2 * scores.std()),
            'confidence_interval sup': float(scores.mean() + 2 * scores.std()),
        }
        for i, score in enumerate(scores, start=1):
            result[f'score_{i}'] = float(score)
        return result

    def cross_validate_svm(self, X, y, cv=5):
        """ Esegue una valutazione della robustezza del modello tramite Stratified K-Fold cross-validation per svm"""

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.svm, X, y, cv=skf, scoring='accuracy')

        result = {
            'mean_accuracy': float(scores.mean()),
            'std_accuracy': float(scores.std()),
            'confidence_interval inf': float(scores.mean() - 2 * scores.std()),
            'confidence_interval sup': float(scores.mean() + 2 * scores.std()),
        }
        for i, score in enumerate(scores, start=1):
            result[f'score_{i}'] = float(score)
        return result

    def analyze_misclassifications(self, X_test, y_test, y_pred):
        """
        Analizza gli errori per identificare pattern sistematici.
        Identifica:
        - campioni mal classificati
        - coppie di classi più frequentemente confuse
        """
        errors = defaultdict(list)

        for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
            if true_label != pred_label:
                distances, _ = self.knn.kneighbors([X_test[i]])

                errors['misclassified_samples'].append({
                    'index': i,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'nearest_distance': distances[0][0]
                })

        confusion_pairs = {}
        for error in errors['misclassified_samples']:
            pair = (error['true_label'], error['predicted_label'])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        errors['most_confused_pairs'] = sorted(
            confusion_pairs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return errors

    # ---------------------------- Ottimizzazione Iperparametri ----------------------------

    def optimize_hyperparameters(self, X_train, y_train):
        """Trova automaticamente i migliori iperparametri.
        Ricerca dei migliori iperparametri KNN tramite GridSearch.

        Esplora:
        - numero di vicini
        - metrica di distanza
        - schema di pesatura
        """
        param_grid = {
            'n_neighbors': [1, 3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'cosine']
        }

        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Aggiorna il modello con i parametri migliori
        self.knn = grid_search.best_estimator_
        self.n_neighbors = grid_search.best_params_['n_neighbors']

        return grid_search.best_params_, grid_search.best_score_


    def predict_with_confidence(self, X_test):
        """Predizioni con score di confidenza."""
        distances, indices = self.knn.kneighbors(X_test)
        predictions = self.knn.predict(X_test)

        results = []

        for i, pred in enumerate(predictions):
            avg_distance = distances[i].mean()
            distance_confidence = 1 / (1 + avg_distance)

            neighbor_labels = self.y_train[indices[i]]
            consensus = (neighbor_labels == pred).sum() / len(neighbor_labels)

            combined_confidence = 0.5 * distance_confidence + 0.5 * consensus

            results.append({
                'prediction': pred,
                'confidence': combined_confidence,
                'avg_distance': avg_distance,
                'neighbor_consensus': consensus
            })

        return results


    # ---------------------------- SVM ----------------------------

    def train_svm(self, X_train, y_train):
        """
        Addestra SVM nello spazio SVD.

        L'SVM costruisce un iperpiano con margine massimo tra le classi.
        """
        self.svm.fit(X_train, y_train)

    def evaluate_svm(self, X_test):
        """
        Predizione tramite SVM già addestrata.
        """
        if self.svm is None:
            raise ValueError("SVM non addestrata. Chiama train_svm prima.")
        return self.svm.predict(X_test)

    def optimize_svm(self, X_train, y_train, cv=5):
        """
        Ricerca automatica iperparametri SVM.

        Esplora:
        - C (regolarizzazione)
        - kernel (linear, rbf)
        - gamma (per RBF)
        """
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

        grid = GridSearchCV(
            SVC(),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        self.svm = grid.best_estimator_

        support_per_classes = grid.best_estimator_.n_support_
        support_total = np.sum(support_per_classes)
        return grid.best_params_, grid.best_score_, support_total, support_per_classes

    def predict_svm_with_distance(self, face_svd):
        """
        Restituisce:
        - predizione
        - distanza dal margine decisionale

        Il valore della decision_function rappresenta
        quanto il punto è distante dall'iperpiano separatore.
        """
        if self.svm is None:
            raise ValueError("SVM non addestrata.")

        decision = self.svm.decision_function(face_svd)
        prediction = self.svm.predict(face_svd)[0]

        margin = np.max(decision)

        return prediction, margin


    def predict_with_confidence_svm(self, X_test_svd):
        """
        Restituisce predizione SVM + confidence nello spazio SVD.
        X_test_svd deve essere già ridotto con svd_reducer.transform
        """
        results = []
        for face_svd in X_test_svd:
            face_svd = face_svd.reshape(1, -1)
            probs = self.svm.predict_proba(face_svd)[0]
            pred_class = self.svm.classes_[np.argmax(probs)]
            confidence = np.max(probs)
            results.append({
                'prediction': pred_class,
                'confidence': confidence
            })
        return results

    # ---------------------------- compare classifiers ----------------------------

    def compare_classifiers(self, X_train, y_train, X_test, y_test):
        """
        Confronta le performance tra:
        - KNN
        - SVM lineare
        - SVM RBF

        Restituisce un dizionario con accuracy, precision, recall, f1-score,
        classification report e tempi.
        """
        results = {}

        # ===================== KNN =====================
        start = time.time()

        self.train_knn(X_train, y_train)
        y_pred_knn = self.knn.predict(X_test)

        knn_time = time.time() - start

        knn_acc = accuracy_score(y_test, y_pred_knn)
        knn_prec = precision_score(y_test, y_pred_knn, average='weighted')
        knn_rec = recall_score(y_test, y_pred_knn, average='weighted')
        knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')
        knn_report = classification_report(y_test, y_pred_knn, output_dict=True)

        results['KNN'] = {
            'accuracy': knn_acc,
            'precision': knn_prec,
            'recall': knn_rec,
            'f1_score': knn_f1,
            'time': knn_time,
        }

        print(
            f"KNN -> Acc: {knn_acc:.4f}, Prec: {knn_prec:.4f}, Rec: {knn_rec:.4f}, F1: {knn_f1:.4f}, Time: {knn_time:.4f}s")
        print(classification_report(y_test, y_pred_knn))

        # ===================== SVM LINEARE =====================
        start = time.time()

        svm_lin = self.svm
        svm_lin.fit(X_train, y_train)
        y_pred_lin = svm_lin.predict(X_test)

        lin_time = time.time() - start

        lin_acc = accuracy_score(y_test, y_pred_lin)
        lin_prec = precision_score(y_test, y_pred_lin, average='weighted')
        lin_rec = recall_score(y_test, y_pred_lin, average='weighted')
        lin_f1 = f1_score(y_test, y_pred_lin, average='weighted')
        lin_report = classification_report(y_test, y_pred_lin, output_dict=True)

        results['SVM Linear'] = {
            'accuracy': lin_acc,
            'precision': lin_prec,
            'recall': lin_rec,
            'f1_score': lin_f1,
            'time': lin_time,
        }

        print(
            f"SVM Linear -> Acc: {lin_acc:.4f}, Prec: {lin_prec:.4f}, Rec: {lin_rec:.4f}, F1: {lin_f1:.4f}, Time: {lin_time:.4f}s")
        print(classification_report(y_test, y_pred_lin))

        # ===================== SVM RBF =====================
        start = time.time()

        svm_rbf = self.svm_rbf
        svm_rbf.fit(X_train, y_train)
        y_pred_rbf = svm_rbf.predict(X_test)

        rbf_time = time.time() - start

        rbf_acc = accuracy_score(y_test, y_pred_rbf)
        rbf_prec = precision_score(y_test, y_pred_rbf, average='weighted')
        rbf_rec = recall_score(y_test, y_pred_rbf, average='weighted')
        rbf_f1 = f1_score(y_test, y_pred_rbf, average='weighted')
        rbf_report = classification_report(y_test, y_pred_rbf, output_dict=True)

        results['SVM RBF'] = {
            'accuracy': rbf_acc,
            'precision': rbf_prec,
            'recall': rbf_rec,
            'f1_score': rbf_f1,
            'time': rbf_time,
        }

        print(
            f"SVM RBF -> Acc: {rbf_acc:.4f}, Prec: {rbf_prec:.4f}, Rec: {rbf_rec:.4f}, F1: {rbf_f1:.4f}, Time: {rbf_time:.4f}s")
        print(classification_report(y_test, y_pred_rbf))

        return results, knn_report, lin_report, rbf_report