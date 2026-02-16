import time
from collections import defaultdict
import numpy as np
from sklearn.metrics import euclidean_distances, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC


class FaceRecognizer:
    """
    Classe responsabile di:
    - Addestramento KNN
    - Valutazione performance
    - Analisi distanze
    - Riconoscimento volti sconosciuti
    """
    def __init__(self,  n_neighbors=1, unknown_threshold=0.5, kernel='rbf', C=10, gamma='scale'):
        self.n_neighbors = n_neighbors # Parametri del classificatore
        self.unknown_threshold = unknown_threshold # Soglia per decidere se un volto è sconosciuto
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric="manhattan", weights="uniform")
        self.svm = SVC(kernel=kernel, C=C, gamma=gamma)

    def train_knn(self, X_train, y_train):
        """
        Addestra il classificatore KNN.
        """
        self.knn.fit(X_train, y_train)

    def evaluate_knn(self, X_test, y_test):
        """
        Valuta il modello e stampa precision, recall, f1-score.
        """
        y_pred = self.knn.predict(X_test)
        return y_pred

    @staticmethod
    def compute_min_distances(X_test, X_train):
        """
        Calcola la distanza minima euclidea tra ogni punto test
        e i punti nel training set.
        Utile per analisi soglia unknown.
        """
        # distances = [
        #     np.min(euclidean_distances([x], X_train))
        #     for x in X_test
        # ]
        distances = euclidean_distances(X_test, X_train)
        distances = np.min(distances, axis=1)
        return distances

    def detect_unknown(self, face, mean_face, svd_reducer, X_train):
        """
        Determina se un volto è conosciuto o sconosciuto usando KNN.
        """
        # Applica riduzione SVD se serve
        face_reduced = svd_reducer.transform(face - mean_face)

        distances = np.linalg.norm(X_train - face_reduced, axis=1)
        min_dist = np.min(distances)

        if min_dist > self.unknown_threshold:
            return "UNKNOWN", min_dist
        else:
            predicted_id = self.knn.predict(face_reduced)[0]
            return predicted_id, min_dist


    def cross_validate(self, X, y, cv=5):
        """
        Esegue cross-validation stratificata per valutare la robustezza del modello.
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.knn, X, y, cv=skf, scoring='accuracy')

        result = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'all_scores': scores,
            'confidence_interval inf': float(scores.mean() - 2 * scores.std()),
            'confidence_interval sup': float(scores.mean() + 2 * scores.std()),
        }

        return result

    def analyze_misclassifications(self, X_test, y_test, y_pred):
        """Analizza pattern negli errori."""
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

        # Coppie più confuse
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

    def optimize_hyperparameters(self, X_train, y_train):
        """Trova automaticamente i migliori iperparametri."""
        param_grid = {
            'n_neighbors': [1, 3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
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

    def predict_with_confidence(self, X_test):
        """Predizioni con score di confidenza."""
        distances, indices = self.knn.kneighbors(X_test)
        predictions = self.knn.predict(X_test)

        results = []
        for i, pred in enumerate(predictions):
            avg_distance = distances[i].mean()
            distance_confidence = 1 / (1 + avg_distance)

            neighbor_labels = self.knn._y[indices[i]]
            consensus = (neighbor_labels == pred).sum() / len(neighbor_labels)

            combined_confidence = (distance_confidence * 0.5 + consensus * 0.5)

            results.append({
                'prediction': pred,
                'confidence': combined_confidence,
                'avg_distance': avg_distance,
                'neighbor_consensus': consensus
            })

        return results

    def optimize_unknown_threshold(self, X_train, X_val):
        """Calcola automaticamente la soglia ottimale per unknown detection."""
        distances = self.compute_min_distances(X_val, X_train)

        # Usa il 95° percentile come soglia ottimale
        optimal_threshold = np.percentile(distances, 95)
        self.unknown_threshold = optimal_threshold

        return {
            'optimal_threshold': optimal_threshold,
            'distance_stats': {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'median': np.median(distances),
                'p95': optimal_threshold
            }
        }

    def train_svm(self, X_train, y_train):
        """
        Addestra un classificatore SVM.
        """
        self.svm.fit(X_train, y_train)

    def evaluate_svm(self, X_test):
        """
        Predice usando SVM.
        """
        if self.svm is None:
            raise ValueError("SVM non addestrata. Chiama train_svm prima.")
        return self.svm.predict(X_test)

    def optimize_svm(self, X_train, y_train, cv=5):
        """
        Ottimizza automaticamente SVM con GridSearch.
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

        return grid.best_params_, grid.best_score_

    def detect_unknown_svm(self, face, mean_face, svd_reducer, X_train):
        """
        Unknown detection usando SVM + distanza nello spazio SVD.
        """
        if self.svm is None:
            raise ValueError("SVM non addestrata.")

        face_centered = face - mean_face
        face_svd = svd_reducer.transform(face_centered)

        min_dist = np.min(euclidean_distances(face_svd, X_train))

        if min_dist > self.unknown_threshold:
            return "UNKNOWN", min_dist
        else:
            predicted_id = self.svm.predict(face_svd)[0]
            return predicted_id, min_dist

    def predict_svm_with_distance(self, face_svd):
        """
        Restituisce predizione SVM + distanza dal margine.
        """
        if self.svm is None:
            raise ValueError("SVM non addestrata.")

        decision = self.svm.decision_function(face_svd)
        prediction = self.svm.predict(face_svd)[0]

        margin = np.max(decision)

        return prediction, margin

    def compare_classifiers(self, X_train, y_train, X_test, y_test):
        """
        Confronta le performance tra:
        - KNN (default con n_neighbors)
        - SVM lineare
        - SVM RBF

        Restituisce un dizionario con accuracy e tempi.
        """
        results = {}

        # ===================== KNN =====================
        start = time.time()
        self.train_knn(X_train, y_train)
        y_pred_knn = self.evaluate_knn(X_test, y_test)
        knn_time = time.time() - start
        knn_acc = accuracy_score(y_test, y_pred_knn)
        results['KNN'] = {'accuracy': knn_acc, 'time': knn_time, 'y_pred': y_pred_knn}
        print(f"KNN Accuracy: {knn_acc * 100:.2f}%, Time: {knn_time:.4f}s")

        # ===================== SVM LINEARE =====================
        start = time.time()
        svm_lin = SVC(kernel='linear', C=1)
        svm_lin.fit(X_train, y_train)
        y_pred_lin = svm_lin.predict(X_test)
        lin_time = time.time() - start
        lin_acc = accuracy_score(y_test, y_pred_lin)
        results['SVM Linear'] = {'accuracy': lin_acc, 'time': lin_time, 'y_pred': y_pred_lin}
        print(f"SVM Linear Accuracy: {lin_acc * 100:.2f}%, Time: {lin_time:.4f}s")

        # ===================== SVM RBF =====================
        start = time.time()
        svm_rbf = SVC(kernel='rbf', C=10, gamma='scale')
        svm_rbf.fit(X_train, y_train)
        y_pred_rbf = svm_rbf.predict(X_test)
        rbf_time = time.time() - start
        rbf_acc = accuracy_score(y_test, y_pred_rbf)
        results['SVM RBF'] = {'accuracy': rbf_acc, 'time': rbf_time, 'y_pred': y_pred_rbf}
        print(f"SVM RBF Accuracy: {rbf_acc * 100:.2f}%, Time: {rbf_time:.4f}s")

        return results