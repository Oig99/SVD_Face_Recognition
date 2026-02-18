# Face Recognition con SVD — Olivetti vs LFW

> Pipeline completa di riconoscimento facciale basata su Singular Value Decomposition (SVD), applicata e confrontata su due dataset con caratteristiche radicalmente diverse.

---

## Indice

1. [Panoramica del Progetto](#panoramica-del-progetto)
2. [Fondamenti Teorici](#fondamenti-teorici)
3. [Struttura del Progetto](#struttura-del-progetto)
4. [Dataset](#dataset)
5. [Pipeline](#pipeline)
6. [Risultati e Confronto](#risultati-e-confronto)
7. [Conclusioni e Possibili Miglioramenti](#conclusioni-e-possibili-miglioramenti)
8. [Installazione e Utilizzo](#installazione-e-utilizzo)

---

## Panoramica del Progetto

Questo progetto implementa un sistema di riconoscimento facciale classico basato su tecniche di algebra lineare e machine learning. Il cuore della pipeline è la **Singular Value Decomposition (SVD)**, che permette di ridurre drasticamente la dimensionalità delle immagini preservando le informazioni discriminanti. I dati proiettati nello spazio ridotto vengono poi classificati tramite KNN e SVM.

L'obiettivo principale non è solo ottenere alta accuracy, ma **confrontare due scenari realisticamente diversi** — immagini controllate (Olivetti) vs immagini eterogenee reali (LFW) — e quantificare il costo computativo e prestazionale di questa complessità aggiuntiva.

---

## Fondamenti Teorici

### Singular Value Decomposition (SVD)

Data una matrice $X \in \mathbb{R}^{m \times n}$ in cui ogni riga rappresenta un'immagine appiattita, la SVD decompone:

$$X = U \Sigma V^T$$

- **$U \in \mathbb{R}^{m \times m}$** — vettori singolari sinistri: coordinate di ogni immagine nello spazio ridotto
- **$\Sigma \in \mathbb{R}^{m \times n}$** — matrice diagonale dei valori singolari $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$, ordinati per importanza decrescente
- **$V^T \in \mathbb{R}^{n \times n}$** — vettori singolari destri: le **eigenfaces**, visualizzabili come pattern facciali primitivi

L'**energia cumulativa** delle prime $k$ componenti misura la quota di varianza catturata:

$$E_k = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{n} \sigma_i^2}$$

Si seleziona il minimo $k$ tale che $E_k \geq 0.95$, garantendo di preservare il 95% dell'informazione con una frazione delle feature originali.

#### Perché funziona per le immagini di volti?

I valori singolari decadono rapidamente: i primi componenti catturano la struttura globale (forma del viso, distribuzione luminosa), mentre i successivi codificano dettagli sempre più fini e rumore. Questa struttura gerarchica dell'informazione fa sì che una riduzione aggressiva — mantenendo solo i primi $k$ componenti — si traduca in una perdita minima di informazione discriminante.

### Centratura dei Dati

Prima della SVD, si sottrae il **volto medio** $\bar{x}$ a ciascuna immagine:

$$X_{\text{centered}} = X - \bar{x}$$

Questa operazione è fondamentale: senza centratura, la prima componente principale descriverebbe semplicemente il valor medio dei pixel — irrilevante ai fini della discriminazione. La centratura sposta l'analisi sulle *differenze* rispetto al volto tipico, rendendo le componenti estratte realmente discriminanti.

### Eigenfaces

Le righe di $V^T$ sono le **eigenfaces**: ogni riga è un'immagine che cattura un modo specifico in cui i volti del dataset variano tra loro. Ogni immagine reale si rappresenta come combinazione lineare di queste basi:

$$x \approx \bar{x} + \sum_{i=1}^{k} w_i \cdot v_i$$

Il vettore dei pesi $\mathbf{w} = (w_1, \ldots, w_k)$ è la firma compatta dell'immagine nello spazio ridotto, ed è questo vettore che viene effettivamente passato al classificatore.

### K-Nearest Neighbors (KNN)

Con $k=1$, il KNN assegna a un campione la classe del suo vicino più prossimo nel training set, misurata tramite distanza di Manhattan nello spazio SVD. Il modello è semplice, non parametrico, e sorprendentemente competitivo quando lo spazio ridotto è già ben strutturato. La sua debolezza emerge in scenari ad alta variabilità intra-classe, dove la prossimità geometrica non implica necessariamente stessa identità.

### Support Vector Machines (SVM)

L'SVM cerca l'iperpiano di separazione che massimizza il margine tra le classi. Nelle versioni multi-classe viene estesa tramite strategia One-vs-One. I kernel testati sono:

- **Lineare** — $K(x_i, x_j) = x_i^T x_j$: efficiente e interpretabile, funziona bene quando lo spazio SVD è già linearmente separabile.
- **RBF (Radial Basis Function)** — $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$: mappa implicitamente i dati in uno spazio a dimensione superiore, catturando relazioni non lineari.

Il parametro $C$ regola il trade-off tra massimizzazione del margine e tolleranza agli errori di training. La sua ottimizzazione via **Grid Search con 5-fold cross-validation** è essenziale per evitare overfitting.

### Unknown Detection

Per rilevare volti non appartenenti al training set, si calcola la distanza euclidea minima tra il campione e tutti gli esempi di training nello spazio SVD. Se questa distanza supera una soglia $\theta$:

$$\theta = \mu_d + 2\sigma_d$$

dove $\mu_d$ e $\sigma_d$ sono media e deviazione standard delle distanze sul validation set, il volto viene dichiarato **UNKNOWN** anziché essere assegnato a forza a una classe. Questa soglia corrisponde circa al 95° percentile della distribuzione delle distanze osservate.

---

## Struttura del Progetto

```
face-recognition-svd/
│
├── src/
│   ├── data_loader.py       # Caricamento Olivetti e LFW, centratura, split
│   ├── svd_engine.py        # SVD completa, selezione componenti, TruncatedSVD
│   ├── recognizer.py        # KNN, SVM, cross-validation, unknown detection
│   └── visualizer.py        # Tutte le visualizzazioni e salvataggio Excel
│
├── result/
│   ├── olivetti/            # Output grafici e report dataset Olivetti
│   └── lfw/                 # Output grafici e report dataset LFW
│
├── pipeline_oliv.py         # Pipeline completa per Olivetti (script standalone)
├── pipeline_lfw.py          # Pipeline completa per LFW (script standalone)
├── face_recognition_project.ipynb  # Notebook con confronto integrato
│
└── README.md
```

### Moduli principali

**`SVDReducerEngine`** — gestisce tutta la parte algebrica: calcola la SVD completa tramite `numpy.linalg.svd`, determina il numero di componenti necessarie per raggiungere la soglia energetica, e applica la riduzione tramite `TruncatedSVD` di scikit-learn per efficienza computazionale.

**`FaceRecognizer`** — incapsula KNN e SVM, espone metodi per training, valutazione, cross-validation stratificata, analisi degli errori, ottimizzazione iperparametri via Grid Search e rilevamento di volti sconosciuti.

**`Visualizer`** — genera e salva tutti i grafici (eigenfaces, energia cumulativa, proiezione 2D, matrice di confusione, distribuzione distanze, confronto ricostruzione) e i report in formato Excel.

---

## Dataset

### Olivetti Faces

| Proprietà | Valore |
|---|---|
| Sorgente | AT&T Laboratories Cambridge |
| Totale immagini | 400 |
| Soggetti distinti | 40 |
| Immagini per soggetto | 10 |
| Risoluzione | 64 × 64 pixel |
| Features originali | 4096 |
| Gamma valori | [0, 1] (normalizzati) |
| Condizioni di acquisizione | Controllate — sfondo uniforme, illuminazione costante |

Il dataset Olivetti è il benchmark classico per sistemi di face recognition basati su eigenfaces. Le immagini variano leggermente in espressione facciale (occhi aperti/chiusi, sorriso), inclinazione della testa e presenza di occhiali, ma le condizioni di acquisizione rimangono costanti. Questa regolarità lo rende ideale per validare la pipeline e misurare la performance baseline.

### LFW (Labeled Faces in the Wild)

| Proprietà | Valore |
|---|---|
| Sorgente | University of Massachusetts |
| Totale immagini (versione filtrata) | 1348 |
| Soggetti distinti | 8 |
| Risoluzione | ~62 × 47 pixel (variabile) |
| Features originali | 1850 |
| Condizioni di acquisizione | Non controllate — illuminazione e pose variabili |

Il dataset LFW è progettato esplicitamente per testare sistemi di riconoscimento in condizioni realistiche. La versione usata in questo progetto è filtrata per includere solo soggetti con almeno 70 immagini, per garantire un bilanciamento sufficiente. La distribuzione delle classi rimane sbilanciata rispetto a Olivetti.

---

## Pipeline

```
Immagini raw
     │
     ▼
Flatten (H×W → vettore 1D)
     │
     ▼
Centratura (X - mean_face)
     │
     ▼
SVD Completa → calcolo energia
     │
     ▼
Selezione componenti (soglia 95%)
     │
     ▼
TruncatedSVD → spazio ridotto
     │
     ├──► Ricostruzione volti (verifica MSE)
     │
     ▼
Train/Test Split (80/20 stratificato)
     │
     ├──► KNN (k=1, Manhattan)
     ├──► SVM Lineare (C=1)
     ├──► SVM RBF (C=10, γ=scale)
     └──► SVM Grid Search (CV=5)
          │
          ▼
     Metriche, cross-validation, matrice di confusione
          │
          ▼
     Unknown Detection (soglia distanze)
```

---

## Risultati e Confronto

### Riduzione Dimensionale

| Metrica | Olivetti | LFW |
|---|---|---|
| Features originali | 4096 | 1850 |
| Componenti SVD (95% energia) | **123** | **167** |
| Fattore di compressione | **33.3×** | **11.1×** |
| MSE ricostruzione medio | 0.000960 | 0.001104 |

Olivetti richiede meno componenti in valore assoluto — nonostante abbia più feature originali — perché la sua struttura è molto più regolare. LFW, più variabile, richiede più componenti per catturare la stessa percentuale di varianza, e il fattore di compressione è tre volte inferiore. In entrambi i casi il MSE di ricostruzione è trascurabile, confermando che il 5% di energia scartato non contiene informazioni discriminanti rilevanti.

### Performance dei Classificatori

| Modello | Olivetti | LFW | Δ |
|---|---|---|---|
| KNN (k=1) | 94.00% | 62.02% | −31.98 pp |
| SVM Lineare | 94.00% | **81.01%** | −12.99 pp |
| SVM RBF | 92.00% | 79.82% | −12.18 pp |
| SVM Best (Grid Search) | **94.00%** | 80.42% | −13.58 pp |

Il pattern è chiaro: su Olivetti KNN è competitivo come SVM (parità a 94%), mentre su LFW il gap si apre drammaticamente — quasi 19 punti percentuali tra KNN e SVM lineare. La ragione è geometrica: in uno spazio a bassa variabilità come Olivetti, la prossimità euclidea è un segnale affidabile di identità; su LFW la distribuzione di ogni classe è molto più dispersa, e la prossimità locale non è più sufficiente — serve un separatore globale come SVM.

Sorprendentemente, su LFW il kernel RBF (79.82%) non supera il lineare (81.01%). Questo suggerisce che nello spazio SVD ridotto la struttura del problema è già prevalentemente lineare: la non-linearità del kernel RBF non aggiunge valore, e anzi può introdurre overfitting locale.

### Cross-Validation (KNN, 5-fold)

| Metrica | Olivetti | LFW |
|---|---|---|
| Mean Accuracy | 91.67% | 62.11% |
| Std Accuracy | ±2.79% | ±2.82% |
| Intervallo di confidenza (95%) | [86.09%, 97.25%] | [56.47%, 67.75%] |

La deviazione standard è simile in valore assoluto sui due dataset. Tuttavia, il coefficiente di variazione (Std / Mean) è del 3.0% su Olivetti e del 4.5% su LFW — segnale che la performance su LFW dipende maggiormente da quale specifico sottoinsieme finisce nel fold di training. Il modello Olivetti è più stabile perché le 10 immagini per persona sono più omogenee tra loro.

### Tempi di Calcolo

| Fase | Olivetti | LFW |
|---|---|---|
| SVD completa | 0.84 s | 1.41 s |
| Riduzione dimensionale | 0.31 s | 0.40 s |

I tempi rimangono contenuti su entrambi i dataset grazie all'uso di `TruncatedSVD` (algoritmo randomizzato) per la fase di trasformazione. Il collo di bottiglia è il Grid Search SVM, che con 5-fold e 16 combinazioni di iperparametri richiede qualche minuto su LFW.

### Errore di Ricostruzione

| Metrica | Olivetti | LFW |
|---|---|---|
| MSE medio | 0.000960 | 0.001104 |
| Std MSE | — | — |

L'errore di ricostruzione è basso su entrambi i dataset. Su Olivetti la distribuzione degli errori è molto compatta (varianza ridotta), mentre su LFW è più dispersa — alcune immagini con pose o illuminazioni estreme vengono ricostruite peggio. Questo conferma che il 95% di energia catturato contiene la quasi totalità dell'informazione strutturale discriminante.

---

## Conclusioni e Possibili Miglioramenti

### Conclusioni

**La SVD è uno strumento di riduzione efficace e affidabile.** In entrambi gli scenari riesce a comprimere l'informazione in modo significativo (33× su Olivetti, 11× su LFW) con perdita trascurabile di qualità ricostruttiva. L'approccio basato su eigenfaces rimane una baseline solida e interpretabile.

**Il costo delle condizioni realistiche è quantificabile e consistente.** Il passaggio da un dataset controllato (Olivetti) a uno "in the wild" (LFW) produce un calo di 13-19 punti percentuali di accuracy a seconda del classificatore. Non è un fallimento della metodologia, ma una misura diretta dell'aumento di complessità del problema.

**SVM batte KNN non appena la variabilità intra-classe aumenta.** Su Olivetti i due modelli si equivalgono. Su LFW SVM lineare supera KNN di quasi 19 punti, dimostrando che una frontiera di decisione globale è molto più robusta della prossimità locale quando le classi si sovrappongono nello spazio delle feature.

**Il kernel lineare è sufficiente nello spazio SVD.** La proiezione SVD linearizza già in larga misura la struttura delle classi: aggiungere non-linearità con RBF non porta benefici misurabili, e in certi casi penalizza leggermente le performance.

**La cross-validation conferma la solidità dei modelli.** La deviazione standard delle accuracy è limitata su entrambi i dataset, indicando che le performance osservate non sono artefatti del particolare split train/test.

### Possibili Miglioramenti

**Preprocessing più aggressivo su LFW.** Normalizzazione dell'istogramma (CLAHE), allineamento facciale tramite detezione dei landmark (occhi, naso, bocca) e ritaglio standardizzato ridurrebbero la varianza intra-classe causata da illuminazione e posa.

**Data augmentation.** Rotazioni, flip orizzontali, jitter di luminosità e zoom potrebbero aumentare artificialmente la dimensione del training set su LFW, dove alcune classi hanno pochi esempi, rendendo i modelli più robusti alle trasformazioni geometriche e fotometriche.

**Tecniche di bilanciamento delle classi.** Su LFW la distribuzione delle classi non è uniforme. Tecniche come SMOTE (Synthetic Minority Oversampling) o class weighting nell'SVM potrebbero ridurre il bias del classificatore verso le classi più rappresentate.

**Deep learning per feature extraction.** Reti pre-addestrate come FaceNet, ArcFace o DeepFace producono embedding in 128-512 dimensioni con proprietà metriche ottimali (stessa classe → distanze piccole, classi diverse → distanze grandi). Combinare questi embedding con un classificatore SVM o un classificatore coseno potrebbe portare le accuracy su LFW oltre il 95%.

**Confronto diretto SVD vs PCA.** PCA (analisi delle componenti principali tramite decomposizione degli autovalori della matrice di covarianza) è la procedura statistica equivalente alla SVD sulla matrice centrata. Un confronto diretto in termini di componenti selezionate, qualità di ricostruzione e accuracy finale potrebbe quantificare le differenze computative e numeriche tra i due approcci.

**Soglia unknown adattiva.** La soglia attuale (95° percentile delle distanze) è calcolata globalmente. Una versione per-classe — dove ogni classe ha la propria soglia calibrata sulle distanze intra-classe — potrebbe migliorare sensibilmente il rilevamento di volti sconosciuti, specialmente su LFW dove le distanze variano molto tra classi.

**Metriche di distanza alternative.** Sostituire la distanza euclidea con la distanza di Mahalanobis (che tiene conto della covarianza delle feature) o il cosine similarity potrebbe rendere il KNN e la soglia unknown più robusti alla varianza non uniforme nello spazio SVD.

---

## Installazione e Utilizzo

### Requisiti

```bash
pip install numpy scikit-learn matplotlib pandas openpyxl pillow
```

### Esecuzione Pipeline

```bash
# Dataset Olivetti
python pipeline_oliv.py

# Dataset LFW
python pipeline_lfw.py
```

### Notebook Interattivo

```bash
jupyter notebook face_recognition_project.ipynb
```

I risultati (grafici e report Excel) vengono salvati nelle cartelle:
- `result/olivetti/` per il dataset Olivetti
- `result/lfw/` per il dataset LFW

### Struttura degli Output

Ogni pipeline genera:
- `sample_faces.png` — campioni di volti dal dataset
- `mean_faces.png` — volto medio calcolato
- `eigenfaces.png` — prime 10 eigenfaces
- `cumulative_energy.png` — energia cumulativa SVD
- `projection.png` — proiezione 2D delle prime componenti
- `all_reconstructions.png` — confronto originali vs ricostruiti
- `confusion_matrix.png` — matrice di confusione del miglior modello
- `distance_distribution.png` — distribuzione distanze per unknown detection
- `reconstruction_error.png` — distribuzione MSE per campione
- `svm_report.xlsx` — classification report SVM RBF
- `model_comparison.xlsx` — confronto accuracy e tempi tra modelli
- `report.xlsx` — report completo KNN
- `result.xlsx` — risultati cross-validation

---

## Riferimenti

- Turk, M., & Pentland, A. (1991). *Eigenfaces for Recognition*. Journal of Cognitive Neuroscience, 3(1), 71–86.
- Huang, G. B., et al. (2007). *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*. University of Massachusetts, Amherst.
- Samaria, F. S., & Harter, A. C. (1994). *Parameterisation of a stochastic model for human face identification*. AT&T Laboratories Cambridge.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830.
