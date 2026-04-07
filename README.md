# Computer Vision - Energy Consumption Forecasting

Progetto universitario per l'esame di *Computer Vision*.

## Panoramica del Progetto

Il progetto è strutturato in due componenti principali:

### 1. Esperimenti di Previsione Energetica (`src/`)
Esperimenti di previsione dei consumi energetici basandosi su dati storici e caratteristiche temporali. Testa il modello proposto (CNN-3D) con altri modelli SOTA quali XGBoost, LSTM e TCN.

### 2. Architettura Sicura di Raccolta Dati (`architecture/`)
Stack completo containerizzato con MQTT, PostgreSQL, Node-RED e Grafana per la raccolta, elaborazione e visualizzazione sicura dei dati di consumo. Diviso un una parte non sicura (`before/`) ed una sicura (`/after`).

---

## Parte 1: Esperimenti di Previsione

### Struttura

```
src/computer_vision/
├── model/
│   └── cnn3d.py        # Modello proposto
├── preprocessing/      # Pipeline di preprocessing
│   ├── interim.py
│   └── final.py
├── transformer/
│   └── cyclical_encoding.py  # Encoding ciclico per features temporali
├── dataset.py          # Caricamento dati
├── train.py            # Logica di training dei modelli
├── test.py             # Logica di testing dei modelli
├── evaluate.py
├── forecast.py         # Funzione CLI per la previsione. Usato nell'immagine Docker
├── forecasters.py
├── metrics.py          # Funzioni di calcolo metriche
├── export.py           # Export dati
├── plot.py             # Visualizzazioni
├── config.py           # Configurazioni globali
├── utils.py            # Utility functions
└── cli.py              # Interfaccia CLI
```

### Dataset

Nel lavoro è stato usato il dataset [ECD-UY](https://figshare.com/collections/_/5428608), in particolare della sezione THC, composto dai consumi di ~110.000 utenze dell'Uruguay.

I dati sono organizzati in tre livelli di elaborazione:
- `data/1.raw/ECD-UY`: File grezzi. Dovrebbero essere 23 file `.csv.tar.gz` più il file `customers.csv`
- `data/2.interim/ECD-UY`: Dati preprocessati intermedi
- `data/3.final/ECD-UY`: Dataset finale pronto per il training
- `output/`: Risultati del training (train/test splits)

### Utilizzo

Il lavoro è gestito da `dvc`. Per riprodurre la pipeline basta usare

```shell
dvc dag # Mostra gli stage disponibili

dvc repro <stage_1> ... <stage_n> # Riproduce fino a degli stage, tipicamente plot e report
```

Per eseugire un esperimento usare

```shell
dvc exp run -S "param_1=value_1" ... train
dvc exp run -S "param_1=value_1" ... test
```

Si faccia riferimento al file `params.yaml` per i parametri disponibili

### Modelli disponibili

Tutti implementati in sktime.

- **CNN 3D**: Modello proposto, con la creazione di una "matrice" dei consumi per analizzare pattern settimanali con delle CNN
- **XGBoost**
- **LSTM**
- **TCN**

---

## Parte 2: Architettura Sicura di Raccolta Dati

### Panoramica Stack

L'architettura è divisa in due fasi:

#### **Before** (`architecture/before/`)
Configurazione base con:
- **Misuratori**: Cinque misuratori che inviano dati all'architettura. Ripropongono il test set usato negli esperimenti
- **Mosquitto**: Message broker MQTT per raccolta dati
- **Node-RED**: Raccolta dati dal broker e salvataggio sul DB
- **PostgreSQL**: Database per storage persistente
- **Grafana**: Visualizzazione di dati e previsioni

#### **After** (`architecture/after/`)
Architettura messa in sicurezza con:
- **Certificati SSL/TLS**: Comunicazione cifrata tra un servizio e l'altro, e tra misuratori e broker.
- **MQTT Secure**: Mosquitto con autenticazione e ACL
- **PostgreSQL Sicuro**: Configurazione con SSL e ruoli
- **Node-RED e Grafana Protetti**: Certificati client per l'autenticazione

### Componenti Principali

```
architecture/after/
├── docker-compose.yaml       # Orchestrazione container
├── postgres.dockerfile       # Immagine PostgreSQL personalizzata
├── generate_certs.sh         # Script generazione certificati SSL/TLS
├── certs/                    # Certificati CA e endpoint
├── config/                   # Configurazioni per la generazione certificati con OpenSSL
│   ├── mosquitto.conf
│   ├── postgres.conf
│   ├── grafana.conf
│   ├── meter-*.conf
│   ├── modello-cnn3d.conf
│   └── nodered.conf
└── data/                     # Dati passati ai container
    ├── meter/                # Dati da far inviare ai sensori, ottenuti tramite comando export.
    ├── forecaster/           # Modello di previsione da caricare nel container
    └── node-red/             # Dati di Node-RED
```

### Avvio dell'Architettura

`taskfile.yaml` raccoglie i comandi per mettere su l'architettura:

- `certs`: Genera i certificati con `generate_-_certs.sh`
- `build-meter`: Costruisce l'immagine Docker per i meter
- `build-forecaster`: Costruisce l'immagine Docker per il modello di previsione
- `architecture-after-up`: Mette su l'architettura con Docker compose

### Note

Si consideri che, per semplicità, utenze e password di amministratore sono contenute direttamente in `docker-compose.yaml`. In un contesto reale le credenziali non vanno salvate in questo modo, visto che verrebbero versionate nel repository del codice sorgente. Piuttosto, tali credenziali dovrebbero essere gestite direttamente sulle macchine che faranno girare i container tramite variabili d'ambiente o, meglio, tramite sei servizi per la gestione dei segreti.

---

## 🛠️ Configurazione Progetto

### Dipendenze

Il progetto fa uso di `uv` per la gestione delle dipendenze e dei pacchetti. Far riferimento al file `pyproject.toml` per una lista delle dipendenze.

Usare `uv sync` per creare un ambiente virtuale, e `uv run computer-vision <comando>` per eseguire un comando della CLI del progetto (usato raramente, visto che la maggior parte dei comandi è usata direttamente dagli stage definiti con `dvc`).
