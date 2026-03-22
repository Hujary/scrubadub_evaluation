# Scrubadub Detector

Prototypische Implementierung zur lokalen Erkennung personenbezogener Daten in Texten mit Scrubadub
- Erkennung über Scrubadub (regex) + spaCy NER (LG)

---

## BEFEHLE

### Virtuelle Umgebung anlegen
```bash
python3.12 -m venv .venv
```

### Virtuelle Umgebung aktivieren
**macOS / Linux (bash/zsh)**
```bash
source .venv/bin/activate
```

### Pip Upgrade
```bash
python -m pip install --upgrade pip setuptools wheel
```

### requirements.txt verwenden
```bash
pip install scrubadub scrubadub-spacy spacy phonenumbers
python -m spacy download de_core_news_lg
```

### Installation VERIFIZIEREN
```bash
python -c "import scrubadub; print(scrubadub.__version__)"
python -c "import scrubadub_spacy; print('scrubadub_spacy ok')"
python -c "from scrubadub.filth import Filth; print('Filth ok')"
python -c "import spacy; nlp = spacy.load('de_core_news_lg'); print('spacy model ok')"
```

### Scrubadub Evaluieren
```bash
python scrubadub_detect.py
```

### Zeitmessung ausführen
```bash
python scrubadub_runtime.py --runs 5
```
