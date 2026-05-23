# local-wakeword-trainer

Trainiert ein eigenes OpenWakeWord-kompatibles Wakeword-Modell aus Handy-Aufnahmen.

Ziel-Artefakte:

```text
models/homie.onnx
models/homie.tflite
```

Das ONNX/TFLite-Modell kann spaeter auf einem Raspberry Pi mit openWakeWord geladen werden. Die Pipeline ist nicht an Home Assistant gebunden.

## Ordner fuer Handy-Aufnahmen

Lege deine Rohaufnahmen hier ab:

```text
mobile_uploads/customword/
  positive/      Wakeword, z.B. "hey homie" oder "homie"
  negative/      Sprache ohne Wakeword, aehnliche Phrasen, normale Kommandos
  background/    Raumgeraeusche, TV, Musik, Kueche, Tastatur, Stille
  rir/           Optional: Room impulse responses als WAV
```

Alle gaengigen Handyformate wie `.m4a`, `.mp3`, `.wav`, `.flac`, `.ogg` werden rekursiv gefunden.

Empfehlung fuer einen ersten brauchbaren Lauf:

```text
positive:   mindestens 20-50 echte Clips, besser 100+
negative:   mindestens 50 Clips, besser mehrere hundert
background: mindestens 10 laengere Clips aus echten Einsatzraeumen
```

Kurze Wakewords wie `homie` sind anfaelliger fuer False Positives. `hey homie` ist meist robuster.

## Training starten

Auf der Ubuntu/RTX-Trainingsmaschine Python 3.10 verwenden:

```bash
python3.10 main.py
```

Die Pipeline macht:

```text
1. .venv erstellen
2. requirements installieren
3. mobile_uploads/customword/* nach data/raw/customword/* konvertieren
4. data/raw/customword/* nach data/augmented/customword/* augmentieren
5. OpenWakeWord-Features berechnen
6. Modell trainieren
7. ONNX und, falls moeglich, TFLite exportieren
```

Nur Setup:

```bash
python3.10 main.py --setup-only
```

Training direkt starten:

```bash
.venv/bin/python scripts/train_openwakeword.py --model-name homie --target-phrase "hey homie"
```

Feature-Dateien neu berechnen:

```bash
.venv/bin/python scripts/train_openwakeword.py --overwrite-features
```

Nur ONNX exportieren, falls TFLite-Konvertierung auf deiner Maschine zickt:

```bash
.venv/bin/python scripts/train_openwakeword.py --no-tflite
```

## Raspberry Pi Nutzung

Das fertige Modell liegt unter:

```text
models/homie.onnx
models/homie.tflite
```

In eigener Python-Logik kannst du es mit openWakeWord laden:

```python
from openwakeword.model import Model

model = Model(wakeword_models=["models/homie.tflite"])
```

Der Audiostream muss 16 kHz, mono, 16-bit PCM liefern.

## Hinweise

`ffmpeg` muss fuer die Konvertierung installiert sein.

Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg python3.10 python3.10-venv
```

Windows ist zum Sortieren/Aufnehmen okay, aber die volle Trainingspipeline ist auf Ubuntu mit Python 3.10 ausgelegt.

## Public Repo und Lizenzen

Dieses Repo enthaelt nur den Pipeline-Code. Audioaufnahmen, Feature-Dateien, Trainingsartefakte und exportierte Modelle werden absichtlich nicht versioniert:

```text
mobile_uploads/
data/
artifacts/
models/
```

Das Projekt nutzt openWakeWord als Dependency. openWakeWord-Code ist Apache-2.0-lizenziert. Die von openWakeWord bereitgestellten vortrainierten Modelle koennen abweichende, nicht-kommerzielle Lizenzbedingungen haben. Pruefe deshalb die openWakeWord-Lizenzhinweise, bevor du trainierte Modellgewichte oder Modelle kommerziell verwendest oder veroeffentlichst.

Veroeffentliche keine privaten Sprachaufnahmen, Hintergrundaufnahmen oder Modellartefakte, wenn du nicht die Rechte daran hast.
