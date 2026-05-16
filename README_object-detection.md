# Duckietown – Taak 3: Voetgangersdetectie (Duckie-ontwijking)

## Projectdoel

Het doel van deze opdracht is het voorkomen van botsingen met voetgangers (duckies) in Duckietown.

Duckies representeren voetgangers die:

- de rijbaan oversteken
- op de weg staan
- de doorgang blokkeren

De Duckiebot moet deze objecten detecteren met behulp van camerabeelden en veilig reageren door te stoppen voordat een botsing plaatsvindt.

---

## Vereisten

De implementatie moet:

- Duckies detecteren met behulp van de onboard camera
- Objectdetectie uitvoeren met een getraind machine learning model
- Botsingen voorkomen onder normale omstandigheden
- Veilig stoppen zodra een duckie op de rijbaan verschijnt

In deze implementatie is gekozen voor:

**Detecteren → Veilig stoppen**

---

## Projectstructuur

```bash
packages/object_detection/
├── __init__.py
├── config.py
└── model.py
```

Belangrijke assets:

```bash
assets/
├── best.onnx
└── classes.yaml
```

---

## Datasetverzameling

De trainingsdata voor het model is zelf verzameld binnen Duckietown.

Proces:

1. Duckiebot ingezet in Duckietown-omgeving
2. Camerabeelden verzameld van verschillende duckies
3. Meerdere hoeken, afstanden en lichtcondities gebruikt
4. Dataset opgebouwd uit realistische detectiesituaties

Hierdoor leert het model objecten herkennen in de echte testomgeving.

---

## Annotatie met SAM3

Voor het labelen van de beelden is gebruik gemaakt van het **SAM3-model (Segment Anything Model)**.

Workflow:

1. Foto’s verzameld in Duckietown
2. Objectsegmentatie uitgevoerd met SAM3
3. Bounding boxes gegenereerd
4. Labels gecontroleerd en gecorrigeerd
5. Dataset geëxporteerd voor training

Voordelen:

- snelle annotatie
- consistente labels
- minder handmatig werk

---

## Modeltraining

Het detectiemodel is getraind op de zelf verzamelde dataset.

Training pipeline:

1. Dataset voorbereiden
2. Duckie-objecten labelen
3. Train/validation split maken
4. Model trainen
5. Beste gewichten exporteren naar ONNX

Output:

```bash
assets/best.onnx
```

Deze `best.onnx` bevat de optimale modelgewichten die tijdens training zijn verkregen.

---

## Systeemwerking

### 1. Camerabeeld ontvangen

De Duckiebot ontvangt continu RGB-beelden van de onboard camera.

---

### 2. Voorbewerking

In `model.py` wordt het beeld voorbereid:

- resize naar modelinput
- BGR → RGB conversie
- normalisatie naar `[0,1]`
- tensorvorming (CHW)
- batchdimensie toevoegen

---

### 3. ONNX inference

Inference gebeurt via:

```python
onnxruntime.InferenceSession()
```

Ondersteunde execution providers:

- CUDAExecutionProvider
- CPUExecutionProvider

---

### 4. Detectiecontrole

Iedere detectie bevat:

```text
x1, y1, x2, y2, confidence, class
```

Confidence filtering:

```python
CONF_THRESHOLD = 0.1
```

Alleen voldoende zekere detecties worden geaccepteerd.

---

### 5. Veiligheidslogica

Wanneer een duckie wordt gedetecteerd:

```text
STOP
```

PWM-output:

```python
left = 0.0
right = 0.0
```

Wanneer geen object wordt gevonden:

```text
DOORRIJDEN
```

met:

```python
FORWARD_PWM = 0.3
```

---

## Configuratie

Bestand:

```bash
packages/object_detection/config.py
```

Belangrijke parameters:

| Parameter | Betekenis |
|-----------|-----------|
| `MODEL_PATH` | Pad naar ONNX-model |
| `CONF_THRESHOLD` | Minimale detectiezekerheid |
| `STOP_DISTANCE` | Toekomstige afstandsdrempel |
| `FORWARD_PWM` | Voorwaartse snelheid |

---

## Mogelijke uitbreidingen

Toekomstige verbeteringen:

- afstandsinschatting via bounding box grootte
- class-specifieke detecties
- uitwijklogica in plaats van alleen stoppen
- non-maximum suppression
- combinatie met lane following

---

## Resultaat

De Duckiebot detecteert voetgangers (duckies) succesvol en voorkomt botsingen door automatisch te stoppen wanneer een object op de rijbaan verschijnt.
