# YOLOv11 Metrics — Precision, Recall og mAP50

## Grundlæggende forståelse

Når YOLO trænes til at finde objekter (f.eks. personer), evaluerer vi modellen med tre metrics. Alle tre er tal mellem **0.0 og 1.0** (svarende til 0%–100%).

---

## Precision

> *Af alle de gange modellen sagde "her er et objekt!" — hvor mange gange havde den ret?*

**Formel:**
```
Precision = Korrekte fund / Alle modellens gæt
```

**Eksempel:**
Modellen laver 40 bounding boxes, og 35 er korrekte.
```
Precision = 35 / 40 = 0.875
```

**Hvornår er høj Precision vigtigst?**
Når det er dyrt at have *falske alarmer* — f.eks. en parkeringsvagt der ikke vil råbe af folk der holder lovligt.

---

## Recall

> *Af alle de rigtige objekter der faktisk var i billederne — hvor mange fandt modellen?*

**Formel:**
```
Recall = Korrekte fund / Alle rigtige objekter i virkeligheden
```

**Eksempel:**
Der er 50 rigtige tumorer i billederne, og modellen finder 35 korrekt.
```
Recall = 35 / 50 = 0.70
```

**Hvornår er høj Recall vigtigst?**
Når det er dyrt at *overse* noget — f.eks. på et hospital der ikke må overse tumorer, selv hvis det betyder nogle falske alarmer.

---

## Precision vs. Recall — trade-off

De to metrics trækker ofte i hver sin retning:

| Ekstrem | Resultat |
|---|---|
| Modellen gætter kun én gang og har ret | Precision = 1.0, Recall = meget lav |
| Modellen råber "objekt!" på absolut alt | Recall = 1.0, Precision = meget lav |

En god model har **begge** højt.

---

## IoU — Intersection over Union

Før vi kan forstå mAP50, skal vi forstå IoU.

Når YOLO tegner en bounding box, sammenlignes den med den "rigtige" boks (annoteret af et menneske). IoU måler **hvor meget de to bokse overlapper geometrisk**.

```
        ┌─────────────┐
        │  Rigtig boks│
        │    ┌────────┼───┐
        │    │ Overlap│   │
        └────┼─────────┘  │
             │  YOLO boks │
             └────────────┘

IoU = Overlap-areal / Samlet areal af begge bokse
```

- IoU = 1.0 → boksene er identiske
- IoU = 0.0 → boksene overlapper slet ikke

---

## mAP50

> *Gennemsnitlig Precision — men kun bounding boxes med mindst 50% overlap (IoU ≥ 0.5) tæller som korrekte.*

- **m** = mean (gennemsnit på tværs af klasser)
- **AP** = Average Precision
- **50** = IoU-grænse på 50%

**Hvorfor 50% og ikke 90%?**
En boks behøver ikke være pixel-perfekt for at være nyttig i praksis. 50% overlap betyder objektet er tydeligt fundet og lokaliseret. Kræver man 90%, straffes modellen for at være *lidt unøjagtig*, selv når den reelt har fundet objektet.

---

## Kravene i opgaven

| Metric | Krav |
|---|---|
| Precision (P) | > 0.5 |
| Recall (R) | > 0.3 |
| mAP50 | > 0.2 |

Disse krav er relativt lave — en model der finder mere end halvdelen af objekterne nogenlunde præcist vil typisk opfylde dem.

---

## Opsummering

| Metric | Spørgsmålet den besvarer |
|---|---|
| **Precision** | Af alle modellens gæt — hvor mange var rigtige? |
| **Recall** | Af alle rigtige objekter — hvor mange fandt modellen? |
| **mAP50** | Gennemsnitlig Precision, når bokse skal have 50%+ overlap for at tælle |