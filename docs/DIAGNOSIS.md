# Diagnose: Warum CraterGAN (hyperion) nie funktioniert hat

Stand: 2026-11-07. Statische Prüfung plus Smoke-Test auf Zufallsdaten. Keine Änderungen am Code.

## 1. Befund in Kürze

Die Implementierung ist ein MLP-GAN mit sieben Schichten und ohne Faltungen. Es gibt mehrere Fehler, die für sich schon Training verhindern: Wertebereich-Mismatch zwischen Generator und Daten, ein falsch übergebener BatchNorm-Parameter, eine Validierungsmetrik, die die Diskriminator-Stärke misst, und fehlendes `detach`. Dazu kommt, dass `CLAUDE.md` eine andere Architektur beschreibt als den Code.

## 2. Smoke-Test

Skript: `scratchpad/cnn-review/smoke/smoke_gan.py`, Umgebung `scratchpad/cnn-review/venv` (torch 2.15, Lightning 2.x). Lauf auf einer Kopie von `cratergan/` ohne `__main__.py`. Daten: gleichverteilte Zufallsbilder in [0,1], also keine Krater. Das Training folgt der Schrittlogik aus `gan.py:112-132`, aber mit eigenen Optimierern, da kein Trainer ohne Daten laufen kann.

| Test | Ergebnis |
|---|---|
| Parameter (64 px / 256 px) | G 4.9 M / 67.9 M, D 2.2 M / 33.7 M |
| Ausgabeform G (256 px) | (8, 1, 256, 256), korrekt |
| BatchNorm1d erster Block | `eps=0.8`, `momentum=0.1` (Parameter falsch belegt, siehe C4) |
| D-Schritt ohne detach | G bekommt Gradienten (sum\|grad_G\| = 3.8e1 bei 64 px) |
| 50 Schritte, 64 px | g_loss 0.665 → 0.594, d_loss 1.41 → 0.90, D(real) 0.94, D(fake) 0.55 |
| Ausgabebereich G nach 50 Schritten | min -0.08, max 0.20, Anteil < 0 nur 4.6 % |
| 5 Schritte, 256 px | D(real) = 1.000, D(fake) = 0.474, val_loss = 0.0000 |

Einschränkungen: Zufallsdaten, nur 50 bzw. 5 Schritte, kein Krater-Datensatz. Die Zahlen belegen Mechanik (Shapes, Gradientenfluss, BN-Parameter, Metrik-Verhalten), nicht Konvergenz.

## 3. Ranking der wahrscheinlichsten Ursachen

### C1. Wertebereich-Mismatch zwischen Generator und Daten (sehr wahrscheinlich)
- Reale Daten: `ToTensor()` liefert [0,1] (`cratergan/module/crater.py:19`).
- Generator endet mit `Tanh` → [-1,1] (`cratergan/module/generator.py:22`).
- Der Diskriminator sieht echte Bilder in [0,1] und gefälschte in [-1,1]. Er kann dann über das Vorzeichen trennen, statt über die Bildstruktur. Der Generator bekommt dadurch ein schwaches, unspezifisches Gradientensignal.
- Fix: Daten auf [-1,1] normieren (`Normalize([0.5],[0.5])`) oder Generator mit Sigmoid und Daten in [0,1] betreiben. Beides gleichzeitig nicht.

### C2. MLP statt Faltungsnetz (sehr wahrscheinlich)
- Eingabe 65 536 Dimensionen; `Linear(65536, 512)` im D (33.7 M Parameter), `Linear(1024, 65536)` im G (67 M). Keine Lokalität, keine Translationsäquivarianz, Pixel werden unabhängig voneinander behandelt.
- MLP-GANs auf Bildern sind bekannt für Mode Collapse und fehlende Struktur. Die Kraterform braucht lokale Filter.
- CLAUDE.md beschreibt ConvTranspose/Conv (`hyperion/CLAUDE.md:103-124`), im Code gibt es das nicht.
- Fix: DCGAN-Stil (G: ConvTranspose + BN + ReLU, D: Conv + LeakyReLU, ohne BN in D bei WGAN-GP). Modellgröße damit deutlich kleiner.

### C3. Validierungsmetrik misst die Diskriminator-Stärke (hoch)
- `gan.py:150-160`: `val_loss = BCE(D(real), 1)`. Das ist kein Qualitätsmaß für Generator-Ausgaben.
- `__main__.py:25-31`: ModelCheckpoint überwacht genau dieses `val_loss` (mode `min`). Der Checkpoint wird also gerade dann gewählt, wenn D gewonnen hat. Im Smoke-Test: val_loss = 0.0000 bei D(real) = 1.0.
- Fix: FID oder KID auf Generator-Samples, oder mindestens D-BCE auf Fakes plus Generator-Loss loggen. Checkpoint auf Generator-Metrik umstellen.

### C4. BatchNorm mit falschem Parameter (hoch, schleichend)
- `generator.py:12`: `nn.BatchNorm1d(out_feat, 0.8)`. Der zweite Positionsparameter ist `eps`, nicht `momentum`. Verifiziert: `eps=0.8`, `momentum=0.1`.
- Ein eps von 0.8 dämpft die Normalisierung stark (Varianz wird um 0.8 erhöht). Das verfälscht die Aktivierungsskalierung im G. Der Fehler stammt vermutlich aus einer TF-Vorlage, wo 0.8 der momentum war.
- Fix: `nn.BatchNorm1d(out_feat, momentum=0.8)` oder bewusst `eps=1e-5` mit Standardmomentum.

### C5. Fehlendes `detach()` und ungünstige Update-Reihenfolge (mittel)
- `gan.py:100`: `x_fake = self(z)` ohne `.detach()` im D-Schritt. Der D-Verlust backpropagiert durch G (Smoke-Test: Gradienten in G). Wegen `zero_grad` kein Fehler, aber doppelte Rechenarbeit und unsauber.
- `gan.py:121-132`: G wird zuerst aktualisiert, D danach. Üblich ist D zuerst (oder k D-Schritte pro G-Schritt).
- Fix: `x_fake = self(z).detach()`, Reihenfolge D → G, optional n_critic.

### C6. Sigmoid + `F.binary_cross_entropy` (mittel)
- `discriminator.py:14` Sigmoid, dann `F.binary_cross_entropy` (`gan.py:84,96,105,157`). Numerisch schlechter als `BCEWithLogitsLoss`, Sättigung an 0/1 bremst den Generator-Gradienten.
- Fix: D ohne Sigmoid, `BCEWithLogitsLoss`. Sigmoid nur für Logging.

### C7. Datenpipeline und Splits (mittel, Reproduzierbarkeit)
- `crater.py:41` `random_split` ohne Seed: Splits sind bei jedem Lauf anders.
- `crater.py:45`: `self.moon_crater` existiert nicht (AttributeError im Test-Stage). Richtig wäre `moon_crater`.
- `crater.py:46`: Test-Set ist eine zufällige 25 %-Teilmenge des gesamten Datensatzes und überschneidet sich mit Train/Val (Leakage).
- `gan.py:116`: `x, _ = batch[:-1]` ist fragil. Es funktioniert nur, weil der Batch aus `(image, mask, info)` besteht. Die Maske wird ignoriert (keine Konditionierung).
- Nicht prüfbar ohne Daten: ob "image" Reflektanz oder DEM ist und welcher Wertebereich vorliegt. Das muss vor dem Training geklärt werden.

### C8. Trainingsregime (mittel)
- `__main__.py:53,57`: `max_epochs=-1`, endlos, ohne Early Stopping oder Scheduler.
- D hat 33.7 M Parameter, G 67.9 M, bei 1:1 Schritten. Ohne Gradient-Penalty oder Label-Smoothing dominiert D schnell (siehe Smoke-Test: D(real) 1.0 nach 5 Schritten bei 256 px).
- `gan.py:57-71` `on_epoch_end`: legt bei Epoche 0 ein zweites Modell für `add_graph` an und loggt alle Histogramme jede Epoche. Kostet Zeit, verursacht keinen Fehler.

### C9. Veralteter Stack und Import-Seiteneffekt (niedrig für die Ursache, hoch für Wartbarkeit)
- `requirements.txt`: torch 1.10, Lightning 1.4.9, Python ≥3.9 (poetry). Nicht gegen diesen Stack getestet; Smoke-Test lief mit torch 2.15 und Lightning 2.x.
- `__main__.py:61` `sys.exit(fire.Fire(training))` ohne Guard: Import startet Training (Audit: 12.6-GB-Download ausgelöst).
- Fix: Guard, Stack pinnen und einmal gegen den Pin testen.

### C10. Kein Krater-Bezug in der Evaluation (Projektebene)
- Es gibt keine Kraterzählung, keine Form- oder Größenstatistik, keinen Abgleich mit Robbins. Ohne diese Metrik kann "funktioniert" nicht definiert werden.

## 4. Empfohlene Reihenfolge (nur Vorschlag)

1. C1 und C4 (Wertebereich, eps): sofort, klein.
2. C3 (Metrik und Checkpoint): Vor jedem neuen Lauf, sonst ist Erfolg nicht messbar.
3. C2 (Architektur auf DCGAN/WGAN-GP): größte Änderung, danach Lauf auf kleiner Datenmenge (64 px).
4. C5, C6 (detach, Reihenfolge, Logits).
5. C7 (Split, Seed, Testfehler, Maske).
6. C9 (Guard, Stack-Pin).

## 5. Zielbild (nur Gliederung)

- **Modell:** konvolutionelles GAN für Kraterpatches. Startpunkt DCGAN, Alternative WGAN-GP oder ein StyleGAN-light für bessere Auflösung.
- **Konditionierung:** Krater-Durchmesser (Klassen) und Altersklasse, als Embedding in G und D.
- **Downstream:** synthetische Bild-/DEM-Paare. Der Generator liefert Form, Hapke-Forward aus `src/sfs` rendert daraus Bilder. Diese Paare speisen DeepSfS (`io`) und Triton (3D-Kraterdetektion).
- **Daten:** DEM-Patches aus PTIFF über `luna-datasets` (`LunaPTIFFDataset`, `image`/`dem_m`-Layer). Labels und Positionen aus Robbins über `ldm-client` (`RobbinsCatalog`).
- **Auswertung:** FID/KID auf Krater-Patches; Downstream-Ablation (DeepSfS/Triton mit vs. ohne synthetische Daten).
- **Vorbedingung:** Hapke-Render-Hook existiert noch nicht (`src/sfs`, siehe AUDIT.md §4). Bis dahin ist das GAN nur Augmentierung ohne Physik.

## 6. Nicht geprüft

- Keine echten Daten, daher keine Aussage über Datenqualität und Konvergenz.
- Pinned Stack (torch 1.10, Lightning 1.4.9) nicht getestet.
- `craterdata` nicht gegen den Pin installiert, da `crater.py` nicht importiert wurde.
