# TWM People Counting

Projekt realizuje zliczanie osób przekraczających wyznaczoną linię wejścia/wyjścia na podstawie nagrania wideo. System wykrywa osoby modelem YOLO, śledzi je między klatkami i osobno zlicza przejścia `IN` oraz `OUT`.

## Zakres

Główny plik programu:

```text
myObjectCounting.py
```

Pipeline:

1. Wczytanie nagrania przez OpenCV.
2. Detekcja osób modelem YOLO.
3. Filtrowanie klasy `person`.
4. Śledzenie osób z użyciem filtru Kalmana.
5. Dopasowanie detekcji do ścieżek przez IoU i algorytm węgierski.
6. Zliczanie przekroczeń linii `IN` / `OUT`.
7. Zapis nagrania wynikowego z bounding boxami, śladami ruchu i licznikami.

## Dane

W projekcie używane są dwa typy danych:

- własne nagrania domowe, odpowiadające scenariuszowi wejścia/wyjścia z pomieszczenia,
- publiczny zbiór Baidu People Counting, używany jako dodatkowy test generalizacji.

Nagrania głębi (`Depth`) nie są używane w aktualnym pipeline, ponieważ YOLO działa na obrazach RGB. Mogą być potraktowane jako rozszerzenie projektu.

## Instalacja

Utworzenie i aktywacja środowiska:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Uruchomienie

Przykład dla filmu 320x240:

```bash
python myObjectCounting.py \
  --source data/new_processed/2015_05_10_11_15_14FrontColor.mp4 \
  --output result.mp4 \
  --conf 0.35 \
  --imgsz 960 \
  --nms-iou 0.70 \
  --iou 0.30 \
  --max-age 12 \
  --min-hits 5 \
  --line 0 150 320 150 \
  --line-margin 8 \
  --count-cooldown 15 \
```

Po zakończeniu program wypisuje:

```text
Final counts: IN=..., OUT=...
```

## Najważniejsze parametry

| Parametr | Znaczenie |
| --- | --- |
| `--source` | ścieżka do nagrania wejściowego |
| `--output` | ścieżka do nagrania wynikowego |
| `--conf` | minimalna pewność detekcji YOLO |
| `--imgsz` | rozmiar wejścia modelu YOLO |
| `--nms-iou` | próg IoU dla NMS w YOLO |
| `--iou` | próg IoU dla dopasowania detekcji do tracków |
| `--max-age` | ile klatek utrzymać track bez detekcji |
| `--min-hits` | ile dopasowań potrzeba do potwierdzenia tracka |
| `--line` | linia zliczania w formacie `x1 y1 x2 y2` |
| `--line-margin` | martwa strefa wokół linii, ogranicza podwójne zliczenia |
| `--count-cooldown` | blokada ponownego zliczenia tego samego tracka |
| `--device` | `cpu` albo `cuda:0` |
| `--no-display` | praca bez okna OpenCV, tylko zapis wyniku |

## Ewaluacja

Wyniki testów są zbierane w:

```text
results_summary.md
```

Do każdego filmu należy ręcznie policzyć `GT_IN` i `GT_OUT`, a następnie porównać je z wynikiem programu.

Automatyczna ewaluacja na podstawie pliku `data/annotations/ground_truth.csv`:

```bash
python evaluate_counts.py \
  --model yolo26s.pt
```

Skrypt zapisuje szczegółowe wyniki do `evaluation_results.csv`, krótkie podsumowanie do `evaluation_summary.md` oraz nagrania wynikowe do katalogu `eval_outputs/`.

Metryka używana w raporcie:

```text
Error_IN = |GT_IN - PRED_IN|
Error_OUT = |GT_OUT - PRED_OUT|
Accuracy = 1 - (Error_IN + Error_OUT) / (GT_IN + GT_OUT)
```

## Ograniczenia

System działa gorzej w przypadku:

- przepalonego tła,
- okluzji,
- niskiej rozdzielczości,
- dużego tłoku przy linii,
- niestabilnych detekcji YOLO.

Położenie linii zliczania ma duży wpływ na wynik. Linię należy ustawić tam, gdzie osoby są widoczne możliwie stabilnie.
