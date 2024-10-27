## Temat projektu

##### Program do analizy meczów piłki nożnej

Projekt polega na stworzeniu programu służącemu analizie meczów piłki nożnej na podstawie nagrań wideo. Umożliwi on użytkownikowi wczytanie nagrania meczu i wygeneruje analizę, która obejmuje:
- Oznaczenie zawodników i bramkarzy z rozróżnieniem na drużyny oraz sędziów na nagraniu,
- Wyświetlanie prędkości poszczególnych piłkarzy,
- Wyświetlanie statystyki posiadania piłki przez drużyny,
- Wyświetlanie radaru boiska z zaznaczonymi pozycjami piłkarzy.

## Autorzy projektu

- Dominika Boguszewska
- Daniel Machniak
- Natalia Pieczko

  
## Harmonogram projektu
  

**Tydzień I** - 28.10-3.11

- Analiza literatury potrzebnej do realizacji projektu
- Utworzenie przestrzeni w Confluence, aby przechowywać źródła wiedzy, dokumentację i notatki ze spotkań
- Zaplanowanie struktury projektu z wykorzystaniem *cookiecutter*
- Skonfigurowanie repozytorium na platformie Gitlab
- Skonfigurowanie issue tracker'a Jira

**Tydzień II** - 4.11-10.11

- Przygotowanie zbioru danych trenujących i testowych
- Wybór rozmiaru modelu YOLO11 na podstawie eksperymentów wydajności i efektywności
- Fine-tuning modelu YOLO11 do śledzenia piłkarzy i piłki

**Tydzień III** - 11.11-17.11

- Implementacja modelu do rozróżniania zawodników pomiędzy drużynami
- Implementacja/Fine-tuning modelu YOLO11-pose do rozpoznawania punktów kluczowych na boisku
- Kolejne eksperymenty i testowanie modeli

**Tydzień IV** - 18.11-24.11

- Implementacja estymacji kamery między klatkami i transformacji perspektywy
- Implementacja logiki do statystyk posiadania piłki
- Sprawdzanie dokładności i testowanie wyników

**Tydzień V** - 25.11-01.12

- Obliczenie położenia zawodników względem boiska (korzystając z perspektywy kamery)
- Obliczenia dystansu przebytego przez zawodnika (potrzebny od obliczenia szybkości)
- Obliczenia szybkości piłkarzy
- Eksperymenty dokładności

**Tydzień VI** - 02.12-08.12

- Implementacja radaru boiska
- Optymalizacja poszczególnych funkcjonalności systemów
- Refactoring kodu

**Tydzień VII** - 09.12-15.12

- Integracja funkcjonalności
- Graficzne dopracowanie formy reprezentacji zawodników

**Tydzień VIII** - 16.12-22.12

- Stworzenie UI z informacjami o przebiegu meczu
- Czas na refinement kodu

**Tydzień IX** - 23.12-29.12 - Przerwa Świąteczna

**Tydzień X** - 30.12-05.01

- Testy końcowe na nagraniach
- Optymalizacja wydajności

**Tydzień XI** - 06.01-12.01

- Opracowanie końcowej dokumentacji projektu

## Bibliografia

- Dokumentacja biblioteki ultralytics [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- Dokumentacja modelu YOLO11 [https://docs.ultralytics.com/tasks/detect/](https://docs.ultralytics.com/tasks/detect/)
- Dokumentacja modelu YOLO11-pose [https://docs.ultralytics.com/tasks/pose/](https://docs.ultralytics.com/tasks/pose/)
- Blog na temat śledzenia piłki w sporcie [https://blog.roboflow.com/tracking-ball-sports-computer-vision/](https://blog.roboflow.com/tracking-ball-sports-computer-vision/)
- Blog na temat kalibracji kamery [https://blog.roboflow.com/camera-calibration-sports-computer-vision/](https://blog.roboflow.com/camera-calibration-sports-computer-vision/)
- Zbiór danych dla YOLO11 do śledzenia zawodników i piłki [https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- Zbiór danych do rozpoznawania punktów kluczowych na boisku [https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)
- Artykuł na temat śledzenia wielu obiektów jednocześnie [https://www.datature.io/blog/introduction-to-bytetrack-multi-object-tracking-by-associating-every-detection-box](https://www.datature.io/blog/introduction-to-bytetrack-multi-object-tracking-by-associating-every-detection-box)
- Model SigLIP [https://huggingface.co/docs/transformers/model_doc/siglip](https://huggingface.co/docs/transformers/model_doc/siglip)
- UMAP [https://github.com/lmcinnes/umap](https://github.com/lmcinnes/umap)
- KMeans [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- Zbiór filmików do testowania programu [https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv/data](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv/data)

## Planowany zakres eksperymentów

- Znalezienie optymalnego rozmiaru modeli YOLO11
- Poprawność śledzenia zawodników i piłki przez model YOLO11
- Kalibracja boiska - poprawność rozpoznawania punktów kluczowych boiska przez model
- Pomiar dystansu i szybkości
- Posiadanie piłki - testy manualne
- Rozróżnianie zawodników pomiędzy drużynami - testy manualne

## Planowana funkcjonalność programu

- Rozpoznawanie i śledzenie zawodników, sędziów, bramkarzy i piłki
- Rozróżnianie zawodników i bramkarzy pomiędzy drużynami
- Obliczanie posiadania piłki przez drużynę
- Transformacja perspektywy na płaszczyznę
- Pomiar prędkości zawodników
- Prezentacja graficzna posiadania piłki
- Prezentacja graficzna radaru boiska z zawodnikami

## Planowany stack technologiczny

- Autoformatter - black, linter - flake8
- Środowisko wirtualne - venv
- make / argparse
- cookiecutter
- Python 3.12
- Tensorboard
- PyTorch
- YOLO, SigLIP, UMAP, KMeans
- Google Colab
- Jira, Confluence
- Gitlab