# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Для части A выбран датасет **STL10**, потому что он хорошо подходит для сопоставимого сравнения простой CNN, аугментаций и transfer learning.
- Для части B выбран датасет **OxfordIIITPet** и трек **segmentation**, потому что здесь удобно поставить бинарную задачу `pet vs background`.
- В части A сравнивались эксперименты `C1-C4`: simple CNN без/с аугментациями и pretrained ResNet18 в режимах `head-only` и `partial fine-tuning`.
- Во второй части сравнивались два режима постобработки `V1-V2` для pretrained `DeepLabV3-ResNet50`, отличающиеся порогом бинаризации foreground.

## 2. Среда и воспроизводимость

- Python: рекомендуется 3.10+
- torch / torchvision: 2.11.0+cpu / 0.26.0+cpu
- Устройство (CPU/GPU): cpu
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.
- FAST_RUN: True (на CPU включается автоматически для более быстрого end-to-end запуска)

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: `STL10`
- Разделение: `train -> train/val = 80/20` с фиксированным seed, плюс официальный `test`
- Базовые transforms: `Resize(64,64) -> ToTensor()`
- Augmentation transforms: `Resize(72,72) -> RandomCrop(64,64) -> RandomHorizontalFlip -> ColorJitter -> ToTensor()`
- Комментарий (2-4 предложения): STL10 содержит 10 классов естественных изображений. Это удобный учебный датасет для сравнения обучения с нуля и transfer learning. Размер изображений выше, чем у CIFAR, поэтому простая CNN уже может извлекать полезные локальные признаки, но pretrained ResNet18 обычно даёт более сильный baseline.

### 3.2. Часть B: structured vision

- Датасет: `OxfordIIITPet`
- Трек: `segmentation`
- Что считается ground truth: бинарная маска `foreground = pet`, где в trimap foreground задаётся как `mask != background`
- Какие предсказания использовались: вероятности классов `cat` и `dog` из pretrained `DeepLabV3-ResNet50`, сведённые к общей `pet_prob`
- Комментарий (2-4 предложения): для OxfordIIITPet естественно оценивать именно область животного, а не полный multi-class segmentation benchmark. Предобученная модель из torchvision не обучалась специально на этом датасете, но даёт разумный zero-shot / transfer baseline. Это делает постановку воспроизводимой и хорошо подходящей для учебной демонстрации `mean IoU`.

## 4. Часть A: модели и обучение (C1-C4)

Опишите коротко и сопоставимо:

- C1 (simple-cnn-base): `SimpleCNN`, обучение с нуля, базовые transforms
- C2 (simple-cnn-aug): та же `SimpleCNN`, но с train-time аугментациями
- C3 (resnet18-head-only): `ResNet18` с pretrained weights, backbone frozen, обучается только `fc`
- C4 (resnet18-finetune): `ResNet18` с pretrained weights, разморожены `layer4 + fc`

Дополнительно:

- Loss: `CrossEntropyLoss`
- Optimizer(ы): `Adam` для SimpleCNN, `AdamW` для ResNet18
- Batch size: 16
- Epochs (макс): C1/C2=1, C3=1, C4=1
- Критерий выбора лучшей модели: `best_val_accuracy`

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран segmentation track

- Модель: `DeepLabV3-ResNet50` (pretrained)
- Что считается foreground: объединённый объект `pet` (кошка или собака) против фона
- V1: базовая постобработка — `pet_prob >= 0.40`
- V2: альтернативная постобработка — `pet_prob >= 0.60`
- Как считался mean IoU: для каждой картинки считалось пересечение и объединение бинарных масок `pred vs gt`, затем бралась средняя IoU по выборке
- Считались ли дополнительные pixel-level метрики: да, `pixel_precision` и `pixel_recall`

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/segmentation_examples.png`, `./artifacts/figures/segmentation_metrics.png`

Короткая сводка (6-10 строк):

- Лучший эксперимент части A: C4
- Лучшая `val_accuracy`: 0.9100
- Итоговая `test_accuracy` лучшего классификатора: 0.8900
- Что дали аугментации (C2 vs C1): изменение `best_val_accuracy` = -0.0300
- Что дал transfer learning (C3/C4 vs C1/C2): максимальный прирост относительно лучшего CNN = 0.6300
- Что оказалось лучше: partial fine-tuning
- Что показал режим V1 во второй части: `precision=0.9561`, `recall=0.8929`, `mean_iou=0.8464`
- Что показал режим V2 во второй части: `precision=0.9732`, `recall=0.8418`, `mean_iou=0.8058`
- Как интерпретируются метрики второй части: `mean IoU` показывает степень совпадения масок, `pixel_precision` штрафует лишние пиксели foreground, `pixel_recall` — пропуски объекта

## 7. Анализ

- Простая CNN служит базовой точкой отсчёта: она обучается с нуля и показывает, сколько качества можно получить без transfer learning.
- Сравнение C1 и C2 показывает влияние аугментаций при одинаковой архитектуре: меняются только train-transforms, поэтому разница интерпретируется достаточно чисто.
- На текущем запуске прирост C2 относительно C1 по best_val_accuracy составил -0.0300.
- Pretrained ResNet18 использует признаки, уже выученные на большом внешнем корпусе изображений, поэтому обычно стартует стабильнее и быстрее.
- В режиме head-only обновляется только последний классификатор, а в режиме partial fine-tuning дополнительно адаптируется layer4, что делает модель гибче.
- Лучший классификатор в этом запуске — C4 с best_val_accuracy=0.9100 и final test_accuracy=0.8900.
- Во второй части в качестве foreground берётся pet-mask, а качество оценивается через mean IoU как основную метрику пересечения предсказанной и истинной области.
- При переходе от V1 к V2 изменение pixel_precision составило 0.0170, а изменение pixel_recall — -0.0511.
- Повышение порога бинаризации обычно уменьшает ложные срабатывания, но может ухудшать полноту, поэтому precision и recall часто меняются в разные стороны.
- Визуально полезно смотреть не только на средние метрики, но и на типовые ошибки: потеря тонких границ, шум по краям и пропуски части объекта.

## 8. Итоговый вывод

- В качестве базового конфига классификации для дальнейшей работы я бы выбрал C4, потому что именно он дал лучший validation-результат в текущем прогоне.
- Главный вывод по transfer learning: pretrained backbone обычно даёт заметно более сильную стартовую точку, чем обучение с нуля на умеренном датасете.
- Главный вывод по segmentation и метрикам: accuracy по изображению целиком здесь недостаточна, а mean IoU, pixel_precision и pixel_recall дают куда более содержательную оценку качества маски.

## 9. Приложение (опционально)

- При желании можно расширить ноутбук:
  - добавить confusion matrix для лучшего классификатора;
  - сравнить ещё один режим fine-tuning;
  - попробовать третий порог бинаризации или другую постобработку для segmentation;
  - увеличить число эпох и отключить `FAST_RUN` перед финальной сдачей.
