# ASR project barebones

## Установка

Для старта необходимо скачатьи с диска [модель](https://drive.google.com/file/d/15ggn5IlBNqpP9XJQn0t9HcrrHfJrlVcS/view?usp=drive_link) и [конфиг](https://drive.google.com/file/d/1zouR5gI1fqcv7yj2d-Yxd7FkztHtQBjg/view?usp=drive_link) в `default_test_data/`.

Затем запустить re
```shell
pip install -r ./requirements.txt
```
## Запуск
Для тестирования нужно запустить следующую команду:
```shell
python test.py
```
Доступны параметры, означающие следующее:
-c -- указать свой config (по дефолту config.json)
-r -- указать свою модель (по дефолту model.pth)
-m -- указать на каком именно датасете планируется тестироваться (по дефолту test-other)
-d -- выбрать автоматически девайс для теста (по дефолту all)
-o -- куда планируется производиться вывод (по дефолту output.json)
-t -- путь к датасету (по дефолту в стандартной папке)
-b -- размер батча, на котором будет тестироваться (по дефолту 25)
-j -- количество джобов для выполнения (по дефолту 1)

## Конфиг
Чтобы вставить свой датасет в конфиг нужно проделать следующее:
В data вставить код:
```shell
    "{label_of_family_dataset}": {
      "batch_size": 32,
      "num_workers": 8,
      "datasets": [
        {
          "type": "{type_of_dataset}",
          "args": {
            "part": "{part_of_dataset_family}"
          }
        }
      ]
    }
```
label_of_family_dataset -- название семейста датасетов, которое стоит передавать в -m (проще назвать test-other)
type_of_dataset -- один из доступных типов датасетов для обучения модели.
part_of_dataset_family -- какую именно часть собираются выбирать из всего семейства датасетов.


По всем вопросам писать в tg @Secret_pirogok
