# DVC

Особенности работы с DVC
- **Отдельные скрипты** для описания каждого шага в pipeline
- **Отдельно храним параметры и метрики** в формате `yaml`
- Зависимости не должны пересекаться с создаваемыми фалами в одном шаге
- Файлы создаваемые во время шага **будут удалены во время его инициализации** (т.е. если файл с пометкой -о уже существует, он будет удален и пересоздан)
- Процессы зависимы и линейны
- Работает совместно с Git

### DVC процесс

1 Первые настройки
```
dvc init 

# файл сторадже
dvc remote add -d localremote /data/dvc-storage

# настройка конфига 
dvc config core.analytics false
```
-------------------

2 Создаем params.yaml

```
# file params.yaml
название модели (эксперимента):
    параметры для эксперимента
```

-------------------

3 Создаем шаги эксперимента. Не через файл main.py (иначе у нас будет все в одном, а отдельно)

```bash
# так как платформа Windows, то у меня ^ для переноса.
# \ - на Linux

# ETL stage
dvc run -n etl  ^
-d config/config.py ^
-d config/logging_conf.py ^
-d data/data/_station_data.csv ^
-d data/data/trip_data.csv ^
-d data/data/weather_data.csv ^
-d src/main/preprocessors.py ^
-d src/main/simple_etl.py ^
-o data/clean_data/cleandata0.1.0.csv ^
python src/main/simple_etl.py

# TRAIN stage
dvc run -n train ^
-d config/config.py ^
-d config/logging_conf.py ^
-d src/main/preprocessors.py ^
-d src/main/train_pipeline.py ^
-d data/clean_data/cleandata0.1.0.csv ^
-o data/split_data/xtest0.1.0.csv ^
-o data/split_data/xtrain0.1.0.csv ^
-o data/split_data/ytest0.1.0.csv ^
-o data/split_data/ytrain0.1.0.csv ^
-o data/model/rf_model0.1.0 ^
python src/main/train_pipeline.py

# PREDICT stage
dvc run -n predict ^
-d data/model/rf_model0.1.0 ^
-d config/config.py ^
-d config/logging_conf.py ^
-d src/main/quality.py ^
-d src/main/preprocessors.py ^
-d src/main/predict.py ^
-d data/split_data/xtest0.1.0.csv ^
-d data/split_data/xtrain0.1.0.csv ^
-d data/split_data/ytest0.1.0.csv ^
-d data/split_data/ytrain0.1.0.csv ^
-M data/metrics/rf_model0.1.0 ^
python src/main/predict.py
```
-------------

4 Изменяйте параметры и эксперементируйте в файле params.yaml

```
# file params.yaml
rf_model:
    n_estimators: 32

lasso_model:
    alpha: 0.9
```
---------------------

5  Репроизводство процесса

```
 dvc repro
```
---------------------------

6 Сравнение метрик
```
dvc params diff

dvc metrics diff
```
-----------------------------

7 Визуализаций метрик
```
dvc plots show -y precision -x recall имя_файла_с_метриками.json

dvc plots diff --targets имя_файла_с_метриками.json -y precision
```
