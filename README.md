# VK-Cup
The repository contains 3rd place solution of VK-Cup 2022 qualification task

## Порядок выполнения ноутбуков в репозитории:
- train_val_test_split.ipynb - разбиение исходного датасета в хронологическом порядка
- recommenders.ipynb - отбор кандидатов с помощбю различных методов
- merge_candidates.ipynb - скрипты для объединения кандидатов и признаков в один dataframe
- features.ipynb - построение признаков
- train_lgb.ipynb - обучение ранкера
- make_prediction_by_chanks.ipynb - финальное предсказание на тесте 
