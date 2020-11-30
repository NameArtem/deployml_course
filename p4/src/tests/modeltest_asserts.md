### Asserts для модели

- проверка на существование и использование нужного класс `assert lib.class.type == "XGBoost"`
- проверка на использование параметров `assert model.params != {}` или `assert model.params != param_dict`

----------------------------------

Группа проверки данных для модели `with pytest.raises(ValueError)`

- проверка на **nan**, **None**, **распределения по переменны** и тп
- **DataQuality** по данным
- assert **порядок колонок** в fit и prеdict модели
- assert `np.all(model.columns == train_x.columns)`
- **feature importances** не должен иметь 0 (как вариант)
- **corr** - корреляции между фичами должны быть ниже порога

