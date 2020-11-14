## Tests for DS

### Python необходимости

- pytest
- понимание декораторов | [описание](https://www.geeksforgeeks.org/decorators-in-python/)
- понимание исключений | [описание](https://realpython.com/python-exceptions/)
- понимание контекстных менеджеров | [описание](https://book.pythontips.com/en/latest/context_managers.html)

Доп. по pytest:

- [getting-started-with-pytest](https://jacobian.org/2016/nov/27/getting-started-with-pytest/)
- [python-mocking](https://www.fugue.co/blog/2016-02-11-python-mocking-101)
- [python-mocking2](https://alexmarandon.com/articles/python_mock_gotchas/)

### Тесты необходимые для DS|DE проектов

1
Smoke test - тесты для опредления метода
```python
import pytest
from sklearn.linear_model import LinearRegression
def test_smoke():
    try:
        assert LinearRegression() is not None
    except NameError:
        pytest.fail("Model does not exist")
```

2
Проверка коннекторов к БД (если нужны для проекта)

**[Ссылка на Хабр, с большим объяснением](https://habr.com/ru/post/141209/)**

```python
import mock

def get_connect():
    engine = sqlalchemy.create_engine('connection_string')
    
    return df.read_sql("select * from *", con = engine)

@mock.patch('pytest_ex.text_connection.sqlalchemy.create_engine')
def test_connection(engine_mock, df):
    new_df = get_connect(engine_mock)
    
    pandas.testing.assert_frame_equal(new_df, df)


# или тестировать функции чтения
def df_from_csv(filename):
    return pd.read_csv(filename)

@mock.patch.object(pd, 'read_csv')
def test_df_from_csv(read_csv_mock, df):
    read_csv_mock.return_value = df
    actual = df_from_csv('file_name.csv')
    # ожидаемый результат
    expected = df
    # assertions
    pd.testing.assert_frame_equal(actual, expected)
```

3
Тесты на равенство / соответствие результатов после трансформации объекта

- Использовать pandas testing:

    -- pandas.testing.assert_frame_equal (сравнение DataFrame)
    
    -- pandas.testing.assert_series_equal  (сравнение колонок)
    
    -- pandas.testing.assert_index_equal  (сравнение строк по индексам)
    
    -- pandas.testing.assert_extension_array_equal  (сравнение любых массивов numpy)
    
-- Использование assert + numpy (методы сравнения объектов):
   
    -- np.allclose
    
    -- np.isclose
    
    -- np.any
    
    -- np.equal
    
-- Использование numpy testing методов:

    -- (Ссылка на документацию numpy asserts)[https://numpy.org/doc/stable/reference/routines.testing.html]

4
Тестирование существования файлов
```python
def df_from_csv(filename):
    """читаем все DataFrame в формате csv"""
    return pd.read_csv(filename)

def test_df_from_csv(filename):
    assert df_from_csv(filename) is not FileNotFoundError
```

5
Тестирование API
```python
import responses

@responses.activate
def test_api_404():
    responses.add(
        responses.GET,
        'https://your_path',
        json='ex_json',
        status=404,
    )


@responses.activate
def test_api_200():
    responses.add(
        responses.GET,
        'https://your_path',
        json={'ex_json'},
        status=200,
    )
```

6
Генерируйте данные для тестов на основе [hypothesis](https://hypothesis.readthedocs.io)

```python
import pandas as pd
from hypothesis import given
from hypothesis import strategies
from hypothesis.extra.pandas import column, data_frames
import builder

@given(
    # создавайте Pandas DataFrame для тестов
    data_frames(
    columns=[
        column(
            name='prog_start',
            elements=strategies.datetimes(
                min_value=pd.Timestamp(2020, 1, 1),
                max_value=pd.Timestamp(2020, 1, 10)
            )
        , unique=True),
        column(
            name='code', 
            elements=strategies.just(float('nan'))
        )
    ])
)
def test_fix_new_boxes_nan_replaced(raw_prog):
    prog = builder.fix_new_boxes(raw_prog)
    assert (prog.mat_code == builder.NO_MAT_CODE).all()
    assert prog.shape == raw_prog.shape
```

### Представлены тесты

1. Smoke test - тест на существование и запуск метода / модели
2. Fit test - тест на результат модели и её обучения (повторно обучается, результат предикта одинаковый)
3. Fix test - тест на использование параметров и проверку результатов
4. Param test - тест на проверку спиcка параметров и определение нужных
5. ```pytest --cov=rtanalysis``` - отчет по функциям тестирования (_test coverage_)

### Книга для ознакомления с "Искусством тестирования"
[ссылка на книгу pdf](http://barbie.uta.edu/~mehra/Book1_The%20Art%20of%20Software%20Testing.pdf)

