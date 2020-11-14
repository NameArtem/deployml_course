## Pytest

**!Note**

>Не добавляйте __init__.py в тест, иначе вы поймаете _No modelu named_
>https://stackoverflow.com/questions/41748464/pytest-cannot-import-module-while-python-can

### Использование

1. Run
```python
pytest /path_to_test_script.py
```

2. Run + html отчет
```python
pytest /path_to_test_script.py --html=report.html --self-contained-html
```

3. Run + code coverage report (результат по каждой функции)
```python
pytest /path_to_test_script.py --cov=src --verbose
```

4. Run + multi CPU
```python
pytest /path_to_test_script.py -n 5
```