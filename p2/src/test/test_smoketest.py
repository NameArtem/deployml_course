# up to parent
import sys
sys.path.append("../..")

from src.main.preprocessors import RFModel

def test_smoke():
    rfm = RFModel()

    assert rfm is not None, "Модели не существует"

