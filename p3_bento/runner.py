from sklearn import svm
from sklearn import datasets
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.adapters import DataframeInput
from bentoml.artifact import SklearnModelArtifact
# модель из коробки
from bentoml import (env,               #среда
                     artifacts,         # функция ETL
                     api,               # общение с API
                     BentoService,      # базовая модель
                     web_static_content # к контенту
                     )

# установка декораторов с параметрами, моделью и путями до файлов (html + оформление)
@env(auto_pip_dependencies=True)
@artifacts([SklearnModelArtifact('model')])     # модель
@web_static_content('./static')                 # путь
class Classifier(BentoService):

    # настраиваемся на api
    @api(input=DataframeInput(), batch=True)
    def test(self, df):
        # из api -> получаем df (группами)
        # в базовые ETL, который возвращается предикт
        return self.artifacts.model.predict(df)


if __name__ == "__main__":
    # стандартный процесс с ирисками )
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

    # инит класса с классификатором
    iris_classifier_service = Classifier()

    # пакуюем новую модель в artifacts(имя, модель)
    iris_classifier_service.pack('model', clf)

    # сохранение и запуск
    saved_path = iris_classifier_service.save()