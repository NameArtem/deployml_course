import pandas as pd

import joblib
import config


def make_prediction(input_data):
    
    # реализовать pipeline, сделать предик
    # вернуть предикт

    return results
   
if __name__ == '__main__':
    
    # пример реализации
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = pd.read_csv(config.TRAINING_DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.2,
        random_state=0)
    
    pred = make_prediction(X_test)
    
    # показать качество модели
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()

