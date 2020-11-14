import preprocessing_functions as pf
import config

# =========== scoring pipeline =========

# собираем все функции в одну, как большой pipeline
def predict(data):
    
    # загрузка данных
    

    # функции обработки данных
    
    # обучение модели
    
    # предикт


    
    return predictions

# ======================================
    
if __name__ == '__main__':
        
    from sklearn.metrics import accuracy_score    
    import warnings
    warnings.simplefilter(action='ignore')

    data = pf.load_data(config.PATH_TO_DATASET)
    
    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config.TARGET)
    
    pred = predict(X_test)

    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
        