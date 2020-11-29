import requests
import pandas as pd
import time

if __name__ == "__main__":

    while True:
        try:
            time.sleep(5)
            # обращаемся к сайту
            response = requests.post("http://127.0.0.1:6789/rf_model/predictor")

            # json -> DF
            df = pd.DataFrame().from_dict(response.json(), orient='index').T[[ 'row_index_r','predict_date_r', 'predict_r']]
            df.columns = ['row_index','predict_date', 'predict']

            print(df.head(1))
            # сохраняем результат
            df.to_csv('data_vault/answer.csv', mode='a', header=False, index=False)
        except KeyboardInterrupt:
            print('Receiving is done...')
            break

