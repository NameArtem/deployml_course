import numpy as np
import pandas as pd
import great_expectations as ge


def data_qual(df: pd.DataFrame):
    """
    Функция для тестирования данных в pipeline

    :param df:
    :return:
    """
    df = ge.from_pandas(df)

    # создаем проверки
    result = df.expect_column_values_to_be_in_set('Subscriber_Type',
                                                  list(df['Subscriber_Type'].unique()),
                                                  mostly=.95)

    if not result['success']:
        err = result["exception_info"]
        raise Exception(f"You get unexpected data in Subscriber_Type column\n{err}")