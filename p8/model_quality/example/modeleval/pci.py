import pandas as pd
import numpy as np
import random

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):

    def psi(expected_array, actual_array, buckets):
        '''
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
           
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)

def dataset_validation(initial_df, new_df):

    initial = pd.DataFrame(initial_df)
    new = pd.DataFrame(new_df)

    if initial.shape[1] == new.shape[1]:
        df = pd.DataFrame()
        for col in initial.columns:
            try:
                col_initial = pd.to_numeric(initial[col], errors='raise')
                col_new = pd.to_numeric(new[col], errors='raise')

                if min(col_initial) == max(col_initial):
                    col_initial = random.choices([-0.000001, 0.000001], k=len(col_initial)) + col_initial

                if min(col_new) == max(col_new):
                    col_new = random.choices([-0.000001, 0.000001], k=len(col_new)) + col_new

                psi = calculate_psi(col_initial, col_new, buckettype='bins', buckets=10, axis=0)
                initial_desc_stat = '{} ({})'.format(round(np.mean(col_initial), 2),round(np.std(col_initial), 2))
                new_desc_stat = '{} ({})'.format(round(np.mean(col_new), 2),round(np.std(col_new), 2))

                if psi < 0.1:
                    conclusion = 'Very slight change'
                elif psi < 0.2:
                    conclusion = 'Some minor change'
                else:
                    conclusion = 'Significant change'

                df = df.append(pd.DataFrame(data={
                    'column name': col
                    ,'baseline mean (std dev)': initial_desc_stat
                    ,'new mean (std dev)': new_desc_stat
                    ,'stability index': round(psi, 2)
                    ,'conclusion': conclusion
                }, index = [0]))
            except ValueError: continue
        df = df.reset_index(drop = True)
    else:
        df = pd.DataFrame()
        print('The 2 csv do not have the same number of columns')
    return df

def dataset_validation_csv(initial_path, new_path):

    initial = pd.read_csv(initial_path)
    new = pd.read_csv(new_path)

    if initial.shape[1] == new.shape[1]:
        df = pd.DataFrame()
        for col in initial.columns:
            try:
                col_initial = pd.to_numeric(initial[col], errors='raise')
                col_new = pd.to_numeric(new[col], errors='raise')

                if min(col_initial) == max(col_initial):
                    col_initial = random.choices([-0.000001, 0.000001], k=len(col_initial)) + col_initial

                if min(col_new) == max(col_new):
                    col_new = random.choices([-0.000001, 0.000001], k=len(col_new)) + col_new

                psi = calculate_psi(col_initial, col_new, buckettype='bins', buckets=10, axis=0)
                initial_desc_stat = '{} ({})'.format(round(np.mean(col_initial), 2),round(np.std(col_initial), 2))
                new_desc_stat = '{} ({})'.format(round(np.mean(col_new), 2),round(np.std(col_new), 2))

                if psi < 0.1:
                    conclusion = 'Very slight change'
                elif psi < 0.2:
                    conclusion = 'Some minor change'
                else:
                    conclusion = 'Significant change'

                df = df.append(pd.DataFrame(data={
                    'column name': col
                    ,'baseline mean (std dev)': initial_desc_stat
                    ,'new mean (std dev)': new_desc_stat
                    ,'stability index': round(psi, 2)
                    ,'conclusion': conclusion
                }, index = [0]))
            except ValueError: continue
        df = df.reset_index(drop = True)
    else:
        df = pd.DataFrame()
        print('The 2 csv do not have the same number of columns')
    return df