import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd

# read
def read_data():
    rw = pd.read_csv('data_vault/test_x.csv').sample(1)
    return rw

def get_index_val(sent):
    ind = int(pd.read_csv('data_vault/sent.csv')['index'].max()+1) if type(pd.read_csv('data_vault/sent.csv')['index'].max()) != float \
                                                                  else 0
    sent.insert(loc=0, column='index', value=ind)
    sent.to_csv('data_vault/sent.csv', mode='a', header=False, index = False)

    return dict(
                index=int(sent['index'].values),
                data=sent.drop('index', axis=1).to_dict(orient='records')[0]
              )


app = FastAPI()
@app.get("/data")
async def index():
    df = read_data()
    return get_index_val(df)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9876)