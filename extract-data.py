import pandas as pd
from sqlite3 import connect
import os

conn = connect("./DB_Auto_PMV_Generator.sqlite")

df_data = pd.read_sql('SELECT * FROM classifier_data WHERE classifier_id >= 10665', conn)
print("read data")
df_key = pd.read_sql('SELECT * FROM classifier_keys WHERE classifier_id >= 10665', conn)
print("read keys")

os.remove("./Classified_Videos.sqlite")
connout = connect("./Classified_Videos.sqlite")

df_data.to_sql('Data', connout)
print("save data")
df_key.to_sql('Keys', connout)
print("save keys")