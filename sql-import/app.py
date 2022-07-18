from os import environ
import pandas as pd
from sqlalchemy import create_engine

#db_uri = environ.get('[DB_FLAVOR]+[DB_PYTHON_LIBRARY]://[USERNAME]:[PASSWORD]@[DB_HOST]:[PORT]/[DB_NAME]')
#db_uri = environ.get('mysql+pymysql://myuser:mypassword@db.example.com:5432/mydatabase')
db_uri = environ.get('SQLALCHEMY_DATABASE_URI')
self.engine = create_engine(db_uri, echo=True)

#pass the engine object to pandas
table_df = pd.read_sql_table(
    table_name,
    con=engine
)
#
table_df = pd.read_sql_table(
    "table_name",
    con=engine,
    schema='public',
    index_col='job_id',
    coerce_float=True,
    columns=[#works like SQL select on x columns
        'column1',
        'column2',
        'column3',
        'column4',
        'column5'
    ],
    parse_dates=[
        'created_at',
        'updated_at'
    ],
    chunksize=500
)

#dataframe data cleaning
#...