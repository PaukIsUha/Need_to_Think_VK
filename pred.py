import pandas as pd

df = pd.read_csv("some_file");
df.columns

ranges_columns = []
for column in df:
    range_columns.append([df[column].min(), df[column].max()])
range_columns
# какие рэнжи брать?
# разбиение