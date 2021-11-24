import pandas as pd

data = [
    [1, 2, 3, 4, 5, 6, -9, -9, -9, -9],
    [1, 2, 3, 4, 5, 6, 7, -9, -9, -9],
    [1, 2, 3, 4, 5, 6, 7, 8, -9, -9],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, -9],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
]

columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(data, columns=columns)

df['sum'] = (df == -9).sum(axis=1)
df = df[df['sum'] < 2][columns]

print(df)
