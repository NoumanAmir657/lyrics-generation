import pandas as pd

df = pd.read_csv("lyrics/lyrics.csv").sample(frac=1)
lyrics = df['lyrics'].unique()

with open("lyrics/lyrics.txt", 'w') as f:
    for lyric in lyrics:
        f.write(lyric)
        f.write('\n')