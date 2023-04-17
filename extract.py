import pandas as pd

df = pd.read_csv("sg.csv")

with open("lyrics/sg_lyrics.txt", 'w') as f:
    for l in df['Lyrics']:
        f.write(l)
        f.write('\n')
    