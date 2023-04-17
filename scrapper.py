from azapi import AZlyrics

api = AZlyrics()

api.artist = input("Enter artist name: ")
songs = api.getSongs()

print(songs)

with open("lyrics/lyrics.txt", 'w') as f:
    for song in songs:
        try:
            f.write(api.getLyrics(url=songs[song]["url"]))
        except:
            continue