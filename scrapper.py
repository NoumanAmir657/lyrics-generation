from azapi import AZlyrics

api = AZlyrics()

api.artist = "Taylor Swift"
songs = api.getSongs()


with open("lyrics.txt", 'w') as f:
    for song in songs:
        try:
            f.write(api.getLyrics(url=songs[song]["url"]))
        except:
            continue