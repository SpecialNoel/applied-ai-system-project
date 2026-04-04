flowchart TD
A([User Profile\nfavorite_genre, favorite_mood\ntarget_energy, target_valence, target_acousticness]) --> B

    B[Load songs from songs.csv] --> C

    C{For each song in catalog}

    C --> D[Score: +0.35 if mood matches]
    C --> E[Score: +0.20 if genre matches]
    C --> F[Score: +0.20 × energy similarity]
    C --> G[Score: +0.10 × valence similarity]
    C --> H[Score: +0.08 × normalized tempo]
    C --> I[Score: +0.05 × acousticness similarity]
    C --> J[Score: +0.02 × danceability]

    D & E & F & G & H & I & J --> K[Sum = final score for this song]

    K --> L{More songs?}
    L -- Yes --> C
    L -- No --> M[Sort all songs by score descending]

    M --> N([Top K Recommendations])
