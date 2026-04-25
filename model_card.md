# 🎧 Model Card: Musiholical

## 1. Model Name

**Musiholical**

---

## 2. Intended Use

The original goal of this music recommender system is to rank songs inputted to the system by several attributes of a song (e.g. genre, mood, energy level, etc.), and produce a list of recommended songs based on the rank. Musiholical, the renewed version of the original system, provides smoother user experience in using the system to rank from the existing songs.

---

## 3. How the Model Works

The model works by following these steps:

1. Retrieve user input query
2. Send the query as a request to the LLM to convert it to estimated profiles
3. Convert existing songs and user profile into text descriptions
4. Embed these text descriptions so that songs that are semantically similar to the user profile get to ranked higher.
5. Present the user with two kinds of outputs:

   a. Classical outputs, which are the top recommending songs that are ranked with exact genre and mood match.

   b. RAG outputs, which are the top recommending songs hat are ranked based on semantic genre and mood similarity.

---

## 4. Data

- The dataset currently used in this system has 18 sample songs, which represents popular genres like pop and lofi, as well as other genres like rock, jazz, classical, etc.. 10 out of these songs were given in the template, and the other 8 songs that added later to test the robustness of the system.
- The dataset covers moods such as happy, chill, intense, relaxed, etc.. However, it does not cover other popular moods like dreamy, bittersweet, playful, motivated, and more.

---

## 5. Strengths

The user type that the system gives reasonable results is the profile where the pop genre and high energy value are high, since the majority of songs in the used dataset are songs with these attributes. The scoring algorithm used in calculating recommendation scores correctly captures the alignments of the genre, mood and energy between the song and the user. Cases where the recommendations matched this intuition are that users who like pop and have a high energy profile would be recommended with pop songs that are energetic, and users with a lower energy profile and prefers chill more would be recommended with lofi songs that are relaxed and peaceful.

---

## 6. Limitations and Bias

There is a bias in the model. If the song catalog used for the calculation is imbalanced (i.e. the catalog contains more songs for a specific metric), then the user who has a minor/niche preference might receive poor recommendations due to the underrepresentation of their preferred songs in the catalog.

---

## 7. Evaluation

### User query: Cyber

Output:

Parsing your preference with Ollama...

Extracted preference:
Query: 'Cyber'
Reasoning: "Cyber" interpreted as intense electronic — high-energy, electronic, melancholic tone.
Profile: {'genre': 'electronic', 'mood': 'intense', 'energy': 1.0, 'acousticness': 0.0, 'valence': 0.0}

User profile: {'genre': 'electronic', 'mood': 'intense', 'energy': 1.0, 'acousticness': 0.0, 'valence': 0.0}

_Classic mode:_

#1 Storm Runner by Voltline
Genre: rock | Mood: intense
Score: 0.70
Why: - Mood matches your favorite 'intense' (+0.35) - Genre 'rock' does not match your favorite 'electronic' (+0.00) - Energy 0.91 vs your target 1.00 (+0.18) - Valence 0.48 vs your target 0.00 (+0.05) - Tempo 152 BPM vs preferred 200 BPM (+0.06) - Acousticness 0.10 vs your target 0.00 (+0.05) - Danceability 0.66 vs preferred 1.00 (+0.01)

_RAG mode:_

#1 Fracture Point by Iron Veil
Genre: metal | Mood: angry
Score: 0.88
Why: - Semantic similarity (mood + genre) 0.88 (+0.48) - Energy 0.98 vs your target 1.00 (+0.20) - Valence 0.22 vs your target 0.00 (+0.08) - Tempo 168 BPM vs preferred 200 BPM (+0.06) - Acousticness 0.04 vs your target 0.00 (+0.05) - Danceability 0.52 vs preferred 1.00 (+0.01)

### User Query: I want some lovely vide songs

Output:

Parsing your preference with Ollama...

Extracted preference:
Query: 'I want some lovely vide songs.'
Reasoning: "I want some lovely vide songs." interpreted as happy pop — mid-energy, electronic, upbeat tone.
Profile: {'genre': 'pop', 'mood': 'happy', 'energy': 0.6, 'acousticness': 0.2, 'valence': 0.9}

User profile: {'genre': 'pop', 'mood': 'happy', 'energy': 0.6, 'acousticness': 0.2, 'valence': 0.9}

_Classic mode:_

#1 Sunrise City by Neon Echo
Genre: pop | Mood: happy
Score: 0.94
Why: - Mood matches your favorite 'happy' (+0.35) - Genre matches your favorite 'pop' (+0.20) - Energy 0.82 vs your target 0.60 (+0.16) - Valence 0.84 vs your target 0.90 (+0.09) - Tempo 118 BPM vs preferred 136 BPM (+0.07) - Acousticness 0.18 vs your target 0.20 (+0.05) - Danceability 0.79 vs preferred 0.60 (+0.02)

_RAG mode:_

#1 Rooftop Lights by Indigo Parade
Genre: indie pop | Mood: happy
Score: 0.89
Why: - Semantic similarity (mood + genre) 0.91 (+0.50) - Energy 0.76 vs your target 0.60 (+0.17) - Valence 0.81 vs your target 0.90 (+0.09) - Tempo 124 BPM vs preferred 136 BPM (+0.07) - Acousticness 0.35 vs your target 0.20 (+0.04) - Danceability 0.82 vs preferred 0.60 (+0.02)

---

## 8. Future Work

- For potential improvements, I believe that refining the prompts which will be sent to LLM along with user query will be a great starting point since for now the algorithm has a basic, working prompt which can request LLM and retrieve responses from it without problem. However, a more carefully designed prompts will greatly convey the LLM about the user query, which would generate better responses that match user preferences on songs.
- Another thing that is worth focusing is that user's tastes on songs should be accurately captured before using this system to rank and recommend best-suit songs. This means that the fields in `UserProfile` should be refined, tracked and updated based on the history of song listening while using the music application.

---

## 9. Personal Reflection

- The biggest learning moment during the development of this project was that I learned that the characteristics of existing songs and user profiles can be embedded for future matching of user queries to songs.
- The utilization of AI tools helped me break down some potential biases existed in the system. I needed to double-check the response provided by these tools when I noticed that the formatting (e.g. the return type of a function) from the response is different from the parameters required by other functions, for which this could cause the system to function incorrectly.
- Despite the simpleness of the request prompt for LLM, it did a decent job in conveying what the LLM should expect and what it should return for each request.
- If I have a chance to extend this project, I would try upgrading the project to analyze user profile (i.e. user tastes on music) based on inputted user history of music listening, since this would help the score calculation and the recommendation system better capture the user's preference, thus make the system more effective and robust.
