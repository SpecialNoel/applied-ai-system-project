# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeRanker 2.0**

---

## 2. Intended Use

The recommender system is designed to rank songs inputted to the system by several attributes of a song (e.g. genre, mood, energy level, etc.), and produce a list of recommended songs based on the rank. The system is for users who already have some records in listening songs with the music application since the system works best with sufficient user profile (e.g. what genre the user likes the best, what mood the user enjoys for the most of time, etc.). The system can be improved for real users after updating calculations on several song metrics, but for now it is more like a classroom exploration, meaning that it might not be practical to be put into real life applications immediately. In other words, the system is designed for a quick and convenient way to rank songs based on metrics involved in the calculation algorithm used in this system, and it is not intended to be used in real life production yet until the calculation algorithm is refined to cover more metrics with higher accuracy on reflecting the weights of these metrics.

---

## 3. How the Model Works

From my understanding, the real-world recommendation system resolves around a large amount of user data (user preferences to songs/videos based on their attributes, such as mood, genre, energy, etc). The system then constructs a value matrix that uses these user data, which can then be utilized to compare the overlapping between the preference of a target user to preferences of all other users (or, only select a range of more relevant users to provide better accuracy).

The design of this music recommendation system follows the same track, that it evaluates the music preferences of users to filter and finally recommend the most relevant songs to the target user. The system will take some most valuable features of a song, such as mood, genre, and energy combined with acousticness (ordered from the most important one to the least), to provide a relevance score for each available song, as well as constructing a list of songs that the target user will most likely be interested in based on their relevance scores.

The `Song` used in this system will contain basic information such as the unique id, title, artist, as well as the metrics that would be used to calculate the recommendation score for the song, such as genre, mood, energy, valence, and more.

The `UserProfile` of the system will store information including but not limited to a user's favorite genre, the most suitable mood, the amount of energy the user might have while using the system, and the likeness towards acoustical music.

The score calculation of the system is the following:

- Give +0.35 if the mood of the song matches the user profile
- Give +0.2 for genre
- Give +(0.2 \* difference in energy value)
- Give +(0.1 \* difference in valence value)
- Give +((0.08 \* (value of tempo (i.e. bpm)-40))/160)
- Give +(0.05 \* difference in acousticness value)
- Give +0.02 for danceability

---

## 4. Data

- The dataset currently used in this system has 18 sample songs, which represents popular genres like pop and lofi, as well as other genres like rock, jazz, classical, etc.. 10 out of these songs were given in the template, and the other 8 songs that added later to test the robustness of the system.
- The dataset covers moods such as happy, chill, intense, relaxed, etc.. However, it does not cover other popular moods like dreamy, bittersweet, playful, motivated, and more.

---

## 5. Strengths

The user type that the system gives reasonable results is the profile where the pop genre and high energy value are high, since the majority of songs in the used dataset are songs with these attributes. The scoring algorithm used in calculating recommendation scores correctly captures the alignments of the genre, mood and energy between the song and the user. Cases where the recommendations matched this intuition are that users who like pop and have a high energy profile would be recommended with pop songs that are energetic, and users with a lower energy profile and prefers chill more would be recommended with lofi songs that are relaxed and peaceful.

---

## 6. Limitations and Bias

Note that it should be acknowledged beforehand that the system contains several biases. For example, songs with higher values in tempo and danceability might be prioritized since the `UserProfile` of the system does not depend on the user's preference on these values, meaning that the higher values of these metrics, the higher score the song will be given when calculating their recommendation scores. Furthermore, if the song catalog used for the calculation is imbalanced (i.e. the catalog contains more songs for a specific metric), then the user who has a minor/niche preference might receive poor recommendations due to the underrepresentation of their preferred songs in the catalog.

---

## 7. Evaluation

### User profile setting: "genre": "pop", "mood": "energetic", "energy": 0.9, "acousticness": 0.1, "valence": 0.2:

- A user who prefers high energy songs is prioritized to be recommended with songs with high energy value and that songs that have energetic or happy moods.

### User profile setting: "genre": "lofi", "mood": "chill", "energy": 0.3, "acousticness": 0.1, "valence": 0.2:

- A user who prefers chill songs is prioritized to be recommended with songs with low energy value and that songs that have focus or sad moods.

### User profile setting: "genre": "pop", "mood": "intense", "energy": 0.8, "acousticness": 0.1, "valence": 0.2:

- A user who prefers intense songs is prioritized to be recommended with pop or rock songs and that songs that have high energy value.

---

## 8. Future Work

- For potential improvements, I believe that refining the calculation algorithm used in this system will be a great starting point since for now the algorithm has a very limited coverage on attributes of a song, as well as a not very thorough reflection on the weights of the existing attributes. To enhance the system, one should take more metrics of a song into consideration.
- Another thing that is worth focusing is that user's tastes on songs should be accurately captured before using this system to rank and recommend best-suit songs. This means that the fields in `UserProfile` should be refined, tracked and updated based on the history of song listening while using the music application.

---

## 9. Personal Reflection

- The biggest learning moment during the development of this project was that when calculating the score used to rank the song, it is preferred to normalize the score to be in range of [0, 1] for a better organization and easier comparison between scores.
- The utilization of AI tools helped me break down some potential biases existed in the system. I needed to double-check the response provided by these tools when I noticed that the formatting (e.g. the return type of a function) from the response is different from the parameters required by other functions, for which this could cause the system to function incorrectly.
- Despite the simpleness of the score calculating algorithm, it did a decent job in calculating a score of the song based on the considered attributes of the song and the target user's tastes in music.
- If I have a chance to extend this project, I would try upgrading the project to analyze user profile (i.e. user tastes on music) based on inputted user history of music listening, since this would help the score calculation and the recommendation system better capture the user's preference, thus make the system more effective and robust.
