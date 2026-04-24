# Natural Language Preference Learning (with Ollama)

This feature allows users to input music preferences using natural language instead of manually setting numeric values.

**Best part: It's completely FREE and runs locally on your machine!**

## How It Works

The `preference_parser.py` module uses **Ollama** (free, open-source LLM) to convert freeform user queries into structured music preferences:

- **Input:** "songs that make me feel like I'm in a 90s indie film"
- **Output:** `{"genre": "rock", "mood": "sad", "energy": 0.6, "acousticness": 0.7, "valence": 0.3}`

## Setup

### 1. Install Ollama

Download and install from: https://ollama.ai

Choose the version for your OS (Mac, Linux, Windows).

### 2. Start Ollama

```bash
ollama serve
```

Keep this terminal running in the background.

### 3. Download a Model

In a new terminal, download a lightweight model:

```bash
ollama pull mistral
```

This downloads ~4GB the first time (takes ~5-10 min depending on internet speed), then runs instantly after.

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Option A: Demo Script (Recommended First Try)

```bash
python3 demo_natural_language.py
```

This runs three example queries and shows recommendations for each:
- "songs that make me feel like I'm in a 90s indie film"
- "background music for a late night study session"
- "motivation for a tough workout"

### Option B: Interactive CLI

```bash
python3 -m src.main
```

When prompted, choose option 1 to enter a natural language query, then describe what kind of music you want.

## How Extraction Works

Ollama (running locally on your machine) analyzes your query and extracts:

| Attribute | Range | Meaning |
|-----------|-------|---------|
| `genre` | One of: pop, rock, lofi, jazz, electronic, acoustic | Music category |
| `mood` | One of: happy, chill, energetic, intense, sad, focused | Emotional tone |
| `energy` | 0.0–1.0 | Calm (0) to high-energy (1) |
| `acousticness` | 0.0–1.0 | Electronic (0) to acoustic (1) |
| `valence` | 0.0–1.0 | Sad/serious (0) to happy/positive (1) |

## Example Queries

Try these to see how the system extracts preferences:

- "I need calming music to focus on work" → chill, focused, low energy
- "Get me pumped for the gym" → high energy, pop/electronic, uplifting
- "Late night melancholy vibes" → sad/intense, low energy, acoustic
- "90s alternative rock nostalgia" → rock, intense, medium-high energy
- "Lo-fi hip hop for studying" → lofi, focused, low energy, low acousticness

## Architecture

```
User Query (natural language)
         ↓
    [Ollama API] (running locally)
         ↓
JSON: {"genre": "...", "mood": "...", "energy": ...}
         ↓
    [Validation & Clamping]
         ↓
Structured User Profile Dict
         ↓
[Recommender] → Top-K Songs
```

## Files

- `src/preference_parser.py` - Core parsing module with two functions:
  - `extract_with_reasoning()` - Returns both preferences and Ollama's reasoning
  - `parse_natural_language_preference()` - Returns only the preferences dict
  - `check_ollama_running()` - Checks if Ollama is available
- `demo_natural_language.py` - Standalone demo script
- `src/main.py` - Updated CLI that conditionally enables this feature

## Troubleshooting

**Error: "Ollama is not running"**
```bash
# In a separate terminal, run:
ollama serve

# Then try again in your original terminal
```

**Error: "Model not found"**
```bash
ollama pull mistral
```

**Error: "Connection refused"**
- Make sure `ollama serve` is running in another terminal
- Check that Ollama is installed: `ollama --version`

**Slow responses?**
- First run of a query can take 10-30 seconds
- Subsequent queries are faster
- If very slow, you may need more system RAM

## Why Ollama?

✅ **Completely Free** - No API keys, no credits needed  
✅ **Runs Locally** - Your data stays on your machine  
✅ **Fast** - After first run, processes queries in seconds  
✅ **Offline** - Works without internet after model is downloaded  
✅ **Open Source** - Transparent and community-driven  

## Future Improvements

- [ ] Learn from user acceptance/rejection → dynamically adjust extraction prompts
- [ ] Build user taste history → recognize evolving preferences
- [ ] Support playlist generation with narrative arc
- [ ] Multi-user "group vibe" queries
- [ ] Try other models for better extraction (e.g., neural-chat, orca)

