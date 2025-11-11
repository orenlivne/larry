# Larry Christiansen Simul Prep
- Fine-tune Maia to create a LarryBot.
- Targetted opening study.

Got it — I’ll provide the **full GitHub README in clean Markdown**, with proper fenced code blocks and no extra escaping. You can copy it directly into `README.md` and it will render perfectly on GitHub.

---

# 🤖 LarryBot: Maia Fine-Tuned Chess Engine

**LarryBot** is a fine-tuned Maia neural chess engine trained on ~2,000 of Larry Christian’s games.
It simulates Larry’s playing style and can play online through **Lichess Bot API**.

Built for macOS M1/M2/M3 with full GPU acceleration using **TensorFlow Metal** and **uv**.

---

## ⚙️ Requirements

* macOS 13+ (M1/M2/M3 recommended)
* [Homebrew](https://brew.sh/)
* [uv](https://github.com/astral-sh/uv)
* Lichess account with BOT token
  (Create at: [lichess.org/account/oauth/token](https://lichess.org/account/oauth/token))

---

## 🚀 Quick Start

### 1. Clone this repo

```bash
git clone https://github.com/orenlivne/larry.git
cd larry
```

### 2. Add Larry’s PGNs

Put all Larry Christian PGNs in:

```
data/larry_games/
```

Each file must end with `.pgn`.

### 3. Run the setup script

```bash
./setup_larrybot.sh
```

This will:

* Install dependencies
* Set up the `uv` Python environment
* Download the base Maia model
* Convert Larry’s PGNs into training data
* Fine-tune the network
* Blend a stronger variant (~2300 ELO)
* Configure a ready-to-run Lichess bot

> Note: If you run this inside your Git repo, the following directories will be created:
> `maia-chess/`, `lichess-bot/`, `models/`, `data/`.
> Make sure to add them to `.gitignore` to avoid committing large files.

---

### 4. Add your Lichess API token

Edit `lichess-bot/config.yml`:

```yaml
token: "YOUR_TOKEN_HERE"
```

Replace `"YOUR_TOKEN_HERE"` with your Lichess token (must have *Play games with the bot API* permission).

---

### 5. Start the bot

```bash
uv run python lichess-bot/lichess-bot.py
```

Your Lichess bot will appear online and can accept challenges.

---

## 🧩 Directory Layout

```
larry/
 ├── setup_larrybot.sh
 ├── data/
 │   ├── larry_games.pgn
 │   └── larry_dataset.txt
 ├── models/
 │   ├── larry_maia/best.pb.gz
 │   └── larry_maia_2300.pb.gz
 ├── maia-chess/
 ├── lichess-bot/
 └── README.md
```

---

## 🎚️ Difficulty Levels

| Model                   | Approx ELO                                   | Description                  |
| ----------------------- | -------------------------------------------- | ---------------------------- |
| `larry_maia/best.pb.gz` | ~2200                                        | Default trained Larry model  |
| `larry_maia_2300.pb.gz` | ~2300                                        | Sharper variant, harder play |
| Lower levels            | Adjust move time in `lichess-bot/config.yml` |                              |

Example:

```yaml
uci_options:
  MoveTime: 0.5
```

---

## 🔄 Updating / Retraining

If you add more PGNs later, you can retrain:

```bash
uv run python maia-chess/train.py \
  --train-data data/larry_dataset.txt \
  --val-data data/larry_dataset.txt \
  --init-from models/larry_maia/best.pb.gz \
  --save-dir models/larry_maia \
  --epochs 2 \
  --batch-size 32 \
  --lr 1e-4 \
  --use-gpu
```

Then rebuild higher-strength variants using:

```bash
uv run python maia-chess/scripts/blend_models.py \
  --a models/larry_maia/best.pb.gz \
  --b maia-chess/data/maia-1900.pb.gz \
  --alpha 0.8 \
  --out models/larry_maia_2300.pb.gz
```

---

## ⚡ Optional: Human-Like Move Delays

To simulate a real opponent, add a random delay in `lichess-bot.py` before making a move:

```python
import random, time
time.sleep(random.uniform(0.4, 2.5))
```

This makes LarryBot’s tempo feel natural during practice games.

---

Do you want me to also write a **`.gitignore` section** you can include in this repo so it ignores all the large directories automatically?
