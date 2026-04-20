# Chess Game Playing with Machine Learning

This project is a chess game for the **CO3061 – Introduction to Artificial Intelligence** assignment. It combines:

- a complete playable chess GUI built with **Pygame**
- a **Search-based baseline agent** using minimax + alpha-beta pruning
- a **Machine Learning-based chess agent** using a lightweight trainable linear model
- a **Random rule-based agent** for baseline evaluation

The project is designed to support demonstration, evaluation, and comparison between the **ML agent** and the **Search agent**.

---

## 1. Project Objectives

The current version of the project supports these goals:

- show that the chess game runs correctly
- demonstrate a playable adversarial game with legal chess moves
- evaluate the ML agent against a random rule-based opponent
- compare the ML agent against the Search baseline
- provide multiple ML skill levels: **poor / average / good**
- provide a simple benchmark mode for report and presentation results

---

## 2. Project Structure

```text
.
├── Chess_new.py                # Main Pygame GUI application
├── Chess_AI.py                 # Search-based agent (minimax + alpha-beta)
├── Chess_ML_Method.py         # Machine Learning-based agent
├── random_rule_based_agent.py # Random legal-move agent
├── chess_ml_model.json        # Saved ML model file (generated automatically)
└── README.md                  # Project documentation
```

### File Roles

#### `Chess_new.py`
Main GUI and interaction logic.

It contains:
- board rendering
- buttons and control panel
- game mode switching
- auto-play logic
- benchmark display for **Search vs ML**

#### `Chess_AI.py`
Search baseline agent.

It uses:
- board evaluation function
- move ordering
- minimax
- alpha-beta pruning
- depth-based levels

#### `Chess_ML_Method.py`
Machine Learning-based agent.

It contains:
- feature extraction from chess positions
- a trainable linear model
- simple training with SGD
- move selection based on predicted board score
- level profiles for **poor / average / good**

#### `random_rule_based_agent.py`
Random baseline opponent.

It only selects a move randomly from legal moves provided by `python-chess`.

---

## 3. Requirements

Install Python packages before running:

```bash
pip install pygame python-chess
```

Recommended Python version:

```text
Python 3.10+
```

---

## 4. How to Run

Run the main GUI:

```bash
python Chess_new.py
```

If the ML model file does not exist yet, `Chess_ML_Method.py` may generate `chess_ml_model.json` automatically on first use.

---

## 5. Game Modes

The GUI currently supports **3 modes**.

### 5.1 Human vs Human

- **White:** Human
- **Black:** Human

Purpose:
- introduce the game
- show that the board, pieces, rules, and interaction work correctly

Use this mode in presentation when you want to demonstrate:
- legal move generation
- manual play
- promotion handling
- game-over detection

---

### 5.2 Random vs ML

- **White:** Random
- **Black:** ML

Purpose:
- demonstrate the required assignment condition where the ML agent plays against a random rule-based agent
- show that the ML agent has multiple skill levels

Available ML levels:
- `poor`
- `average`
- `good`

Use the **Level** button to switch between these levels.

This mode is mainly for:
- validating that the ML agent can beat a weaker baseline
- demonstrating the effect of difficulty levels in the video/demo

---

### 5.3 Search vs ML

- **White:** Search
- **Black:** ML

Purpose:
- compare the Machine Learning method against the Search baseline
- generate benchmark results for the report and presentation

This is the only mode where the **Benchmark** button is enabled.

---

## 6. Controls in the GUI

### Main Buttons

- **Undo**: undo the last move
- **Reset**: reset the board to the initial state
- **Human vs Human**: switch to manual mode
- **Random vs ML**: switch to random vs ML mode
- **Search vs ML**: switch to search vs ML mode
- **Run Benchmark**: run benchmark (**only active in Search vs ML**)
- **Start Auto / Stop Auto**: automatically let both sides play according to the current mode
- **Level**: switch ML level between `poor`, `average`, and `good`

---

## 7. Benchmark Explanation

Benchmark is only available in **Search vs ML** mode.

### What it does
It runs multiple automatic games between:
- the **Search agent**
- the **ML agent**

The benchmark records:
- **White / Black roles**
- **W-D-L (ML)** = Wins, Draws, Losses of the ML agent
- **Average time per move** for both Search and ML

### Current Benchmark Output
When benchmark finishes, the status panel shows:

- `White: Search | Black: ML`
- `W-D-L (ML): x-y-z`
- `Time avg/move: Search ...s | ML ...s`

### Why benchmark is useful
It helps the group:
- compare ML with the search baseline more objectively
- collect numbers for the report
- explain speed and performance trade-offs in the presentation

---

## 8. ML Levels

The ML agent supports three difficulty levels:

### `poor`
- more exploration
- weaker move quality
- useful for showing level difference

### `average`
- balanced performance
- suitable for default tests and demos

### `good`
- strongest ML setting in this project
- more deterministic and less exploratory

In the GUI, the same level name is also used in **Search vs ML** mode so that the comparison stays simple and consistent.

---

## 9. Search Agent Summary

The Search baseline in `Chess_AI.py` uses:

- minimax
- alpha-beta pruning
- piece-square evaluation
- named strength levels mapped to search depth

This agent is used as the **baseline search algorithm** for comparison.

---

## 10. ML Agent Summary

The ML method in `Chess_ML_Method.py` uses:

- handcrafted board features
- a lightweight **linear evaluation model**
- optional bootstrapping from the Search evaluation function
- SGD training for weight adjustment
- move selection from predicted board scores

This makes the project easy to run and explain in class without requiring large external datasets or GPU training.

---

## 11. Suggested Demo Flow for Presentation

A good presentation order is:

1. **Human vs Human**
   - introduce the board and game rules
   - show that the game runs correctly

2. **Random vs ML**
   - show that ML can play against a weak baseline
   - demonstrate level changes: poor / average / good

3. **Search vs ML**
   - explain that Search is the baseline algorithm
   - run benchmark
   - present W-D-L and time comparison

---

## 12. Notes

- The benchmark may take longer when agent strength is higher.
- `Run Benchmark` is intentionally limited to **Search vs ML** for a cleaner evaluation flow.
- In the current GUI design, benchmark information is shown directly in the status panel.
- If `icon.png` is missing, the program still runs normally with the default window icon.

---
## 13. Quick Start

```bash
pip install pygame python-chess
python .\Chess_new.py
```

Then in the GUI:
- choose a mode
- use **Start Auto** for automatic play
- use **Level** to change ML strength
- use **Run Benchmark** in **Search vs ML** to collect comparison results

