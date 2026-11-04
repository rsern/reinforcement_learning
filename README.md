# Reinforcement Learning Workshop: Q-Learning from Scratch

Teach an agent to escape a grid maze using nothing but Python and NumPy.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/YOUR-ORG/YOUR-REPO)

## Get started (about 2 minutes)

1. Click the **Open in GitHub Codespaces** badge above (or **Code > Codespaces > Create codespace on main**).
2. Wait for the setup to finish. It installs everything automatically.
3. Open **`rl_demo.ipynb`**.
4. If asked to pick a kernel, choose **Python Environments > Python 3.11**.
5. Click a code cell and press **Shift + Enter** to run it. Go top to bottom.

Stuck? Click **Restart** at the top of the notebook, then **Run All**.

## What's in here

| File | What it is |
|---|---|
| `rl_demo.ipynb` | The workshop notebook. This is the only file you need to open. |
| `plotting.py` | Drawing helpers. You don't need to read it. |
| `requirements.txt` | Python packages (numpy, matplotlib). |
| `.devcontainer/` | Tells Codespaces how to set up the environment. |

## Running locally instead

```bash
pip install -r requirements.txt jupyter
jupyter notebook rl_demo.ipynb
```

## Want to go further?

- [Gymnasium](https://gymnasium.farama.org/): ready-made RL environments. Try your Q-learning code on `FrozenLake-v1`.
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/): a gentle intro to deep RL.
- *Reinforcement Learning: An Introduction* by Sutton and Barto: the classic textbook, free online.

## Before the workshop (for the presenter)

- Replace `YOUR-ORG/YOUR-REPO` in the badge link with this repo's path.
- Free GitHub accounts include Codespaces hours each month, and students can get more through the GitHub Student Developer Pack.
- Create one codespace yourself ahead of time to confirm it builds cleanly.
