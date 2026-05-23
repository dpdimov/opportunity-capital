# A Computational Thought Experiment on Opportunity Capital with LLM Agents

This is the code for Dimov, D. (2026). Giving Texture to Theory: A Computational
Thought Experiment on Opportunity Capital with LLM Agents. *Journal of Business
Venturing Insights*, In press.

The thought experiment runs across a 4×2 factorial design: four story archetypes
crossed with two levels of founder–venture fit (high and low). This design creates
8 distinct scenarios in which an entrepreneur of specified fit and with specified
story archetype goes through five rounds of dialogue with an investor. This is an
operationalisation of Dimov, D., & Günestepe, K. (2024). Capitalizing the future:
Opportunity capital as symbolic significance of an entrepreneur's future-venture
story. *Entrepreneurship & Regional Development*, 36(9–10), 1145–1160.

## Install

```
git clone git@github.com:dpdimov/opportunity-capital.git
cd opportunity-capital
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

You need at least one provider key set, matching the models you intend to run
(see **Models** below).

## Run

```
python future_sim.py                  # all 8 scenarios
python future_sim.py --triad 0        # single scenario (0–7)
python future_sim.py --verbose        # full transcripts per round
python future_sim.py --kts            # enable KTS thinking-style scoring
python future_sim.py --seed 42        # reproducible model_pool allocation
```

Each run writes to `results/<YYYYMMDD_HHMMSS>/`:
`dialogue.csv`, `alignment.csv`, `capital.csv`, `scores.json`, `comparative.json`
(plus `kts_scores.csv` if `--kts` is set).

## Design

`config/sim_config.json` defines 8 scenarios:

| ID  | Archetype    | Fit  | Entrepreneur | Investor     |
|-----|--------------|------|--------------|--------------|
| T1a | prosaic      | high | Jamie Osei   | Sofia Reyes  |
| T1b | prosaic      | low  | Rachel Kim   | Sofia Reyes  |
| T2a | revelatory   | high | Marcus Reid  | Sofia Reyes  |
| T2b | revelatory   | low  | Divya Patel  | Sofia Reyes  |
| T3a | metaphorical | high | Aya Chen     | Priya Sharma |
| T3b | metaphorical | low  | Ben Nakamura | Priya Sharma |
| T4a | fantastic    | high | Kenji Tanaka | Priya Sharma |
| T4b | fantastic    | low  | Lena Okafor  | Priya Sharma |

Investor assignment is domain-matched: **Sofia Reyes** for consumer/operations
stories (prosaic, revelatory); **Priya Sharma** for deep-tech stories
(metaphorical, fantastic).

Key variables (all 0–10, scored by the evaluator each round):

- **Opportunity Capital** = √(Story Realism × Protagonist Credibility) — geometric mean
- **Commitment** — investor inclination to invest; investment triggers at ≥ 8
- **Legitimacy** — how realistic the investor finds the venture story
- **Alignment** — convergence between the entrepreneur's intended and the investor's decoded message
- **Commitment Gap** = Legitimacy − Commitment

## Models

By default all three roles (entrepreneur, investor, evaluator) use
`claude-sonnet-4-6`. You can override:

- **Per persona** — edit the `model` field in any `config/entrepreneur_*.json`
  or `config/investor_*.json`.
- **Evaluator** — set `evaluator_model` in `config/sim_config.json`.
- **Randomized allocation** — add a `model_pool` list to `sim_config.json` to
  randomly assign models across entrepreneur/investor slots (see
  `allocate_models` in `future_sim.py`).

Provider routing in `llm_client.py` keys off the model-name prefix:

- `claude-*` → Anthropic
- `gemini-*` → Google GenAI
- `gpt-*` / `o1-*` / `o3-*` / `o4-*` → OpenAI

API keys are read from `.env` (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`OPENAI_API_KEY`).

## File map

- `future_sim.py` — CLI entrypoint, scenario loop, output writers
- `dialogue.py` — single-round execution (entrepreneur utterance → investor response → scoring)
- `sim_state.py` — dataclasses for personas, rounds, dialogue state
- `capital.py` — entrepreneur/investor capital profiles, opportunity capital (geometric mean)
- `stories.py` — the four archetype definitions (prosaic/revelatory/metaphorical/fantastic)
- `prompts.py` — all LLM prompts
- `scoring.py` — per-scenario scoring and comparative analysis
- `llm_client.py` — multi-provider query wrapper with JSON extraction + retries
- `config/` — 8 entrepreneur personas, 2 investor personas, `sim_config.json`

## License

Creative Commons Attribution-NonCommercial 4.0 International (see `LICENSE`).
