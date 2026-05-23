#!/usr/bin/env python3
"""
Capitalizing the Future: Opportunity Capital Simulation

Main simulation loop and CLI. Runs entrepreneur-investor dialogue triads
to study how opportunity capital accumulates through dialogical calibration.

Based on Dimov & Gunestepe (2024) "Capitalizing the future: opportunity capital
as symbolic significance of an entrepreneur's future-venture story".

Usage:
    python future_sim.py                          # Run with default config
    python future_sim.py --config config/sim_config.json
    python future_sim.py --verbose                # Full dialogue transcript
    python future_sim.py --kts                    # Enable KTS scoring
    python future_sim.py --triad 0                # Run only the first triad
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sim_state import DialogueState, EntrepreneurProfile, InvestorProfile
from stories import get_story
from dialogue import run_round, score_kts
from scoring import score_triad, score_comparative

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


# ── Config Loading ──────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    """Load simulation config from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def load_persona(persona_path: str) -> Dict[str, Any]:
    """Load an entrepreneur or investor persona from JSON."""
    # Resolve relative to project root
    full_path = BASE_DIR / persona_path
    with open(full_path) as f:
        return json.load(f)


# ── Output Writers ──────────────────────────────────────────────────────────

def write_dialogue_csv(
    run_dir: Path,
    all_rounds: List[Dict[str, Any]],
) -> None:
    """Write full dialogue transcript to CSV."""
    if not all_rounds:
        return
    filepath = run_dir / "dialogue.csv"
    fieldnames = [
        "triad_id", "round", "entrepreneur_name", "investor_name",
        "story_archetype", "entrepreneur_message", "entrepreneur_utterance",
        "investor_message_star", "investor_imagined_future", "investor_response",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rounds)


def write_alignment_csv(
    run_dir: Path,
    all_alignment: List[Dict[str, Any]],
) -> None:
    """Write per-round alignment scores to CSV."""
    if not all_alignment:
        return
    filepath = run_dir / "alignment.csv"
    fieldnames = [
        "triad_id", "round", "alignment_score", "alignment_gaps",
        "legitimacy_score", "commitment_inclination",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_alignment)


def write_capital_csv(
    run_dir: Path,
    all_capital: List[Dict[str, Any]],
) -> None:
    """Write per-round capital state to CSV."""
    if not all_capital:
        return
    filepath = run_dir / "capital.csv"
    fieldnames = [
        "triad_id", "round", "story_realism", "protagonist_credibility",
        "opportunity_capital_score",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_capital)


def write_kts_csv(
    run_dir: Path,
    all_kts: List[Dict[str, Any]],
) -> None:
    """Write per-round KTS scores to CSV."""
    if not all_kts:
        return
    filepath = run_dir / "kts_scores.csv"
    fieldnames = [
        "triad_id", "round",
        "entrepreneur_uncertainty", "entrepreneur_possibility", "entrepreneur_style",
        "investor_uncertainty", "investor_possibility", "investor_style",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_kts)


# ── Model Pool Allocation ──────────────────────────────────────────────────

def allocate_models(
    model_pool: List[str],
    n_triads: int,
    rng: random.Random | None = None,
) -> Dict[int, Dict[str, str]]:
    """Randomly assign models to entrepreneurs and investors.

    Guarantees every model in the pool appears at least once across
    all slots (2 * n_triads). Remaining slots are filled randomly.
    If *rng* is provided, uses it for reproducible randomness.
    """
    rng = rng or random.Random()
    total_slots = 2 * n_triads
    pool = list(model_pool)

    if len(pool) > total_slots:
        raise ValueError(
            f"model_pool has {len(pool)} models but only {total_slots} "
            f"slots ({n_triads} triads x 2). Cannot guarantee full coverage."
        )

    # Start with one guaranteed slot per model, then fill remaining randomly
    slots = list(pool)  # one per model
    remaining = total_slots - len(slots)
    slots.extend(rng.choices(pool, k=remaining))
    rng.shuffle(slots)

    assignments = {}
    for i in range(n_triads):
        assignments[i] = {
            "entrepreneur": slots[2 * i],
            "investor": slots[2 * i + 1],
        }
    return assignments


# ── Triad Execution ────────────────────────────────────────────────────────

def run_triad(
    triad_config: Dict[str, Any],
    triad_index: int,
    sim_config: Dict[str, Any],
    verbose: bool = False,
    kts_enabled: bool = False,
) -> tuple[DialogueState, List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Run a single entrepreneur-investor-story triad dialogue."""

    # Load personas
    e_data = load_persona(triad_config["entrepreneur"])
    i_data = load_persona(triad_config["investor"])
    story = get_story(triad_config["story"])

    entrepreneur = EntrepreneurProfile.from_dict(e_data)
    investor = InvestorProfile.from_dict(i_data)

    # Apply pre-assigned models if provided (from model_pool allocation)
    model_assignments = sim_config.get("_model_assignments", {})
    assignment = model_assignments.get(triad_index)
    if assignment:
        entrepreneur.model = assignment["entrepreneur"]
        investor.model = assignment["investor"]

    triad_id = f"triad_{triad_index + 1}"

    # Initialize dialogue state
    state = DialogueState(
        triad_id=triad_id,
        entrepreneur=entrepreneur,
        investor=investor,
        story=story,
    )

    # Config parameters
    model = sim_config.get("model", "claude-sonnet-4-6")
    evaluator_model = sim_config.get("evaluator_model", model)
    temperature = sim_config.get("temperature", 0.7)
    max_rounds = sim_config.get("max_rounds", 5)
    commitment_threshold = sim_config.get("commitment_threshold", 8.0)
    legitimacy_threshold = sim_config.get("legitimacy_threshold", 6.0)
    kts_method = sim_config.get("kts_method", "heuristic")

    print(f"\n{'='*60}")
    print(f"TRIAD {triad_index + 1}: {entrepreneur.name} x {investor.name} ({investor.firm})")
    print(f"Story: {story.name} ({story.archetype})")
    print(f"Models: entrepreneur={entrepreneur.model}, investor={investor.model}")
    print(f"{'='*60}")

    # Collect output rows
    dialogue_rows: List[Dict] = []
    alignment_rows: List[Dict] = []
    capital_rows: List[Dict] = []
    kts_rows: List[Dict] = []

    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Round {round_num}/{max_rounds} ---")

        round_state = run_round(
            state=state,
            round_number=round_num,
            model=model,
            evaluator_model=evaluator_model,
            temperature=temperature,
            commitment_threshold=commitment_threshold,
            legitimacy_threshold=legitimacy_threshold,
            verbose=verbose,
        )

        # Append to state
        state.rounds.append(round_state)

        # Collect CSV rows
        dialogue_rows.append({
            "triad_id": triad_id,
            "round": round_num,
            "entrepreneur_name": entrepreneur.name,
            "investor_name": investor.name,
            "story_archetype": story.archetype,
            "entrepreneur_message": round_state.entrepreneur_message[:2000],
            "entrepreneur_utterance": round_state.entrepreneur_utterance[:2000],
            "investor_message_star": round_state.investor_message_star[:2000],
            "investor_imagined_future": round_state.investor_imagined_future[:2000],
            "investor_response": round_state.investor_response[:2000],
        })

        alignment_rows.append({
            "triad_id": triad_id,
            "round": round_num,
            "alignment_score": round_state.alignment_score,
            "alignment_gaps": round_state.alignment_gaps[:1000],
            "legitimacy_score": round_state.legitimacy_score,
            "commitment_inclination": round_state.commitment_inclination,
        })

        if round_state.opportunity_capital:
            oc = round_state.opportunity_capital
            capital_rows.append({
                "triad_id": triad_id,
                "round": round_num,
                "story_realism": oc.story_realism,
                "protagonist_credibility": oc.protagonist_credibility,
                "opportunity_capital_score": oc.opportunity_capital_score,
            })

        # Optional KTS scoring
        if kts_enabled:
            try:
                e_kts, i_kts = score_kts(
                    round_state.entrepreneur_utterance,
                    round_state.investor_response,
                    method=kts_method,
                    model=model,
                )
                kts_rows.append({
                    "triad_id": triad_id,
                    "round": round_num,
                    "entrepreneur_uncertainty": e_kts["uncertainty_score"],
                    "entrepreneur_possibility": e_kts["possibility_score"],
                    "entrepreneur_style": e_kts["style"],
                    "investor_uncertainty": i_kts["uncertainty_score"],
                    "investor_possibility": i_kts["possibility_score"],
                    "investor_style": i_kts["style"],
                })
            except Exception as e:
                log.warning("KTS scoring failed for round %d: %s", round_num, e)

        # Print round summary
        oc = round_state.opportunity_capital
        oc_str = f"{oc.opportunity_capital_score:.1f}" if oc else "N/A"
        print(
            f"  Alignment: {round_state.alignment_score:.1f} | "
            f"Legitimacy: {round_state.legitimacy_score:.1f} | "
            f"Commitment: {round_state.commitment_inclination:.1f} | "
            f"Opportunity Capital: {oc_str}"
        )

        # Check early termination
        if round_state.investor_commits:
            state.outcome = "committed"
            state.final_round = round_num
            print(f"\n  >> INVESTOR COMMITS in round {round_num}!")
            break
        elif round_state.investor_rejects:
            state.outcome = "rejected"
            state.final_round = round_num
            print(f"\n  >> INVESTOR REJECTS in round {round_num}: {round_state.rejection_reason}")
            break
    else:
        state.outcome = "stalemate"
        state.final_round = max_rounds
        print(f"\n  >> Dialogue ended: stalemate after {max_rounds} rounds")

    return state, dialogue_rows, alignment_rows, capital_rows, kts_rows


# ── Main Simulation ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Capitalizing the Future: Opportunity Capital Simulation"
    )
    parser.add_argument(
        "--config", default="config/sim_config.json",
        help="Path to simulation config JSON",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full dialogue transcript each round",
    )
    parser.add_argument(
        "--kts", action="store_true",
        help="Enable KTS thinking-style scoring",
    )
    parser.add_argument(
        "--triad", type=int, default=None,
        help="Run only a specific triad (0-indexed)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible model assignments",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load config
    config_path = BASE_DIR / args.config
    config = load_config(config_path)

    kts_enabled = args.kts or config.get("kts_enabled", False)

    # Seeded RNG for reproducible model allocation
    seed = args.seed if args.seed is not None else config.get("seed")
    rng = random.Random(seed)

    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output_dir or config.get("output_dir", "results")
    run_dir = BASE_DIR / output_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {run_dir}")

    # Select triads to run
    triads = config.get("triads", [])
    if args.triad is not None:
        if args.triad >= len(triads):
            print(f"Error: triad index {args.triad} out of range (0-{len(triads)-1})")
            sys.exit(1)
        triads_to_run = [(args.triad, triads[args.triad])]
    else:
        triads_to_run = list(enumerate(triads))

    # Allocate models from pool if configured
    model_pool = config.get("model_pool")
    if model_pool:
        indices = [idx for idx, _ in triads_to_run]
        assignments = allocate_models(model_pool, len(indices), rng=rng)
        # Remap allocation indices to actual triad indices
        config["_model_assignments"] = {
            indices[i]: assignments[i] for i in range(len(indices))
        }
        print(f"Model pool: {model_pool}")
        for idx, assign in config["_model_assignments"].items():
            print(f"  Triad {idx+1}: entrepreneur={assign['entrepreneur']}, investor={assign['investor']}")

    # Run all triads
    all_dialogue = []
    all_alignment = []
    all_capital = []
    all_kts = []
    triad_scores = []

    for idx, triad_config in triads_to_run:
        state, d_rows, a_rows, c_rows, k_rows = run_triad(
            triad_config=triad_config,
            triad_index=idx,
            sim_config=config,
            verbose=args.verbose,
            kts_enabled=kts_enabled,
        )

        all_dialogue.extend(d_rows)
        all_alignment.extend(a_rows)
        all_capital.extend(c_rows)
        all_kts.extend(k_rows)

        # Score this triad
        scores = score_triad(state)
        triad_scores.append(scores)

    # Write outputs
    write_dialogue_csv(run_dir, all_dialogue)
    write_alignment_csv(run_dir, all_alignment)
    write_capital_csv(run_dir, all_capital)
    if kts_enabled:
        write_kts_csv(run_dir, all_kts)

    # Write final scores
    with open(run_dir / "scores.json", "w") as f:
        json.dump(triad_scores, f, indent=2)

    # Write comparative analysis
    comparative = score_comparative(triad_scores)
    comparative["config"] = {
        "model": config.get("model"),
        "evaluator_model": config.get("evaluator_model"),
        "temperature": config.get("temperature"),
        "max_rounds": config.get("max_rounds"),
        "commitment_threshold": config.get("commitment_threshold"),
        "legitimacy_threshold": config.get("legitimacy_threshold"),
        "kts_enabled": kts_enabled,
        "run_id": run_id,
    }
    with open(run_dir / "comparative.json", "w") as f:
        json.dump(comparative, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Triads run: {len(triad_scores)}")
    print(f"Committed: {comparative.get('n_committed', 0)}")
    print(f"Rejected: {comparative.get('n_rejected', 0)}")
    print(f"Stalemate: {comparative.get('n_stalemate', 0)}")
    print(f"Avg opportunity capital: {comparative.get('avg_opportunity_capital', 0):.2f}")
    print(f"\nResults saved to: {run_dir}")

    for scores in triad_scores:
        print(f"\n  {scores['triad_id']}: {scores['entrepreneur']} x {scores['investor']}")
        print(f"    Story: {scores['story_name']} ({scores['story_archetype']})")
        print(f"    Outcome: {scores['outcome']} (round {scores['rounds_completed']})")
        print(f"    Final OC: {scores['final_opportunity_capital']:.2f} | Peak: {scores['peak_opportunity_capital']:.2f}")
        print(f"    Alignment: {scores['final_alignment']:.2f} (trend: {scores['alignment_trend']:+.2f})")


if __name__ == "__main__":
    main()
