"""
Final evaluation and scoring for the opportunity capital simulation.

Assesses the complete dialogue trajectory and produces final scores
for each triad (entrepreneur x investor x story configuration).
"""

from __future__ import annotations

from typing import Any, Dict, List

from sim_state import DialogueState, RoundState


def score_triad(state: DialogueState) -> Dict[str, Any]:
    """Produce final scores for a completed dialogue.

    Evaluates:
      - Imagination alignment: Did the two parties converge?
      - Capital recognition: Did the investor recognize entrepreneur's capital?
      - Legitimacy achieved: Was the story perceived as realistic?
      - Commitment achieved: Did the investor commit?
      - Opportunity capital weight: Final symbolic value
    """
    rounds = state.rounds
    if not rounds:
        return _empty_scores(state)

    # Trajectory analysis
    alignment_scores = [r.alignment_score for r in rounds]
    legitimacy_scores = [r.legitimacy_score for r in rounds]
    commitment_scores = [r.commitment_inclination for r in rounds]
    oc_scores = [
        r.opportunity_capital.opportunity_capital_score
        for r in rounds
        if r.opportunity_capital
    ]

    # Convergence: is alignment improving over rounds?
    if len(alignment_scores) >= 2:
        alignment_trend = alignment_scores[-1] - alignment_scores[0]
    else:
        alignment_trend = 0.0

    # Final values
    final_alignment = alignment_scores[-1]
    final_legitimacy = legitimacy_scores[-1]
    final_commitment = commitment_scores[-1]
    final_oc = oc_scores[-1] if oc_scores else 0.0
    peak_oc = max(oc_scores) if oc_scores else 0.0

    # Outcome determination
    outcome = state.outcome
    if not outcome:
        if any(r.investor_commits for r in rounds):
            outcome = "committed"
        elif any(r.investor_rejects for r in rounds):
            outcome = "rejected"
        else:
            outcome = "stalemate"

    return {
        "triad_id": state.triad_id,
        "entrepreneur": state.entrepreneur.name,
        "investor": state.investor.name,
        "story_archetype": state.story.archetype,
        "story_name": state.story.name,
        "rounds_completed": len(rounds),
        "outcome": outcome,

        # Final scores
        "final_alignment": round(final_alignment, 2),
        "final_legitimacy": round(final_legitimacy, 2),
        "final_commitment": round(final_commitment, 2),
        "final_opportunity_capital": round(final_oc, 2),
        "peak_opportunity_capital": round(peak_oc, 2),

        # Trajectory
        "alignment_trend": round(alignment_trend, 2),
        "alignment_trajectory": [round(s, 2) for s in alignment_scores],
        "legitimacy_trajectory": [round(s, 2) for s in legitimacy_scores],
        "commitment_trajectory": [round(s, 2) for s in commitment_scores],
        "opportunity_capital_trajectory": [round(s, 2) for s in oc_scores],

        # Averages
        "avg_alignment": round(sum(alignment_scores) / len(alignment_scores), 2),
        "avg_legitimacy": round(sum(legitimacy_scores) / len(legitimacy_scores), 2),
        "avg_commitment": round(sum(commitment_scores) / len(commitment_scores), 2),
        "avg_opportunity_capital": round(sum(oc_scores) / len(oc_scores), 2) if oc_scores else 0.0,

        # Models used
        "entrepreneur_model": state.entrepreneur.model,
        "investor_model": state.investor.model,
    }


def score_comparative(triad_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce cross-triad comparative analysis."""
    if not triad_scores:
        return {"triads": [], "summary": "No triads to compare."}

    # Rank by final opportunity capital
    ranked = sorted(
        triad_scores,
        key=lambda t: t["final_opportunity_capital"],
        reverse=True,
    )

    # Summary statistics
    all_oc = [t["final_opportunity_capital"] for t in triad_scores]
    all_alignment = [t["final_alignment"] for t in triad_scores]
    committed = [t for t in triad_scores if t["outcome"] == "committed"]
    rejected = [t for t in triad_scores if t["outcome"] == "rejected"]

    return {
        "triads": ranked,
        "n_triads": len(triad_scores),
        "n_committed": len(committed),
        "n_rejected": len(rejected),
        "n_stalemate": len(triad_scores) - len(committed) - len(rejected),
        "avg_opportunity_capital": round(sum(all_oc) / len(all_oc), 2),
        "max_opportunity_capital": round(max(all_oc), 2),
        "min_opportunity_capital": round(min(all_oc), 2),
        "avg_alignment": round(sum(all_alignment) / len(all_alignment), 2),
        "highest_oc_triad": ranked[0]["triad_id"] if ranked else None,
        "lowest_oc_triad": ranked[-1]["triad_id"] if ranked else None,
    }


def _empty_scores(state: DialogueState) -> Dict[str, Any]:
    """Return empty scores for a triad with no completed rounds."""
    return {
        "triad_id": state.triad_id,
        "entrepreneur": state.entrepreneur.name,
        "investor": state.investor.name,
        "story_archetype": state.story.archetype,
        "story_name": state.story.name,
        "rounds_completed": 0,
        "outcome": "no_dialogue",
        "final_alignment": 0.0,
        "final_legitimacy": 0.0,
        "final_commitment": 0.0,
        "final_opportunity_capital": 0.0,
        "peak_opportunity_capital": 0.0,
        "alignment_trend": 0.0,
        "alignment_trajectory": [],
        "legitimacy_trajectory": [],
        "commitment_trajectory": [],
        "opportunity_capital_trajectory": [],
        "avg_alignment": 0.0,
        "avg_legitimacy": 0.0,
        "avg_commitment": 0.0,
        "avg_opportunity_capital": 0.0,
        "entrepreneur_model": state.entrepreneur.model,
        "investor_model": state.investor.model,
    }
