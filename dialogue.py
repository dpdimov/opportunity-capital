"""
Dialogue mechanics implementing Dor's communication pipeline.

Orchestrates one full round of the entrepreneur-investor dialogue:
  1. Entrepreneur formalize + utter
  2. Investor decode + imagine
  3. Investor respond
  4. Evaluator assess alignment

Also handles alignment tracking, early termination, and KTS integration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from llm_client import query_llm
from capital import compute_opportunity_capital, OpportunityCapital
from sim_state import DialogueState, RoundState
from prompts import (
    entrepreneur_formalize_utter,
    investor_decode_imagine,
    investor_respond,
    evaluator_alignment,
)

log = logging.getLogger(__name__)


# ── Score extraction helpers ────────────────────────────────────────────────

REALISM_MAP = {
    "unrealistic": 1.5,
    "questionable": 3.5,
    "plausible": 5.5,
    "convincing": 7.5,
    "highly compelling": 9.0,
}

CREDIBILITY_MAP = {
    "not credible": 1.5,
    "somewhat credible": 3.5,
    "credible": 5.5,
    "very credible": 7.5,
    "exceptionally credible": 9.0,
}


def _verbal_to_score(verbal: str, mapping: Dict[str, float]) -> float:
    """Convert a verbal assessment to a numeric score via fuzzy matching."""
    verbal_lower = verbal.strip().lower()
    for key, score in mapping.items():
        if key in verbal_lower:
            return score
    # Fallback: moderate
    return 5.0


# ── Single Round Execution ──────────────────────────────────────────────────

def run_round(
    state: DialogueState,
    round_number: int,
    model: str = "claude-sonnet-4-6",
    evaluator_model: str = "claude-sonnet-4-6",
    temperature: float = 0.7,
    commitment_threshold: float = 8.0,
    legitimacy_threshold: float = 6.0,
    verbose: bool = False,
) -> RoundState:
    """Execute one full round of the Dor communication pipeline.

    Returns a RoundState with all fields populated.
    """
    round_state = RoundState(round_number=round_number)

    # Determine per-agent models
    e_model = state.entrepreneur.model or model
    i_model = state.investor.model or model

    llm_kwargs = {"temperature": temperature, "max_tokens": 4096}

    # ── Step 1: Entrepreneur formalize + utter ──────────────────────────
    if verbose:
        print(f"  [Round {round_number}] Entrepreneur formulating pitch...")

    sys_p, usr_p = entrepreneur_formalize_utter(state, round_number)
    e_result = query_llm(sys_p, usr_p, model=e_model, **llm_kwargs)

    round_state.entrepreneur_message = str(e_result.get("message", ""))
    round_state.entrepreneur_utterance = str(e_result.get("utterance", ""))

    if verbose:
        print(f"  [Round {round_number}] Entrepreneur uttered: {round_state.entrepreneur_utterance[:200]}...")

    # ── Step 2: Investor decode + imagine ───────────────────────────────
    if verbose:
        print(f"  [Round {round_number}] Investor decoding and imagining...")

    sys_p, usr_p = investor_decode_imagine(
        state, round_state.entrepreneur_utterance, round_number
    )
    i_decode_result = query_llm(sys_p, usr_p, model=i_model, **llm_kwargs)

    decoded_message = i_decode_result.get("decoded_message", {})
    round_state.investor_message_star = str(decoded_message)
    round_state.investor_imagined_future = str(
        i_decode_result.get("imagined_future", "")
    )

    if verbose:
        print(f"  [Round {round_number}] Investor imagined: {round_state.investor_imagined_future[:200]}...")

    # ── Step 3: Investor respond ────────────────────────────────────────
    if verbose:
        print(f"  [Round {round_number}] Investor responding...")

    sys_p, usr_p = investor_respond(
        state,
        round_state.investor_imagined_future,
        decoded_message,
        round_number,
    )
    i_respond_result = query_llm(sys_p, usr_p, model=i_model, **llm_kwargs)

    round_state.investor_response = str(i_respond_result.get("response", ""))

    # Extract internal assessment
    assessment = i_respond_result.get("internal_assessment", {})
    story_realism_verbal = str(assessment.get("story_realism", "plausible"))
    protagonist_credibility_verbal = str(
        assessment.get("protagonist_credibility", "credible")
    )
    commitment_direction = str(assessment.get("commitment_direction", ""))

    story_realism_score = _verbal_to_score(story_realism_verbal, REALISM_MAP)
    protagonist_credibility_score = _verbal_to_score(
        protagonist_credibility_verbal, CREDIBILITY_MAP
    )

    if verbose:
        print(f"  [Round {round_number}] Investor responded: {round_state.investor_response[:200]}...")
        print(f"  [Round {round_number}] Investor assessment: realism={story_realism_verbal}, credibility={protagonist_credibility_verbal}")

    # ── Step 4: Evaluator alignment assessment ──────────────────────────
    if verbose:
        print(f"  [Round {round_number}] Evaluator assessing alignment...")

    # Parse entrepreneur message for evaluator
    e_message_dict = e_result.get("message", {})
    if isinstance(e_message_dict, str):
        e_message_dict = {"content": e_message_dict}

    sys_p, usr_p = evaluator_alignment(
        e_message_dict,
        decoded_message if isinstance(decoded_message, dict) else {"content": str(decoded_message)},
        round_number,
        state.story.archetype,
    )
    eval_result = query_llm(sys_p, usr_p, model=evaluator_model, **llm_kwargs)

    round_state.alignment_score = float(eval_result.get("alignment_score", 5.0))
    round_state.alignment_gaps = str(eval_result.get("alignment_assessment", ""))
    round_state.legitimacy_score = float(eval_result.get("legitimacy_score", 5.0))
    round_state.commitment_inclination = float(
        eval_result.get("commitment_inclination", 5.0)
    )

    if verbose:
        print(
            f"  [Round {round_number}] Evaluator: alignment={round_state.alignment_score:.1f}, "
            f"legitimacy={round_state.legitimacy_score:.1f}, "
            f"commitment={round_state.commitment_inclination:.1f}"
        )

    # ── Step 5: Compute opportunity capital ─────────────────────────────
    # Blend evaluator scores with investor self-assessment
    blended_realism = 0.5 * story_realism_score + 0.5 * round_state.legitimacy_score
    blended_credibility = 0.5 * protagonist_credibility_score + 0.5 * (
        round_state.alignment_score * 0.6 + round_state.commitment_inclination * 0.4
    )

    round_state.opportunity_capital = compute_opportunity_capital(
        story_realism=blended_realism,
        protagonist_credibility=blended_credibility,
    )

    if verbose:
        oc = round_state.opportunity_capital
        print(
            f"  [Round {round_number}] Opportunity capital: "
            f"realism={oc.story_realism:.1f}, credibility={oc.protagonist_credibility:.1f}, "
            f"score={oc.opportunity_capital_score:.1f}"
        )

    # ── Step 6: Check for early termination ─────────────────────────────
    if round_state.commitment_inclination >= commitment_threshold:
        round_state.investor_commits = True
        if verbose:
            print(f"  [Round {round_number}] ** INVESTOR COMMITS **")

    if (
        round_state.legitimacy_score < legitimacy_threshold * 0.5
        and round_number >= 2
    ):
        # Investor finds the story fundamentally unrealistic after at least 2 rounds
        round_state.investor_rejects = True
        round_state.rejection_reason = (
            f"Story legitimacy too low ({round_state.legitimacy_score:.1f}) "
            f"after {round_number} rounds."
        )
        if verbose:
            print(f"  [Round {round_number}] ** INVESTOR REJECTS: {round_state.rejection_reason} **")

    return round_state


# ── KTS Scoring (optional) ──────────────────────────────────────────────────

def score_kts(
    entrepreneur_utterance: str,
    investor_response: str,
    method: str = "heuristic",
    model: str = "claude-sonnet-4-6",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Score entrepreneur and investor text using KTS framework.

    Returns (entrepreneur_kts_dict, investor_kts_dict).
    """
    import sys
    from pathlib import Path

    kts_dir = str(Path(__file__).resolve().parent.parent / "kts")
    if kts_dir not in sys.path:
        sys.path.insert(0, kts_dir)

    from kts_analyzer import analyze_text

    e_kts = analyze_text(entrepreneur_utterance, method=method, model=model)
    i_kts = analyze_text(investor_response, method=method, model=model)

    return e_kts.to_dict(), i_kts.to_dict()
