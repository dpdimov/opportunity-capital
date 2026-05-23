"""
State models for the opportunity capital dialogue simulation.

Tracks: entrepreneur profile, investor profile, dialogue rounds,
and the evolving shared imaginary between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from capital import EntrepreneurCapital, InvestorCapital, OpportunityCapital
from stories import FutureVentureStory


# ── Persona Profiles ────────────────────────────────────────────────────────

@dataclass
class EntrepreneurProfile:
    """Full entrepreneur persona for the simulation."""
    name: str
    background: str
    model: str = "claude-sonnet-4-6"
    capital: EntrepreneurCapital = field(default_factory=EntrepreneurCapital)
    personality: str = ""           # Communication style, tendencies
    motivation: str = ""            # Why pursuing this venture

    def to_verbal(self) -> str:
        lines = [
            f"Name: {self.name}",
            f"Background: {self.background}",
        ]
        if self.personality:
            lines.append(f"Communication style: {self.personality}")
        if self.motivation:
            lines.append(f"Motivation: {self.motivation}")
        lines.append(f"\nCapital profile:\n{self.capital.to_verbal()}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntrepreneurProfile":
        capital_data = d.pop("capital", {})
        capital = EntrepreneurCapital.from_dict(capital_data) if capital_data else EntrepreneurCapital()
        return cls(capital=capital, **{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class InvestorProfile:
    """Full investor persona for the simulation."""
    name: str
    firm: str
    background: str
    model: str = "claude-sonnet-4-6"
    capital: InvestorCapital = field(default_factory=InvestorCapital)
    personality: str = ""
    investment_thesis: str = ""     # What they look for

    def to_verbal(self) -> str:
        lines = [
            f"Name: {self.name} ({self.firm})",
            f"Background: {self.background}",
        ]
        if self.personality:
            lines.append(f"Communication style: {self.personality}")
        if self.investment_thesis:
            lines.append(f"Investment thesis: {self.investment_thesis}")
        lines.append(f"\nCapital profile:\n{self.capital.to_verbal()}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InvestorProfile":
        capital_data = d.pop("capital", {})
        capital = InvestorCapital.from_dict(capital_data) if capital_data else InvestorCapital()
        return cls(capital=capital, **{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Dialogue Round State ────────────────────────────────────────────────────

@dataclass
class RoundState:
    """Full state of a single dialogue round."""
    round_number: int

    # Entrepreneur side (Dor's formalize + utter)
    entrepreneur_message: str = ""          # Structured venture concept (the signified)
    entrepreneur_utterance: str = ""        # The pitch text (the signifier)

    # Investor side (Dor's decode + imagine + respond)
    investor_message_star: str = ""         # Decoded/reconstructed venture concept
    investor_imagined_future: str = ""      # Rich imagined experience
    investor_response: str = ""             # Questions, challenges, signals

    # Evaluator assessments
    alignment_score: float = 0.0            # 0-10: message vs message* alignment
    alignment_gaps: str = ""                # Description of where they diverge
    legitimacy_score: float = 0.0           # 0-10: does investor find it realistic?
    commitment_inclination: float = 0.0     # 0-10: moving toward investment?

    # Opportunity capital (computed)
    opportunity_capital: Optional[OpportunityCapital] = None

    # Early termination signals
    investor_commits: bool = False
    investor_rejects: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.opportunity_capital:
            d["opportunity_capital"] = self.opportunity_capital.to_dict()
        return d


# ── Dialogue Session State ──────────────────────────────────────────────────

@dataclass
class DialogueState:
    """Complete state of one entrepreneur-investor dialogue session."""
    triad_id: str
    entrepreneur: EntrepreneurProfile = field(default_factory=lambda: EntrepreneurProfile("", ""))
    investor: InvestorProfile = field(default_factory=lambda: InvestorProfile("", "", ""))
    story: FutureVentureStory = field(default_factory=lambda: FutureVentureStory("", "", "", "", "", "", {}))
    rounds: List[RoundState] = field(default_factory=list)
    outcome: str = ""               # "committed" / "rejected" / "stalemate"
    final_round: int = 0

    def latest_round(self) -> Optional[RoundState]:
        return self.rounds[-1] if self.rounds else None

    def dialogue_history_verbal(self) -> str:
        """Build verbal summary of dialogue so far for LLM context."""
        if not self.rounds:
            return "This is the first round of the conversation."

        lines = []
        for r in self.rounds:
            lines.append(f"--- Round {r.round_number} ---")
            lines.append(f"Entrepreneur pitched: {r.entrepreneur_utterance[:500]}")
            lines.append(f"Investor responded: {r.investor_response[:500]}")
            lines.append("")
        return "\n".join(lines)

    def capital_trajectory_verbal(self) -> str:
        """Verbal summary of how opportunity capital has evolved."""
        if not self.rounds:
            return "No capital assessment yet."

        lines = []
        for r in self.rounds:
            if r.opportunity_capital:
                oc = r.opportunity_capital
                lines.append(
                    f"Round {r.round_number}: "
                    f"story realism={oc.story_realism:.1f}, "
                    f"protagonist credibility={oc.protagonist_credibility:.1f}, "
                    f"opportunity capital={oc.opportunity_capital_score:.1f}"
                )
        return "\n".join(lines) if lines else "No capital trajectory yet."
