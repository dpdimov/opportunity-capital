"""
Capital type definitions, verbal descriptions, and opportunity capital computation.

Implements Bourdieu's capital framework adapted for entrepreneur-investor dialogue:
  - Entrepreneur capital stack (cultural, social, symbolic)
  - Investor capital stack (economic, cultural, social)
  - Opportunity capital (emergent, computed per round from story realism + protagonist credibility)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


# ── Verbal level helpers ────────────────────────────────────────────────────

VERBAL_LEVELS = ("low", "moderate", "strong", "exceptional")

VERBAL_DESCRIPTIONS = {
    "low": "limited",
    "moderate": "reasonable",
    "strong": "impressive",
    "exceptional": "extraordinary",
}


def verbal_desc(level: str) -> str:
    """Return a description word for a verbal level."""
    return VERBAL_DESCRIPTIONS.get(level, level)


# ── Entrepreneur Capital ────────────────────────────────────────────────────

@dataclass
class EntrepreneurCapital:
    """Capital profile of an entrepreneur (all verbal levels)."""
    cultural_capital_embodied: str = "moderate"      # Skills, experience, domain expertise
    cultural_capital_objectified: str = "low"         # Prototypes, IP, prior products
    cultural_capital_institutional: str = "moderate"  # Degrees, accelerator alumni, awards
    social_capital: str = "moderate"                  # Network strength, warm intros
    symbolic_capital: str = "moderate"                # Recognized prestige

    # Rich descriptions for LLM context
    embodied_detail: str = ""
    objectified_detail: str = ""
    institutional_detail: str = ""
    social_detail: str = ""
    symbolic_detail: str = ""

    def to_verbal(self) -> str:
        """Generate verbal summary for LLM consumption (no raw scores)."""
        parts = []
        parts.append(
            f"Domain expertise and skills: {verbal_desc(self.cultural_capital_embodied)}"
            + (f" -- {self.embodied_detail}" if self.embodied_detail else "")
        )
        parts.append(
            f"Tangible assets (prototypes, IP): {verbal_desc(self.cultural_capital_objectified)}"
            + (f" -- {self.objectified_detail}" if self.objectified_detail else "")
        )
        parts.append(
            f"Institutional credentials: {verbal_desc(self.cultural_capital_institutional)}"
            + (f" -- {self.institutional_detail}" if self.institutional_detail else "")
        )
        parts.append(
            f"Network and relationships: {verbal_desc(self.social_capital)}"
            + (f" -- {self.social_detail}" if self.social_detail else "")
        )
        parts.append(
            f"Recognized prestige: {verbal_desc(self.symbolic_capital)}"
            + (f" -- {self.symbolic_detail}" if self.symbolic_detail else "")
        )
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntrepreneurCapital":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Investor Capital ────────────────────────────────────────────────────────

@dataclass
class InvestorCapital:
    """Capital profile of an investor."""
    economic_capital: str = "strong"        # Fund size, dry powder
    cultural_capital: str = "moderate"       # Domain expertise, pattern recognition
    social_capital: str = "moderate"         # Network, deal flow, syndicate

    economic_detail: str = ""
    cultural_detail: str = ""
    social_detail: str = ""

    # Domain specialization (affects how well investor can evaluate certain stories)
    domain_expertise: list = field(default_factory=list)

    def to_verbal(self) -> str:
        """Generate verbal summary for LLM consumption."""
        parts = []
        parts.append(
            f"Investment capacity: {verbal_desc(self.economic_capital)}"
            + (f" -- {self.economic_detail}" if self.economic_detail else "")
        )
        parts.append(
            f"Domain expertise and pattern recognition: {verbal_desc(self.cultural_capital)}"
            + (f" -- {self.cultural_detail}" if self.cultural_detail else "")
        )
        parts.append(
            f"Network and deal flow: {verbal_desc(self.social_capital)}"
            + (f" -- {self.social_detail}" if self.social_detail else "")
        )
        if self.domain_expertise:
            parts.append(f"Specialized in: {', '.join(self.domain_expertise)}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InvestorCapital":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Opportunity Capital ─────────────────────────────────────────────────────

@dataclass
class OpportunityCapital:
    """Emergent symbolic value of the future-venture story, computed per round."""
    story_realism: float = 0.0              # 0-10: how realistic the investor finds the story
    protagonist_credibility: float = 0.0    # 0-10: how credible the entrepreneur appears
    opportunity_capital_score: float = 0.0  # Weighted combination

    def to_dict(self) -> Dict[str, float]:
        return {
            "story_realism": round(self.story_realism, 2),
            "protagonist_credibility": round(self.protagonist_credibility, 2),
            "opportunity_capital_score": round(self.opportunity_capital_score, 2),
        }


def compute_opportunity_capital(
    story_realism: float,
    protagonist_credibility: float,
    realism_weight: float = 0.5,
    credibility_weight: float = 0.5,
) -> OpportunityCapital:
    """
    Compute opportunity capital from its two constituents.

    The score is a weighted combination, but not purely additive --
    both components must be present for high opportunity capital.
    Low credibility dampens even a realistic story, and vice versa.
    """
    # Geometric mean component ensures both must be reasonable
    if story_realism <= 0 or protagonist_credibility <= 0:
        geometric = 0.0
    else:
        geometric = (story_realism * protagonist_credibility) ** 0.5

    # Weighted arithmetic mean
    arithmetic = (
        realism_weight * story_realism
        + credibility_weight * protagonist_credibility
    )

    # Blend: 60% geometric (penalizes imbalance), 40% arithmetic
    score = 0.6 * geometric + 0.4 * arithmetic

    return OpportunityCapital(
        story_realism=story_realism,
        protagonist_credibility=protagonist_credibility,
        opportunity_capital_score=round(score, 2),
    )
