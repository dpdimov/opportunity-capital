"""
Future-venture story archetypes (2x2 from the paper's Figure 1).

Each story combines object novelty x description novelty to produce
four archetypes: prosaic, revelatory, metaphorical, fantastic.

Stories provide the venture concept, market context, and key uncertainties
that drive the entrepreneur-investor dialogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FutureVentureStory:
    """A future-venture story archetype with full context."""
    name: str
    archetype: str                      # prosaic / revelatory / metaphorical / fantastic
    object_novelty: str                 # familiar / novel
    description_novelty: str            # familiar / novel
    venture_concept: str                # Rich description of the venture idea
    market_context: str                 # Industry and competitive landscape
    key_uncertainties: Dict[str, str]   # technology, product, market, distribution, organization
    example_reference: str = ""         # Real-world analogy


# ── The Four Archetypes ─────────────────────────────────────────────────────

STORY_ARCHETYPES: Dict[str, FutureVentureStory] = {

    "prosaic": FutureVentureStory(
        name="FreshRoots Kitchen",
        archetype="prosaic",
        object_novelty="familiar",
        description_novelty="familiar",
        venture_concept=(
            "A fast-casual restaurant chain focused on locally-sourced, seasonal menus "
            "in mid-size American cities. Standard franchise model with centralized supply "
            "chain, standardized recipes, and a loyalty app. The concept is well-understood: "
            "quality casual dining at scale."
        ),
        market_context=(
            "The fast-casual segment is mature and competitive. Chipotle, Sweetgreen, and "
            "dozens of regional chains have proven the model. Market growth is steady at 5-7% "
            "annually. Differentiation comes through brand, location strategy, and operational "
            "excellence rather than novel technology or business model."
        ),
        key_uncertainties={
            "technology": "Low -- standard POS, supply chain, and app technology.",
            "product": "Moderate -- menu differentiation and consistent quality across locations.",
            "market": "Low -- well-understood demand in proven segment.",
            "distribution": "Moderate -- site selection and franchise partner quality.",
            "organization": "Moderate -- scaling operations while maintaining food quality.",
        },
        example_reference="Similar to early-stage Sweetgreen or CAVA before rapid expansion.",
    ),

    "revelatory": FutureVentureStory(
        name="SpaceShare",
        archetype="revelatory",
        object_novelty="familiar",
        description_novelty="novel",
        venture_concept=(
            "A platform that transforms underutilized commercial real estate -- empty offices, "
            "idle warehouses, dormant retail -- into on-demand micro-fulfillment centers for "
            "e-commerce brands. The objects are mundane (vacant buildings), but the description "
            "reframes them as distributed logistics infrastructure. Landlords earn yield on dead "
            "space; brands get same-day delivery without building warehouses."
        ),
        market_context=(
            "Commercial vacancy rates are at historic highs post-pandemic. E-commerce brands "
            "struggle with last-mile delivery costs. The insight is connecting two familiar "
            "problems -- empty buildings and expensive delivery -- through a novel framing. "
            "Competitors include traditional 3PLs and Flexe, but none frame it as a real-estate "
            "yield play."
        ),
        key_uncertainties={
            "technology": "Low-moderate -- existing warehouse management + routing software.",
            "product": "High -- the novel framing must convince both landlords and brands.",
            "market": "Moderate -- demand exists but the two-sided market is unproven.",
            "distribution": "High -- must build supply (buildings) and demand (brands) simultaneously.",
            "organization": "Moderate -- operations across heterogeneous spaces is complex.",
        },
        example_reference="Airbnb-like redescription: mundane objects (spare rooms / empty offices) reframed as hospitality / logistics infrastructure.",
    ),

    "metaphorical": FutureVentureStory(
        name="CogniFlow",
        archetype="metaphorical",
        object_novelty="novel",
        description_novelty="familiar",
        venture_concept=(
            "A neuromorphic computing platform that runs AI models on brain-inspired chips, "
            "achieving 100x energy efficiency over GPUs. The technology is genuinely novel -- "
            "spiking neural networks on custom silicon -- but described in familiar terms: "
            "'It's like having a data center in a shoebox' and 'GPU performance at laptop power.' "
            "The metaphors make the exotic accessible."
        ),
        market_context=(
            "AI inference costs are exploding. Every major tech company is seeking alternatives "
            "to GPU-heavy architectures. Intel's Loihi and IBM's TrueNorth have explored "
            "neuromorphic computing but haven't achieved commercial viability. The market pull "
            "is enormous if the technology works -- edge AI, autonomous vehicles, IoT."
        ),
        key_uncertainties={
            "technology": "Very high -- neuromorphic silicon is unproven at commercial scale.",
            "product": "High -- must demonstrate real performance advantages over GPUs.",
            "market": "Low-moderate -- massive demand for efficient AI inference is clear.",
            "distribution": "Moderate -- must integrate with existing AI frameworks (PyTorch, etc.).",
            "organization": "High -- requires rare chip design + AI talent combination.",
        },
        example_reference="Dropbox: novel cloud sync technology described as 'a magic pocket' -- familiar metaphor for unfamiliar capability.",
    ),

    "fantastic": FutureVentureStory(
        name="Orbital Forge",
        archetype="fantastic",
        object_novelty="novel",
        description_novelty="novel",
        venture_concept=(
            "An orbital manufacturing facility that produces ultra-pure semiconductor wafers "
            "and exotic alloys in microgravity. The object is genuinely novel (a factory in "
            "space) and the description is equally novel -- 'gravity-free crystallography,' "
            "'vacuum-native fabrication,' 'orbital supply chain.' Nothing about this maps to "
            "familiar experience. The venture requires the investor to imagine a fundamentally "
            "different manufacturing paradigm."
        ),
        market_context=(
            "Space manufacturing is pre-commercial. Varda Space Industries and Space Forge are "
            "early pioneers. Launch costs have dropped 90% with SpaceX, making orbital "
            "manufacturing economically conceivable for the first time. The semiconductor "
            "industry desperately needs purer wafers for next-gen chips, but no one has proven "
            "the economics of making them in orbit."
        ),
        key_uncertainties={
            "technology": "Extreme -- microgravity manufacturing is experimental.",
            "product": "Very high -- must prove quality advantages justify orbital costs.",
            "market": "High -- potential buyers exist but won't commit without proof.",
            "distribution": "Extreme -- re-entry logistics for finished goods are unsolved.",
            "organization": "Very high -- requires aerospace + semiconductor + manufacturing expertise.",
        },
        example_reference="No close analogy -- this is genuinely fantastic in the paper's sense. Closest: early SpaceX before reusable rockets were proven.",
    ),
}


def get_story(archetype: str) -> FutureVentureStory:
    """Look up a story archetype by name."""
    if archetype not in STORY_ARCHETYPES:
        raise ValueError(
            f"Unknown story archetype '{archetype}'. "
            f"Choose from: {', '.join(STORY_ARCHETYPES.keys())}"
        )
    return STORY_ARCHETYPES[archetype]


def story_to_verbal(story: FutureVentureStory) -> str:
    """Generate a verbal summary of the story for LLM context."""
    lines = [
        f"Venture: {story.name}",
        f"Archetype: {story.archetype} (object={story.object_novelty}, description={story.description_novelty})",
        f"\nConcept: {story.venture_concept}",
        f"\nMarket context: {story.market_context}",
        "\nKey uncertainties:",
    ]
    for dim, desc in story.key_uncertainties.items():
        lines.append(f"  - {dim.capitalize()}: {desc}")
    if story.example_reference:
        lines.append(f"\nAnalogy: {story.example_reference}")
    return "\n".join(lines)
