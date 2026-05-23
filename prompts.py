"""
Prompt generators for each phase of the Dor communication pipeline.

Each function returns (system_prompt, user_prompt) for a single LLM call.
The prompts implement the paper's core mechanism: instruction of imagination
through dialogical calibration.

Four calls per round:
  1. Entrepreneur formalize + utter
  2. Investor decode + imagine
  3. Investor respond
  4. Evaluator alignment assessment
"""

from __future__ import annotations

from sim_state import DialogueState, EntrepreneurProfile, InvestorProfile
from stories import FutureVentureStory, story_to_verbal


# ── 1. Entrepreneur: Formalize + Utter ──────────────────────────────────────

def entrepreneur_formalize_utter(
    state: DialogueState,
    round_number: int,
) -> tuple[str, str]:
    """Generate prompts for the entrepreneur to formalize their venture concept
    and craft a pitch utterance.

    Maps to Dor's pipeline: (1) Formalize inner experience -> message, (2) Utter message -> utterance.
    """
    entrepreneur = state.entrepreneur
    story = state.story
    history = state.dialogue_history_verbal()

    system_prompt = f"""You are {entrepreneur.name}, an entrepreneur pitching your venture to an investor.

YOUR BACKGROUND:
{entrepreneur.to_verbal()}

YOUR VENTURE:
{story_to_verbal(story)}

YOUR TASK:
You are in a pitch dialogue with an investor. Your goal is to help them SEE your future venture --
to construct a vivid, compelling, and realistic image of the future you are building.

You must produce two things:
1. MESSAGE: Your structured understanding of the venture -- what it IS, why it matters, how it works.
   This is the signified content, the concept behind your words.
2. UTTERANCE: Your actual pitch -- the words you choose, the metaphors, the framing, the rhetoric.
   This is how you TELL the story. Choose your words carefully based on what you know about your
   investor and how the conversation has gone so far.

The gap between message and utterance is where your craft as a storyteller lives.
A strong message with a weak utterance fails. A weak message with a brilliant utterance
is eventually exposed.

Respond ONLY with JSON in this format:
{{
  "message": {{
    "venture_essence": "What this venture fundamentally IS (2-3 sentences)",
    "value_proposition": "Why this matters and for whom",
    "mechanism": "How it works -- the core logic",
    "differentiation": "What makes this different from alternatives",
    "future_state": "What the world looks like if this succeeds (vivid description)",
    "key_risks_acknowledged": "What uncertainties you're aware of"
  }},
  "utterance": "Your actual pitch (3-5 paragraphs). Speak naturally as {entrepreneur.name}. Use metaphors, examples, and framing appropriate to your background and the investor you're addressing. This should sound like a real person talking, not a template.",
  "strategic_intent": "What you're trying to accomplish in this round (1 sentence, for your own tracking)"
}}"""

    if round_number == 1:
        user_prompt = f"""This is your FIRST meeting with the investor. You know nothing about them yet except their firm name.

Craft your opening pitch for {story.name}. You need to establish both the venture concept and your own credibility.

Remember: the investor will try to reconstruct your venture in their own imagination. Your words are instructions for their imagination. Choose them to maximize the chance that what they imagine matches what you see."""
    else:
        user_prompt = f"""This is ROUND {round_number} of your dialogue with the investor.

CONVERSATION SO FAR:
{history}

Based on the investor's previous response, adapt your pitch. Address their concerns, answer their questions,
and deepen their understanding of the venture. Each round should add new layers of specificity and realism.

Your goal: bring the investor's imagined future closer to YOUR imagined future."""

    return system_prompt, user_prompt


# ── 2. Investor: Decode + Imagine ───────────────────────────────────────────

def investor_decode_imagine(
    state: DialogueState,
    entrepreneur_utterance: str,
    round_number: int,
) -> tuple[str, str]:
    """Generate prompts for the investor to decode the entrepreneur's utterance
    and reconstruct the venture in their own imagination.

    Maps to Dor's pipeline: (3) Decode utterance -> message*, (4) Imagine -> reconstructed future.
    """
    investor = state.investor
    story = state.story
    history = state.dialogue_history_verbal()

    system_prompt = f"""You are {investor.name}, a venture capital investor at {investor.firm}.

YOUR BACKGROUND:
{investor.to_verbal()}

YOUR TASK:
An entrepreneur is pitching you a venture. You must:
1. DECODE their utterance -- what do you think they're actually saying? Strip away the rhetoric
   and extract the core message. What is the venture? What is the mechanism? What is the bet?
2. IMAGINE the future they're describing -- using YOUR OWN experience, knowledge, and pattern
   recognition. What does this venture look like in your mind? What details do you fill in
   from your own expertise? What feels realistic and what feels like a gap?

Important: You are NOT just passively receiving information. You are actively CONSTRUCTING
an imagined future using the entrepreneur's words as instructions. Your cultural capital
(domain expertise, pattern recognition, industry background) shapes how you imagine this venture.
What you see may differ from what the entrepreneur intends -- and that gap is critical.

Respond ONLY with JSON in this format:
{{
  "decoded_message": {{
    "venture_essence": "What you think this venture IS (in your own words)",
    "value_proposition": "The value you understood",
    "mechanism": "How you think it works",
    "differentiation": "What seems genuinely different (if anything)",
    "gaps_noticed": "What's missing, unclear, or doesn't add up"
  }},
  "imagined_future": "Describe what you SEE when you imagine this venture succeeding. Be specific and vivid. Use your own domain expertise to fill in details. Where do you see brilliance? Where do you see fantasy? (2-4 paragraphs)",
  "reconstruction_confidence": "How confident are you that your mental image matches what the entrepreneur is actually building? What might you be getting wrong?"
}}"""

    user_prompt = f"""ROUND {round_number} of your dialogue.

THE ENTREPRENEUR JUST SAID:
\"{entrepreneur_utterance}\"

{"CONVERSATION HISTORY:" + chr(10) + history if round_number > 1 else "This is your first interaction with this entrepreneur."}

Now decode their message and construct your own imagined version of this venture.
Remember: your imagination is shaped by YOUR expertise in {', '.join(investor.capital.domain_expertise) if investor.capital.domain_expertise else 'your investment domains'}."""

    return system_prompt, user_prompt


# ── 3. Investor: Respond ────────────────────────────────────────────────────

def investor_respond(
    state: DialogueState,
    investor_imagined_future: str,
    decoded_message: dict,
    round_number: int,
) -> tuple[str, str]:
    """Generate prompts for the investor to respond to the entrepreneur.

    Maps to Dor's pipeline: (5) Respond -- questions, challenges, signals of interest.
    """
    investor = state.investor
    entrepreneur = state.entrepreneur
    history = state.dialogue_history_verbal()
    capital_trajectory = state.capital_trajectory_verbal()

    system_prompt = f"""You are {investor.name}, a venture capital investor at {investor.firm}.

YOUR BACKGROUND:
{investor.to_verbal()}

YOUR TASK:
You've listened to an entrepreneur's pitch and constructed your own imagined version of their
venture. Now you must RESPOND. Your response serves multiple functions:
- Signal your level of interest (or lack thereof)
- Probe uncertainties and test the entrepreneur's depth of understanding
- Challenge assumptions that seem unrealistic
- Request specifics that would make the story more concrete
- Share relevant pattern matches from your experience (if any)

You are assessing TWO things simultaneously:
1. THE STORY: Is this venture realistic? Can you imagine it actually working?
2. THE PROTAGONIST: Is this entrepreneur credible? Do they have the capital (skills, network,
   resources, recognition) to execute this story?

Respond ONLY with JSON in this format:
{{
  "response": "Your actual spoken response to the entrepreneur (2-4 paragraphs). Be natural, direct, and specific. Ask real questions. Share genuine reactions. This should sound like {investor.name} talking in a real meeting.",
  "internal_assessment": {{
    "story_realism": "How realistic do you find this venture story? (verbal: unrealistic / questionable / plausible / convincing / highly compelling)",
    "protagonist_credibility": "How credible is this entrepreneur as the person to build this? (verbal: not credible / somewhat credible / credible / very credible / exceptionally credible)",
    "key_concerns": ["List your top 2-3 concerns"],
    "positive_signals": ["List what's working well (if anything)"],
    "commitment_direction": "Are you moving toward investment, away from it, or staying neutral? Why? (1-2 sentences)"
  }}
}}"""

    user_prompt = f"""ROUND {round_number} of your dialogue with {entrepreneur.name}.

WHAT YOU IMAGINED after their pitch:
{investor_imagined_future}

YOUR DECODED VERSION of their venture:
{decoded_message}

{"CONVERSATION HISTORY:" + chr(10) + history if round_number > 1 else "This is your first interaction."}

{"CAPITAL TRAJECTORY:" + chr(10) + capital_trajectory if round_number > 1 else ""}

Now respond to {entrepreneur.name}. Remember: your response is also an instruction --
it tells the entrepreneur how to calibrate their next pitch. What you ask about reveals
what matters to you. What you challenge reveals where the story is weak."""

    return system_prompt, user_prompt


# ── 4. Evaluator: Alignment Assessment ──────────────────────────────────────

def evaluator_alignment(
    entrepreneur_message: dict,
    investor_decoded_message: dict,
    round_number: int,
    story_archetype: str,
) -> tuple[str, str]:
    """Generate prompts for the blind evaluator to assess alignment between
    entrepreneur's intended message and investor's decoded message*.

    Maps to Dor's pipeline: (6) Calibration -- compare message vs message*.
    """
    system_prompt = """You are an objective evaluator assessing communication alignment in an entrepreneur-investor dialogue.

YOUR TASK:
Compare the entrepreneur's INTENDED message (what they meant to communicate) with the
investor's DECODED message (what they understood). Assess how well the imagined futures align.

This is Dor's calibration step: are the two parties converging on a shared imaginary object,
or are they talking past each other?

Consider:
- Semantic alignment: Do they describe the same venture in similar terms?
- Structural alignment: Do they agree on the mechanism, value prop, and differentiation?
- Imaginative alignment: Are they seeing the same future?
- Gap significance: Are the gaps trivial (wording differences) or fundamental (different ventures)?

Respond ONLY with JSON in this format:
{
  "alignment_score": <float 0-10, where 10 = perfect alignment>,
  "alignment_assessment": "Overall assessment of how well the imagined futures match (2-3 sentences)",
  "convergence_areas": ["List areas where entrepreneur and investor clearly agree"],
  "divergence_areas": ["List areas where their imagined futures differ"],
  "gap_significance": "Are the gaps trivial, moderate, or fundamental? Why?",
  "calibration_direction": "Is alignment improving, stable, or deteriorating compared to what you'd expect?",
  "legitimacy_score": <float 0-10, where 10 = investor finds the story completely realistic>,
  "commitment_inclination": <float 0-10, where 10 = investor is ready to commit>
}"""

    user_prompt = f"""ROUND {round_number} | Story archetype: {story_archetype}

ENTREPRENEUR'S INTENDED MESSAGE (what they meant):
{_dict_to_text(entrepreneur_message)}

INVESTOR'S DECODED MESSAGE (what they understood):
{_dict_to_text(investor_decoded_message)}

Assess the alignment between these two mental models of the venture.
Remember: some divergence is natural and even healthy (the investor's expertise adds detail).
The question is whether they're building the SAME imaginary object or DIFFERENT ones."""

    return system_prompt, user_prompt


def _dict_to_text(d: dict) -> str:
    """Convert a dict to readable text for evaluator context."""
    lines = []
    for k, v in d.items():
        if isinstance(v, list):
            v = "; ".join(str(item) for item in v)
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)
