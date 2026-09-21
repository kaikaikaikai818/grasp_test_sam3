"""English operator names for the five trial tools; unknown names pass through."""

PROMPTS = {
    "screwdriver": "a screwdriver",
    "adjustable wrench": "an adjustable wrench",
    "tape measure": "a tape measure",
    "tape dispenser": "a tape dispenser",
    "rubber mallet": "a rubber mallet",
}


def normalize_prompt(text: str) -> str:
    name = " ".join(text.strip().lower().split())
    if not name:
        raise ValueError("English tool name cannot be empty")
    for category, prompt in PROMPTS.items():
        if name in (category, prompt):
            return prompt
    return name


def known_category(prompt: str):
    canonical = normalize_prompt(prompt)
    return next((name for name, value in PROMPTS.items() if value == canonical), None)
