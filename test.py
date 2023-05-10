import openai
openai.api_key = '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'

# First, let's define a simple dataset consisting of words and their antonyms.
words = ["sane", "direct", "informally", "unpopular", "subtractive", "nonresidential",
    "inexact", "uptown", "incomparable", "powerful", "gaseous", "evenly", "formality",
    "deliberately", "off"]
antonyms = ["insane", "indirect", "formally", "popular", "additive", "residential",
    "exact", "downtown", "comparable", "powerless", "solid", "unevenly", "informality",
    "accidentally", "on"]

# First, let's define a simple dataset consisting of words and their antonyms.
hints = ["sane", "direct", "informally", "unpopular", "subtractive", "nonresidential",
    "inexact", "uptown", "incomparable", "powerful", "gaseous", "evenly", "formality",
    "deliberately", "off"]
reviews = ["insane", "indirect", "formally", "popular", "additive", "residential",
    "exact", "downtown", "comparable", "powerless", "solid", "unevenly", "informality",
    "accidentally", "on"]

# Now, we need to define the format of the prompt that we are using.

eval_template = \
"""Instruction: [PROMPT]
Input: [INPUT]
Output: [OUTPUT]"""

# Now, let's use APE to find prompts that generate antonyms for each word.
from automatic_prompt_engineer import ape

result, demo_fn = ape.simple_ape(
    dataset=(words, antonyms),
    eval_template=eval_template,
)

print(result)

from automatic_prompt_engineer import ape

manual_prompt = "Write an antonym to the following word."

human_result = ape.simple_eval(
    dataset=(words, antonyms),
    eval_template=eval_template,
    prompts=[manual_prompt],
)

print(human_result)