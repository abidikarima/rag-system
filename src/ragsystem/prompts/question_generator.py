QUESTION_GENERATOR = """
You are creating evaluation questions for a document search engine.

A user will type your question into the search engine. The search engine must
find the passage below among thousands of passages from many different
documents (news, contracts, financial reports, manuals, scientific papers,
administrative forms).

Write ONE question about the passage inside <passage>...</passage>.

REQUIRED DIFFICULTY: {COMPLEXITY}

{COMPLEXITY_RULES}

RULES FOR EVERY QUESTION:
- Short: at most {MAX_WORDS} words, one sentence, ending with "?".
- Sound like a real user: natural and fairly general, not a copy of a sentence.
- The answer MUST be found in the passage. Do not use outside knowledge.
- Specific enough that this passage is clearly the best match: mention the
  topic, organisation, product, person or event the passage is about.
  Use the section headings below to identify the subject if the passage
  alone is not explicit. A question like "What is the total amount?"
  without saying what or whose total is NOT acceptable.
- NEVER refer to the passage itself: do not write "the passage", "the text",
  "the document", "the excerpt", "this article", "according to", "mentioned".
- Do not ask about page numbers, section numbers, headers, footers,
  or formatting.
- Do not ask yes/no questions.

Return ONLY valid JSON, with no Markdown and no explanation:

{{
  "question": "...",
  "answer": "Short answer taken from the passage."
}}

Section headings (context only): {HEADINGS}

<passage>
{PASSAGE}
</passage>
"""


COMPLEXITY_RULES = {
    "easy": (
        "EASY: ask about one fact that is stated explicitly in the passage. "
        "You may reuse the important names, but rephrase the rest of the "
        "sentence."
    ),
    "medium": (
        "MEDIUM: ask about a fact or a reason stated in the passage, but "
        "paraphrase it: use synonyms and different wording instead of the "
        "passage's key phrases. The reader must understand the passage, "
        "not just match words."
    ),
    "hard": (
        "HARD: the answer must combine at least two pieces of information "
        "from the passage, or require a simple inference (comparison, cause "
        "and effect, purpose, condition). Use different vocabulary from the "
        "passage and avoid its exact key phrases, while still naming the "
        "subject so the question is unambiguous."
    ),
}
