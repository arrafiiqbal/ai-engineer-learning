from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """A reusable prompt template with variable substitution."""
    name: str
    system: str
    template: str

    def render(self, **kwargs) -> str:
        """Fill in template variables."""
        return self.template.format(**kwargs)


# ── Template 1: Sentiment Analysis ──────────────────────────
sentiment_template = PromptTemplate(
    name="sentiment_analysis",
    system="""You are an expert NLP analyst. 
Always respond in exactly the format requested. 
Never add extra commentary.""",
    template="""
<task>Analyze the sentiment of the input text.</task>

<instructions>
    Return ONLY this exact format:
    SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL]
    REASON: [one sentence max]
    CONFIDENCE: [HIGH/MEDIUM/LOW]
</instructions>

<input>{text}</input>
"""
)


# ── Template 2: Summarizer ───────────────────────────────────
summarizer_template = PromptTemplate(
    name="summarizer",
    system="""You are a concise technical writer. 
Summaries must be clear, accurate, and within the word limit.""",
    template="""
<task>Summarize the following text.</task>

<constraints>
    - Maximum {max_words} words
    - Keep technical terms intact
    - Focus on key insights only
</constraints>

<input>{text}</input>
"""
)


# ── Template 3: Code Reviewer ────────────────────────────────
code_review_template = PromptTemplate(
    name="code_reviewer",
    system="""You are a senior Python engineer doing code review.
Be direct, specific, and constructive.
Focus on bugs, performance, and best practices.""",
    template="""
<task>Review the following Python code.</task>

<instructions>
    Return your review in this exact format:
    BUGS: [list any bugs or None]
    IMPROVEMENTS: [list improvements or None]
    BEST_PRACTICES: [what is done well]
    SCORE: [1-10]
</instructions>

<code>{code}</code>
"""
)


# ── Template 4: Data Analyst ─────────────────────────────────
data_analyst_template = PromptTemplate(
    name="data_analyst",
    system="""You are a senior data analyst explaining insights 
to a non-technical business stakeholder.
Use plain language, avoid jargon, focus on business impact.""",
    template="""
<task>Analyze this data and provide business insights.</task>

<context>{context}</context>

<data>{data}</data>

<instructions>
    Return ONLY this exact format:
    KEY_FINDING: [most important insight in one sentence]
    BUSINESS_IMPACT: [what this means for the business]
    RECOMMENDED_ACTION: [one concrete next step]
</instructions>
"""
)


# ── Template 5: AI Tutor ─────────────────────────────────────
ai_tutor_template = PromptTemplate(
    name="ai_tutor",
    system="""You are an expert AI engineering tutor teaching 
a senior data analyst transitioning into AI engineering.
Relate concepts to data analytics where possible.
Be practical, concise, and example-driven.""",
    template="""
<task>Explain the following AI engineering concept.</task>

<student_background>
    Senior data analyst with Python and SQL experience.
    Currently in Week {week} of an 8-week AI engineering sprint.
</student_background>

<concept>{concept}</concept>

<instructions>
    - Explain in plain English first
    - Give one concrete analogy to data analytics
    - Show a minimal code example if relevant
    - End with one practical takeaway
</instructions>
"""
)