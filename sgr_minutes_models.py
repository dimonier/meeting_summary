from pydantic import BaseModel, Field
from typing import List

class QAItem(BaseModel):
    """Individual question and answer pair using SGR Cycle pattern."""
    question: str = Field(description="The question asked during the meeting")
    answer: str = Field(description="The answer provided to the question")


class ProtocolItem(BaseModel):
    """Individual item in a protocol section using SGR Cycle pattern."""
    content: str = Field(description="The content of this protocol item")


class DecisionRecords(BaseModel):
    """Key decision made during the meeting with full context and consequences."""
    context: str = Field(description="The context or situation that led to the question")
    question: str = Field(description="The core question or issue being addressed.")
    considered_options: List[str] = Field(description="List of options that were considered")
    proposed_decision: str = Field(description="The final decision made. Should exactly match one of `considered_options`. No follow-up actions are allowed in this foeld.")
    status: str = Field(description="Status of the proposed_decision: Предложено, Принято, or Отложено. 'Предложено' means that the discussion or research is not finished yet. 'Принято' means that one of the considered options is approved (proposed_decision matches one of considered_options), and no changes on this topic are expected. 'Отложено' means that the question does not matter at the moment. 'Отклонено' means that none of considered_options are good enough.")
    consequences: List[str] = Field(description="List of consequences of the proposed decision, each prefixed with (+) for positive or (-) for negative")
    rationale: str = Field(description="The reasoning or justification behind proposed decision")


class Assignment(BaseModel):
    """Task assignment with responsible person and deadline."""
    task: str = Field(description="What needs to be done")
    responsible: str = Field(description="Who is responsible for this task")
    deadline: str = Field(description="When this task should be completed")


class MinutesResponse(BaseModel):
    """Structured output for one-shot minutes generation with SGR Cycle patterns.

    Fields are ordered to guide the model's reasoning (SGR Cascade + Cycle):
    1) Generate each protocol section as lists of items (SGR Cycle)
    2) Extract Questions & Answers as list of Q&A pairs (SGR Cycle)
    3) Generate meeting title
    """

    meeting_goals: List[ProtocolItem] = Field(
        description="List of meeting goals and objectives",
        min_length=2
    )
    established_facts: List[ProtocolItem] = Field(
        description="List of the most significant facts established and confirmed during the meeting"
    )
    existing_problems: List[ProtocolItem] = Field(
        description="List of the most significant problems and issues identified during the meeting"
    )
    assignments: List[Assignment] = Field(
        description="List of task assignments with responsible persons and deadlines where specified"
    )
    decision_records: List[DecisionRecords] = Field(
        description="Decision Records - a list of decisions made during the meeting."
    )
    # Q&A extraction using SGR Cycle pattern
    qa_items: List[QAItem] = Field(
        description="A concise list of questions and their corresponding answers from the meeting transcript, focusing solely on relevant discussion topics",
        min_length=3,
        max_length=20
    )
    open_questions: List[ProtocolItem] = Field(
        description="List of open questions that have no confident answers and no decisions or assignments were made on them"
    )
    meeting_title: str = Field(
        description="Short, descriptive meeting title in Russian (up to 60 characters)"
    )
