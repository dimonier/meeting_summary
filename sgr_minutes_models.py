from pydantic import BaseModel, Field
from typing import List

class QAItem(BaseModel):
    """Individual question and answer pair using SGR Cycle pattern."""
    question: str = Field(description="The question asked during the meeting")
    answer: str = Field(description="The answer provided to the question")


class ProtocolItem(BaseModel):
    """Individual item in a protocol section using SGR Cycle pattern."""
    content: str = Field(description="The content of this protocol item")


class MinutesResponse(BaseModel):
    """Structured output for one-shot minutes generation with SGR Cycle patterns.

    Fields are ordered to guide the model's reasoning (SGR Cascade + Cycle):
    1) Generate each protocol section as lists of items (SGR Cycle)
    2) Extract Questions & Answers as list of Q&A pairs (SGR Cycle)
    3) Generate meeting title
    """

    meeting_goals: List[ProtocolItem] = Field(
        description="List of meeting goals and objectives",
        min_length=2,
        max_length=5
    )
    established_facts: List[ProtocolItem] = Field(
        description="List of the most significant facts established and confirmed during the meeting",
        max_length=10
    )
    existing_problems: List[ProtocolItem] = Field(
        description="List of the most significant problems and issues identified during the meeting",
        max_length=10
    )
    decisions_made: List[ProtocolItem] = Field(
        description="List of the most significant decisions, agreements and resolutions made during the meeting",
        max_length=10
    )
    tasks_to_execute: List[ProtocolItem] = Field(
        description="List of action items and tasks to be completed with known details such as responsible and deadline",
        max_length=10
    )
    # Q&A extraction using SGR Cycle pattern
    qa_items: List[QAItem] = Field(
        description="A concise list of questions and their corresponding answers from the meeting transcript, focusing solely on relevant discussion topics",
        min_length=3,
        max_length=20
    )
    open_questions: List[ProtocolItem] = Field(
        description="List of open questions that have no confident answers and unresolved issues"
    )
    meeting_title: str = Field(
        description="Short, descriptive meeting title in Russian (up to 40 characters)"
    )
