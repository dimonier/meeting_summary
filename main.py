from __future__ import annotations

import os
import re
import logging
import logging.handlers
import argparse
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from meeting_summary_prompts import system_message
from sgr_minutes_models import MinutesResponse
from glossary import Glossary, load_glossary, merge_glossaries

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
def _setup_file_logging() -> None:
    """Configure daily rotating file logging and keep 30 days."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "meeting_summary.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path, when="midnight", backupCount=30, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

_setup_file_logging()

# Load environment variables from .env file
load_dotenv()

# Use environment variable to control storage type, default to JSON storage
API_BASE_CLOUD = os.getenv("API_BASE_CLOUD")
API_KEY_CLOUD = os.getenv("API_KEY_CLOUD")
MODEL_ID_CLOUD = os.getenv("MODEL_CLOUD")

API_BASE_LOCAL = os.getenv("API_BASE_LOCAL")
API_KEY_LOCAL = os.getenv("API_KEY_LOCAL")
MODEL_ID_LOCAL = os.getenv("MODEL_LOCAL")

def get_meeting_date_regex(file_name: str) -> tuple[datetime.date, str]:
    """
    Extract meeting date from filename using regex patterns.
    Supports YYYY-MM-DD and YYYY-MMDD formats.
    Returns: (date_object, formatted_string_for_protocol)
    """
    filename = os.path.basename(file_name)
    logger.info(f"Extracting date from filename: {filename}")

    # Map weekday numbers to Russian day names
    weekdays_ru = {
        0: "понедельник",
        1: "вторник",
        2: "среда",
        3: "четверг",
        4: "пятница",
        5: "суббота",
        6: "воскресенье"
    }

    # Try to match YYYY-MM-DD or YYYY-MMDD patterns
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})|(\d{4}-\d{4})", filename)

    meeting_date = None
    if date_match:
        if date_match.group(1):  # Matched YYYY-MM-DD
            meeting_date_str = date_match.group(1)
            meeting_date = datetime.strptime(meeting_date_str, "%Y-%m-%d").date()
        elif date_match.group(2):  # Matched YYYY-MMDD
            meeting_date_str = date_match.group(2)
            # Convert YYYY-MMDD to YYYY-MM-DD for parsing
            meeting_date_str = meeting_date_str[:4] + '-' + meeting_date_str[5:7] + '-' + meeting_date_str[7:]
            meeting_date = datetime.strptime(meeting_date_str, "%Y-%m-%d").date()

    if meeting_date:
        logger.info(f"Extracted date: {meeting_date}")
    else:
        today = datetime.now().date()
        meeting_date = today
        logger.info(f"No date found in filename. Using today's date: {meeting_date}")

    day_of_week_num = meeting_date.weekday()
    day_of_week_name = weekdays_ru[day_of_week_num]

    final_answer_str = f"Дата совещания: {day_of_week_name} {meeting_date.strftime('%Y-%m-%d')}"
    return meeting_date, final_answer_str

def load_transcript(file_name: str) -> str:
    with open(file_name, encoding="utf-8") as file:
        transcript = file.read()
    return transcript


def get_summary_file_name(
    file_name: str,
    model_id: str,
    suffix: str = "",
    details: bool = False,
    elapsed_seconds: int | None = None,
    meeting_date: datetime.date | None = None,
    keep_base_name: bool = False,
) -> str:
    """Build final summary file path.

    Includes base transcript name, optional LLM-provided suffix, model id, and
    optionally appends elapsed seconds in square brackets before the extension.
    """
    file_dir = os.path.dirname(file_name)
    file_base_name = os.path.basename(file_name)
    file_base_name_without_ext = os.path.splitext(file_base_name)[0]
    safe_model_name = model_id.split("/")[-1]

    # If meeting_date is provided, use date prefix instead of original filename
    if meeting_date is not None:
        date_prefix = meeting_date.strftime("%Y-%m-%d")
    else:
        date_prefix = datetime.now().strftime("%Y-%m-%d")

    base = file_base_name_without_ext if keep_base_name else date_prefix

    base = f"{base}-{suffix}" if suffix else base

    if details:
        base = f"{base}-{safe_model_name}"
        if elapsed_seconds is not None:
            base = f"{base}[{elapsed_seconds}s]"

    output_file_name = f"{base}.md"

    return os.path.join(file_dir, output_file_name) if file_dir else output_file_name


def _build_system_message_with_glossary(glossary: Optional[Glossary]) -> str:
    """Build final system message, optionally enriched with glossary rules."""
    if glossary is None or glossary.is_empty():
        return system_message

    glossary_block = glossary.to_prompt_block()
    if not glossary_block:
        return system_message

    return f"{system_message}\n\n{glossary_block}"


def get_minutes_one_call(
    file_name: str,
    meeting_date: str,
    client: OpenAI,
    model_id: str,
    temperature: float,
    glossary: Optional[Glossary] = None,
) -> MinutesResponse:
    """Run a single SGR-guided call that returns all minutes artifacts with detailed protocol sections.

    The model is guided to fill the schema fields in a cascade order:
    topics -> protocol sections -> Q&A -> filename suffix.
    """
    transcript = load_transcript(file_name)

    user_message = (
        f"Дата встречи: {meeting_date}\n\n"
        "Транскрипт встречи:\n\n"
        f"{transcript}"
    )

    try:
        completion = client.chat.completions.parse(
            model=model_id,
            response_format=MinutesResponse,
            messages=[
                {
                    "role": "system",
                    "content": _build_system_message_with_glossary(glossary),
                },
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=60000,
            timeout=3600,
        )

        if not completion.choices or not completion.choices[0].message:
            raise ValueError("Empty LLM response")

        minutes: MinutesResponse | None = getattr(completion.choices[0].message, "parsed", None)
        if minutes is None:
            raise ValueError("Parsed MinutesResponse is missing in the response")

        return minutes

    except Exception as e:
        logger.error(f"Error in SGR parsing: {e}")
        raise

def make_filesystem_safe_suffix(suffix: str) -> str:
    # Replace spaces with dashes and remove invalid characters
    safe_suffix = re.sub(r'[^\w\-_а-яёА-ЯЁ]', '', suffix.replace(' ', '-'))
    # Remove multiple consecutive dashes
    safe_suffix = re.sub(r'-+', '-', safe_suffix)
    # Remove leading/trailing dashes
    safe_suffix = safe_suffix.strip('-')
    # Limit length to 30 characters
    safe_suffix = safe_suffix[:60]
    return safe_suffix


def main():
    """Main function to execute SGR-based meeting minutes generation.

    Uses a single LLM call with detailed protocol sections to generate all parts of the protocol:
    - Individual protocol sections (goals, facts, problems, decisions, tasks, questions)
    - Questions & Answers table
    - Filename suffix
    """

    parser = argparse.ArgumentParser(description="Summarize meeting transcript using Schema Guided Reasoning technique.")
    parser.add_argument("transcript_file", help="Path to the meeting transcript file.")
    parser.add_argument("-m", "--model", help="Model to use for summarization. If not specified, uses default model for selected provider.")

    parser.add_argument("--online", action="store_true", help="Use online cloud LLM provider instead of local.")
    parser.add_argument("-t", "--temperature", type=float, default=0.15, help="Sampling temperature for the LLM (default: 0.15).")
    parser.add_argument("--details", action="store_true", help="Add model name and generation time to output filename (default: False).")
    parser.add_argument("--keep_base_name", action="store_true", help="Keep original transcript filename as base (default: use date prefix).")
    parser.add_argument(
        "--extra-glossary",
        dest="extra_glossary",
        help="Path to an additional glossary YAML file that overrides base entries on conflicts.",
    )
    parser.add_argument(
        "--no-glossary-cleanup",
        dest="no_glossary_cleanup",
        action="store_true",
        help="Disable automatic post-processing cleanup of incorrect term variants in parentheses (default: cleanup enabled).",
    )

    args = parser.parse_args()
    transcript_file_name = args.transcript_file
    use_online = args.online
    keep_base_name = args.keep_base_name if args.keep_base_name else False
    extra_glossary_path: Optional[str] = args.extra_glossary

    # Determine which provider configuration to use
    if use_online:
        api_base = API_BASE_CLOUD
        api_key = API_KEY_CLOUD
        default_model = MODEL_ID_CLOUD
        provider_type = "online cloud"
    else:
        api_base = API_BASE_LOCAL
        api_key = API_KEY_LOCAL
        default_model = MODEL_ID_LOCAL
        provider_type = "local"

    # Use specified model or default for selected provider
    model_id = args.model if args.model else default_model

    logger.info("Using %s provider with model: %s", provider_type, model_id)

    openai_client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )

    # Get meeting date using algorithmic extraction
    meeting_date_obj, meeting_date_str = get_meeting_date_regex(transcript_file_name)

    logger.info("Processing meeting from %s", meeting_date_str)

    # Load base glossary.yaml if present, and optional extra glossary if provided
    glossary: Optional[Glossary] = None
    base_glossary_path = "glossary.yaml"

    try:
        if os.path.exists(base_glossary_path):
            base_glossary = load_glossary(base_glossary_path)
        else:
            base_glossary = None

        extra_glossary: Optional[Glossary] = None
        if extra_glossary_path:
            extra_glossary = load_glossary(extra_glossary_path)

        if base_glossary and extra_glossary:
            glossary = merge_glossaries(base_glossary, extra_glossary)
        else:
            glossary = extra_glossary or base_glossary

    except FileNotFoundError as e:
        # Extra glossary path is wrong – fail fast with clear log
        logger.error("Glossary file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to load glossary files: %s", e)
        raise

    generation_start_time = datetime.now()
    minutes = get_minutes_one_call(
        file_name=transcript_file_name,
        meeting_date=meeting_date_str,
        client=openai_client,
        model_id=model_id,
        temperature=args.temperature,
        glossary=glossary,
    )
    generation_end_time = datetime.now()
    generation_seconds = (generation_end_time - generation_start_time).total_seconds()

    meeting_title_filename_suffix = make_filesystem_safe_suffix(minutes.meeting_title)
    logger.info("Got response, meeting title: %s", minutes.meeting_title)

    # Assemble full protocol from individual sections using Cycle pattern data
    def format_section_items(items):
        """Format a list of protocol items into a numbered list."""
        if not items:
            return "Отсутствуют"

        numbered_items = []
        for i, item in enumerate(items, 1):
            content = item.content if hasattr(item, 'content') else str(item)
            numbered_items.append(f"{i}. {content}")

        return "\n".join(numbered_items)

    def format_tasks_items(items):
        """Format tasks but hide unset fields like 'Не указано' for deadline/responsible."""
        if not items:
            return "Отсутствуют"

        def _clean_unset_fields(text: str) -> str:
            # Remove patterns like 'Ответственный: Не указано', 'Ответственные: Не указано'
            text = re.sub(r"(,\s*)?Ответственн(?:ый|ые):\s*Не\s*указано\.?", "", text, flags=re.IGNORECASE)
            # Remove patterns like 'Срок: Не указано', 'Сроки: Не указано', 'Дедлайн: Не указано'
            text = re.sub(r"(,\s*)?(?:Сроки?|Дедлайн):?\s*Не\s*указано\.?", "", text, flags=re.IGNORECASE)
            # Normalize extra spaces and trailing punctuation left after removals
            text = re.sub(r"\s{2,}", " ", text).strip()
            text = re.sub(r"[\s,;:\-]+$", "", text)
            return text

        numbered_items = []
        for i, item in enumerate(items, 1):
            content = item.content if hasattr(item, 'content') else str(item)
            content = _clean_unset_fields(content)
            numbered_items.append(f"{i}. {content}")

        return "\n".join(numbered_items)

    def format_key_decisions(decisions):
        """Format key decisions as a markdown table with 8 columns."""
        if not decisions:
            return "Отсутствуют"

        # Table header
        table = "| № | Контекст | Вопрос | Рассмотренные варианты | Решение | Статус | Последствия | Обоснование |\n"
        table += "|---|----------|--------|------------------------|---------|--------|-------------|-------------|\n"

        for i, decision in enumerate(decisions, 1):
            # Format considered options as bullet list
            options_text = "<br>".join([f"- {opt}" for opt in decision.considered_options])
            
            # Format consequences with line breaks
            consequences_text = "<br>".join(decision.consequences)
            
            # Escape pipe characters in cell content
            context = decision.context.replace("|", "\\|")
            question = decision.question.replace("|", "\\|")
            decision_text = decision.proposed_decision.replace("|", "\\|")
            status = decision.status.replace("|", "\\|")
            rationale = decision.rationale.replace("|", "\\|")
            
            table += f"| {i} | {context} | {question} | {options_text} | {decision_text} | {status} | {consequences_text} | {rationale} |\n"

        return table

    def format_assignments(assignments):
        """Format assignments as a markdown table with 4 columns."""
        if not assignments:
            return "Отсутствуют"

        # Table header
        table = "| № | Что сделать | Ответственный | Срок |\n"
        table += "|---|-------------|---------------|------|\n"

        for i, assignment in enumerate(assignments, 1):
            # Escape pipe characters in cell content
            task = assignment.task.replace("|", "\\|")
            responsible = assignment.responsible.replace("|", "\\|")
            deadline = assignment.deadline.replace("|", "\\|")
            
            table += f"| {i} | {task} | {responsible} | {deadline} |\n"

        return table

    # Build Q&A section to embed into the summary
    qa_markdown = "| № | Вопрос | Ответ |\n"
    qa_markdown += "|----|---------|--------|\n"
    for i, qa_item in enumerate(minutes.qa_items, 1):
        qa_markdown += f"| {i} | {qa_item.question} | {qa_item.answer} |\n"

    full_protocol = f"""# Протокол совещания: {minutes.meeting_title}

{meeting_date_str}

### Цели встречи

{format_section_items(minutes.meeting_goals)}

### Установленные факты

{format_section_items(minutes.established_facts)}

### Имеющиеся проблемы

{format_section_items(minutes.existing_problems)}

### Ключевые решения

{format_key_decisions(minutes.decision_records)}

### Поручения

{format_assignments(minutes.assignments)}

### Вопросы и ответы

{qa_markdown}

### Открытые вопросы

{format_section_items(minutes.open_questions)}
"""

    # Apply glossary-based cleanup to remove incorrect variants in parentheses
    if glossary and not glossary.is_empty() and not args.no_glossary_cleanup:
        full_protocol = glossary.clean_protocol_text(full_protocol)
        logger.info("Applied glossary cleanup to protocol text")

    # Write full protocol with meeting date header
    elapsed_seconds_whole = int(generation_seconds)
    summary_file_name_with_time = get_summary_file_name(
        file_name=transcript_file_name,
        model_id=model_id,
        suffix=meeting_title_filename_suffix,
        details=args.details,
        elapsed_seconds=elapsed_seconds_whole,
        meeting_date=meeting_date_obj if not args.keep_base_name else None,
        keep_base_name=keep_base_name,
    )
    with open(summary_file_name_with_time, "w", encoding="utf-8") as file:
        file.write(full_protocol)

    # Q&A is now embedded into the summary; no separate Q&A file is written
    logger.info(
        "Summary saved to file: %s; generation time: %.2f seconds",
        summary_file_name_with_time,
        generation_seconds,
    )

if __name__ == "__main__":
    main()
