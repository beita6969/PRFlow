"""Reference-free Hotpot answer guidance for the existing Worker response."""

HOTPOT_RESULT_FIELDS = (
    "evidence", "summary", "confidence", "unresolved_issues", "tool_summary", "answer",
)


def hotpot_output_instruction(*, is_output_agent: bool) -> str:
    scope = (
        "Resolve the original question in public_task_context, including every hop or "
        "side of a comparison. A narrower assigned_task must not replace that question. "
        if is_output_agent else
        "Resolve your assigned local responsibility using the public passages and visible packets. "
    )
    return (
        "Return one final JSON object in this key order: evidence, summary, confidence, "
        "unresolved_issues, tool_summary, answer. "
        + scope
        + "First select the relevant passage facts in evidence, with passage titles and brief "
        "supporting quotes; do not invent quotations. In summary, state a concise resolved "
        "conclusion connecting those facts. For a comparison, verify both values, units, "
        "and the requested direction; for a shared property, a broad category is valid if "
        "both passages support it. Follow the precise requested relation and its qualifiers "
        "when choosing among nested locations, dates, or related entities. "
        "Determine the requested answer type from the question: entity, location, title, "
        "date, number, category, list, or yes/no. Write answer LAST, consistent with the "
        "resolved conclusion. If the conclusion corrects an earlier candidate, submit the "
        "corrected value. Use the supporting passage's answer phrase where possible, "
        "without paraphrasing, explanatory suffixes, or alternative guesses. For a count "
        "or rank, return the number alone when the question already specifies its unit. "
        "For a type or genre, return the category name. Keep qualifiers needed to identify "
        "the requested entity or location, and every requested member of a list. Preserve "
        "the number and conjunction of a quoted category or name; do not mechanically "
        "singularize, shorten, or split names. For yes/no questions return yes or no. "
        "Put uncertainty in unresolved_issues and confidence, not in alternative answers. "
    )


def hotpot_submission_instruction() -> str:
    return (
        "Submission contract: The selected output Agent answers the complete original "
        "HotpotQA question using the supplied passages and visible evidence. The question "
        "determines the answer type and granularity. Put the supported answer phrase in "
        "answer, preserving required qualifiers and list members; put the evidence and "
        "resolved conclusion before answer. Intermediate Agents may return local findings."
    )
