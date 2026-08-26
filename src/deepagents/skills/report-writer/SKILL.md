---
name: report-writer
description: Write concise, self-contained Markdown reports when the user requests a report or saved summary.
---

# Report Writer Skill

Use only for explicit report or saved-summary requests, not for every normal answer.

## Output

Save reports under `/reports/<kebab-case-topic>-report.md` with:

- Question
- Approach
- Key findings
- Answer
- Sources or tools used

Keep the report factual, concise, and readable without the chat transcript.
