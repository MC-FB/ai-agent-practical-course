from __future__ import annotations

PARALLEL_PLAN = """
question: "Which person was born earlier?"
nodes:
  - id: q1
    label: "Find first birth date"
    task_type: date_lookup
    operation: answer
    question: "When was Ada Lovelace born?"
    depends_on: []
    sources: [context-0]
    prompt:
      system: "Return JSON only."
      user_template: "Question: {resolved_question}"
    input_map: {}
    output_schema:
      type: object
      required: [answer]
      properties:
        answer:
          type: string
  - id: q2
    label: "Find second birth date"
    task_type: date_lookup
    operation: answer
    question: "When was Alan Turing born?"
    depends_on: []
    sources: [context-1]
    prompt:
      system: "Return JSON only."
      user_template: "Question: {resolved_question}"
    input_map: {}
    output_schema:
      type: object
      required: [answer]
      properties:
        answer:
          type: string
  - id: q3
    label: "Compare dates"
    task_type: comparison
    operation: compare
    question: "Who was born earlier: {q1.answer} or {q2.answer}?"
    depends_on: [q1, q2]
    sources: []
    prompt:
      system: "Return JSON only."
      user_template: |
        Inputs: {dependencies}
        Question: {resolved_question}
    input_map:
      left_date: q1.answer
      right_date: q2.answer
    child_output_policy: "Both child answers are required."
    output_schema:
      type: object
      required: [answer, reasoning]
      properties:
        answer:
          type: string
        reasoning:
          type: string
final_node: q3
"""
