from __future__ import annotations

from typing import Any


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        for key in ("content", "text", "value", "response", "answer", "output", "completion"):
            if key in value:
                text = _coerce_text(value[key])
                if text:
                    return text
        return ""

    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            text = _coerce_text(item)
            if text:
                pieces.append(text)
        return "\n".join(pieces).strip()

    return ""


def pick_first_text(example: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key not in example:
            continue
        text = _coerce_text(example.get(key))
        if text:
            return text
    return ""


def _normalize_role(role: Any) -> str:
    text = _coerce_text(role).lower()
    if text in {"assistant", "model", "gpt", "bot"}:
        return "assistant"
    if text in {"system", "developer"}:
        return "system"
    return "user"


def extract_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    message_keys = ["messages", "conversation", "conversations", "dialogue", "dialog"]
    raw_messages: Any = None
    for key in message_keys:
        value = example.get(key)
        if isinstance(value, list) and value:
            raw_messages = value
            break

    if raw_messages is None:
        return []

    messages: list[dict[str, str]] = []
    for item in raw_messages:
        if isinstance(item, dict):
            role = _normalize_role(item.get("role", item.get("from", item.get("speaker", "user"))))
            content = _coerce_text(item.get("content", item.get("value", item.get("text", ""))))
        else:
            role = "user"
            content = _coerce_text(item)

        if content:
            messages.append({"role": role, "content": content})

    return messages


def _render_messages_as_prompt(messages: list[dict[str, str]], tokenizer: Any | None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        except Exception:
            pass

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "system":
            label = "System"
        elif role == "assistant":
            label = "Assistant"
        else:
            label = "User"
        parts.append(f"{label}:\n{content}")
    parts.append("Assistant:\n")
    return "\n\n".join(parts)


def format_general_example(
    example: dict[str, Any],
    tokenizer: Any | None = None,
    default_system_prompt: str = "You are a helpful assistant.",
) -> tuple[str, str] | None:
    system_prompt = pick_first_text(example, ["system", "system_prompt"])
    if not system_prompt:
        system_prompt = default_system_prompt

    instruction = pick_first_text(example, ["instruction", "prompt", "question", "input"])
    context = pick_first_text(example, ["context", "input_context"])
    response = pick_first_text(example, ["response", "output", "answer", "completion"])

    if instruction and response:
        user_text = instruction
        if context:
            user_text = f"{instruction}\n\nContext:\n{context}"
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt = _render_messages_as_prompt(prompt_messages, tokenizer=tokenizer)
        return prompt, response

    prompt_text = pick_first_text(example, ["prompt", "input_text", "question", "input"])
    completion_text = pick_first_text(example, ["completion", "response", "output", "answer"])
    if prompt_text and completion_text:
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        prompt = _render_messages_as_prompt(prompt_messages, tokenizer=tokenizer)
        return prompt, completion_text

    messages = extract_messages(example)
    if not messages:
        return None

    last_assistant_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "assistant":
            last_assistant_idx = idx
            break

    if last_assistant_idx <= 0:
        return None

    response = messages[last_assistant_idx]["content"]
    prompt_messages = messages[:last_assistant_idx]
    if not any(m.get("role") == "system" for m in prompt_messages):
        prompt_messages = [{"role": "system", "content": system_prompt}] + prompt_messages

    if not any(m.get("role") == "user" for m in prompt_messages):
        return None

    prompt = _render_messages_as_prompt(prompt_messages, tokenizer=tokenizer)
    return prompt, response


def format_math_example(
    example: dict[str, Any],
    tokenizer: Any | None = None,
) -> tuple[str, str] | None:
    question = pick_first_text(example, ["question", "problem", "prompt", "input", "instruction"])
    answer = pick_first_text(example, ["answer", "solution", "output", "completion", "response"])

    if question and answer:
        prompt_messages = [
            {
                "role": "system",
                "content": "You are a careful mathematician. Solve the problem step by step and finish with a final answer.",
            },
            {"role": "user", "content": f"Problem:\n{question}"},
        ]
        prompt = _render_messages_as_prompt(prompt_messages, tokenizer=tokenizer)
        return prompt, answer

    return format_general_example(
        example,
        tokenizer=tokenizer,
        default_system_prompt="You are a careful mathematician. Solve problems step by step and provide a final answer.",
    )
