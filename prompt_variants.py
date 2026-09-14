"""Prompt presets and artifact naming for controlled TruthfulQA experiments."""

from pathlib import Path


PROMPT_PRESETS = {
    'concise': 'Answer the question concisely.',
    'short-factual': 'Provide a short factual answer.',
    'most-accurate': 'Give only the most accurate answer.',
}


def format_qa_prompt(prompt_name, question, answer=''):
    return f'{PROMPT_PRESETS[prompt_name]} Q: {question} A:{answer}'


def prompt_artifact_tag(prompt_name):
    """Keep the released concise-prompt paths unchanged."""
    if prompt_name not in PROMPT_PRESETS:
        raise ValueError(f'unknown prompt preset: {prompt_name}')
    return '' if prompt_name == 'concise' else f'_{prompt_name}'


def tagged_path(path, prompt_name):
    path = Path(path)
    return path.with_name(f'{path.stem}{prompt_artifact_tag(prompt_name)}{path.suffix}')
