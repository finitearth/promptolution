"""Utils for formatting prompts and outputs."""
from typing import List, Union


def extract_from_tag(text: str, start_tag: str, end_tag: str) -> Union[List[str], str]:
    """Extracts content from a string between specified start and end tags."""
    was_list = True
    if isinstance(text, str):
        text = [text]
        was_list = False

    outs = []
    for t in text:
        out = t.split(start_tag)[-1].split(end_tag)[0].strip()
        outs.append(out)
    if was_list:
        return outs
    return outs[0]
