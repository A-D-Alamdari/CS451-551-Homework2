"""
grading/test_parser.py -- Berkeley-style .test / .solution parser (3.10+).

Both file formats are very similar: a line-oriented "key: value"
format where values can be either quoted strings (single line) or
triple-quoted multi-line blocks.

Example ``.test`` file::

    class: "ValueIterationTest"
    testGrid: "base_camp"
    iterations: "100"
    discount: "0.9"

    success: \"\"\"Value iteration agrees with reference on base_camp
    within the tolerance 1e-3.\"\"\"
    failure: \"\"\"Value iteration disagrees with reference.\"\"\"

Public functions:

* :func:`parse_test`     -- parse a ``.test`` file into a dict.
* :func:`parse_solution` -- parse a ``.solution`` file into a dict.
* :func:`parse_file`     -- the shared low-level parser both call.

Whitespace handling: trailing whitespace is stripped from every
line; CRLF line endings are normalised to LF so the parser works
identically on Windows-authored and Unix-authored test files.
Blank lines and lines starting with ``#`` at the top level are
ignored. Inside a triple-quoted block every character is preserved
verbatim (except the final ``\\r`` of a CRLF line ending).
"""

import os
import re
from typing import Any


# A line is either:
#   key: "value on one line"
#   key: \"\"\" ... the start of a multi-line block
_LINE_RE = re.compile(
    r'^\s*'
    r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)'
    r'\s*:\s*'
    r'(?P<rest>.*)$'
)


def parse_file(path: str) -> dict[str, Any]:
    """Parse a Berkeley-style key/value file into a plain ``dict``.

    Raises :class:`FileNotFoundError` if the path does not exist,
    :class:`ValueError` on a malformed multi-line block (unterminated
    triple-quoted string) or any other syntactic surprise that
    would otherwise produce a silent bad parse.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, 'r', encoding='utf-8', newline='') as f:
        raw = f.read()

    # Normalise CRLF / CR line endings to LF so parsing is
    # platform-independent.
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    lines = raw.split('\n')

    result: dict[str, Any] = {}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Skip blank lines and top-level comments.
        if not stripped or stripped.startswith('#'):
            i += 1
            continue

        match = _LINE_RE.match(line)
        if not match:
            raise ValueError(
                f'{path}:{i + 1}: cannot parse line: {line!r}'
            )

        key = match.group('key')
        rest = match.group('rest').rstrip()

        # Case 1: single-line value, wrapped in double quotes.
        if rest.startswith('"') and not rest.startswith('"""'):
            if not (rest.endswith('"') and len(rest) >= 2):
                raise ValueError(
                    f'{path}:{i + 1}: unterminated single-line value for {key!r}'
                )
            result[key] = rest[1:-1]
            i += 1
            continue

        # Case 2: single-line triple-quoted value that closes on the
        # same line (e.g. key: """short""").
        if rest.startswith('"""') and rest.endswith('"""') and len(rest) >= 6:
            result[key] = rest[3:-3]
            i += 1
            continue

        # Case 3: multi-line triple-quoted block.
        if rest.startswith('"""'):
            body_parts: list[str] = []
            # Content on the opening line after the triple quote.
            first = rest[3:]
            i += 1
            # Scan forward for the closing triple quote.
            closed = False
            while i < n:
                current = lines[i]
                if '"""' in current:
                    # Closing line; the part before the """ is the
                    # last line of the body.
                    end = current.index('"""')
                    tail = current[:end]
                    if first:
                        body_parts.append(first)
                        first = ''
                    body_parts.append(tail)
                    i += 1
                    closed = True
                    break
                else:
                    if first:
                        body_parts.append(first)
                        first = ''
                    body_parts.append(current)
                    i += 1
            if not closed:
                raise ValueError(
                    f'{path}: unterminated triple-quoted block for {key!r}'
                )
            result[key] = '\n'.join(body_parts)
            continue

        # Case 4: bare value without any quoting. Berkeley test files
        # sometimes use this form for numeric or boolean settings.
        result[key] = rest
        i += 1

    return result


def parse_test(path: str) -> dict[str, Any]:
    """Parse a ``.test`` file. Thin wrapper around :func:`parse_file`."""
    return parse_file(path)


def parse_solution(path: str) -> dict[str, Any]:
    """Parse a ``.solution`` file. Thin wrapper around :func:`parse_file`."""
    return parse_file(path)
