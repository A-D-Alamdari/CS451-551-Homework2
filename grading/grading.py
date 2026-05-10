"""
grading/grading.py -- grade-book for the Mars Rover autograder (3.10+).

Part of the "Mars Rover: Red Planet Rescue" grading framework.

:class:`Grades` accumulates points and messages as the autograder
works through questions and test cases, then formats the final
report with a per-question table and a total.

Typical usage from the autograder::

    grades = Grades('Mars Rover: Red Planet Rescue', [
        ('Q1', 18),
        ('Q2', 18),
        ...
    ])
    grades.grade(module_dict, exception_map)
    grades.produce_output()
"""

from typing import Any, Callable


class Grades:
    """Accumulator for autograder points and messages.

    Parameters
    ----------
    project_name : str
        Human-readable project title printed in the report header.
    questions_and_maxes : list[tuple[str, int]]
        Ordered list of ``(question_name, max_points)`` pairs that
        defines the full grading rubric. Each entry becomes a row
        in the final output table.
    """

    def __init__(
        self,
        project_name: str,
        questions_and_maxes: list[tuple[str, int]],
    ):
        self.project_name: str = project_name
        self.questions: list[str] = [q for q, _ in questions_and_maxes]
        self.maxes: dict[str, int] = dict(questions_and_maxes)
        self.points: dict[str, int] = {q: 0 for q in self.questions}
        self.messages: dict[str, list[str]] = {q: [] for q in self.questions}
        self.current_question: str | None = None
        self._prerequisites: dict[str, set[str]] = {q: set() for q in self.questions}

    # ------------------------------------------------------------------
    # Context: which question is currently being graded
    # ------------------------------------------------------------------

    def start_question(self, name: str) -> None:
        """Set the active question so subsequent calls route there."""
        if name not in self.maxes:
            raise KeyError(f'Unknown question {name!r}')
        self.current_question = name
        self.points[name] = 0

    # ------------------------------------------------------------------
    # Point manipulation
    # ------------------------------------------------------------------

    def add_points(self, points: int) -> None:
        """Add ``points`` to the current question's tally."""
        q = self._require_current()
        self.points[q] = min(self.points[q] + points, self.maxes[q])

    def deduct_points(self, points: int) -> None:
        """Remove ``points`` from the current question's tally (floor 0)."""
        q = self._require_current()
        self.points[q] = max(self.points[q] - points, 0)

    def assign_full_credit(self, points: int | None = None) -> None:
        """Set the current question to full credit (or ``points`` if given)."""
        q = self._require_current()
        self.points[q] = points if points is not None else self.maxes[q]

    def fail(self, message: str = '') -> None:
        """Zero out the current question and record a failure message."""
        q = self._require_current()
        self.points[q] = 0
        if message:
            self.add_message(message)

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(self, message: str) -> None:
        """Append a message to the current question's log."""
        q = self._require_current()
        self.messages[q].append(message)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_total(self) -> tuple[int, int]:
        """Return ``(total_earned, total_possible)``."""
        earned = sum(self.points[q] for q in self.questions)
        possible = sum(self.maxes[q] for q in self.questions)
        return earned, possible

    # ------------------------------------------------------------------
    # Grading driver
    # ------------------------------------------------------------------

    def grade(
        self,
        grade_fn: Callable[['Grades'], None] | None = None,
        exception_map: dict[str, type] | None = None,
    ) -> None:
        """Run ``grade_fn(self)`` and catch any unexpected exceptions.

        If ``grade_fn`` is ``None`` (the typical case when the
        autograder manages its own loop), this method is a no-op.

        ``exception_map`` can map question names to exception types
        that should be caught and converted to failures rather than
        propagated. This is useful for isolating student crashes so
        they only zero the question they belong to, not the whole
        grading run.
        """
        if grade_fn is None:
            return
        if exception_map is None:
            exception_map = {}
        try:
            grade_fn(self)
        except Exception as exc:
            q = self.current_question
            if q and type(exc) in exception_map.values():
                self.fail(f'{type(exc).__name__}: {exc}')
            else:
                raise

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def produce_output(self) -> str:
        """Format and return the full grading report as a string.

        Also prints the report to stdout. The report includes a
        header, per-question messages and scores, a separator, and
        a total line with a percentage.
        """
        lines: list[str] = []
        width = 56

        lines.append('=' * width)
        lines.append(f'  {self.project_name} -- Autograder Results')
        lines.append('-' * width)

        for q in self.questions:
            earned = self.points[q]
            possible = self.maxes[q]
            lines.append(f'  {q}: {earned}/{possible}')
            for msg in self.messages[q]:
                lines.append(f'    {msg}')

        lines.append('-' * width)
        total_earned, total_possible = self.get_total()
        pct = 100.0 * total_earned / total_possible if total_possible else 0.0
        lines.append(
            f'  TOTAL: {total_earned}/{total_possible}  ({pct:.1f}%)'
        )
        lines.append('=' * width)

        report = '\n'.join(lines)
        print(report)
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_current(self) -> str:
        if self.current_question is None:
            raise RuntimeError(
                'No question is active -- call start_question() first'
            )
        return self.current_question
