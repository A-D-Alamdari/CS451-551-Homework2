"""
grading/test_classes.py -- base classes for test questions and cases (3.10+).

Part of the "Mars Rover: Red Planet Rescue" grading framework.

Provides the abstract scaffolding for the autograder: a *Question*
groups one or more :class:`TestCase` instances, decides how to
translate individual pass/fail outcomes into points, and reports
results to a :class:`grading.grading.Grades` object.

Two concrete question strategies ship here:

* :class:`PassAllTestsQuestion` -- all test cases in the question
  must pass for the student to earn any credit. One failure zeroes
  the whole question. This is the strictest mode and is suitable
  for questions where partial correctness is meaningless.
* :class:`NumberPassedQuestion` -- each passing test case earns a
  proportional share of the question's max points (rounded down
  to the nearest integer). This is gentler and gives students
  incremental feedback.
"""

from typing import Any


class TestCase:
    """A single test within a question.

    Subclasses must override :meth:`execute` (run the test, call
    ``self.test_pass`` or ``self.test_fail`` on the grades object)
    and optionally :meth:`write_solution` (emit a ``.solution``
    file for instructor use).

    Parameters
    ----------
    question : Question
        The parent question this test belongs to.
    test_dict : dict[str, Any]
        Key/value pairs parsed from the ``.test`` file by
        :func:`grading.test_parser.parse_test`.
    """

    def __init__(self, question: 'Question', test_dict: dict[str, Any]):
        self.question = question
        self.test_dict = test_dict
        self.path = test_dict.get('path', '')
        self.name = test_dict.get('name', self.path)

    def execute(self, grades: Any, module_dict: dict, solution_dict: dict) -> None:
        """Run the test. Must call ``self.test_pass`` or ``self.test_fail``.

        Parameters
        ----------
        grades : grading.grading.Grades
            The grade-book to report pass/fail into.
        module_dict : dict
            Name -> module mapping of the student's loaded code.
        solution_dict : dict
            Key/value pairs from the corresponding ``.solution``
            file (may be empty if no solution file exists).
        """
        raise NotImplementedError

    def write_solution(self, module_dict: dict, file_path: str) -> None:
        """Write the reference solution for this test to ``file_path``.

        Used only by the instructor workflow that regenerates
        ``.solution`` files; the default implementation is a no-op.
        Subclasses that need solution files override it.
        """
        pass

    # ------------------------------------------------------------------
    # Helpers called by subclass execute() implementations
    # ------------------------------------------------------------------

    def test_pass(self, grades: Any) -> None:
        """Signal that this test case passed."""
        grades.add_message(f'  \u2705 {self.name}')
        return True

    def test_fail(self, grades: Any, message: str = '') -> None:
        """Signal that this test case failed."""
        msg = f'  \u274C {self.name}'
        if message:
            msg += f' -- {message}'
        grades.add_message(msg)
        return False


class Question:
    """A graded question that owns one or more :class:`TestCase` objects.

    Parameters
    ----------
    max_points : int
        The maximum number of points this question is worth.
    display_name : str
        Human-readable label for the question (used in the report).
    """

    def __init__(self, max_points: int, display_name: str = ''):
        self.max_points: int = max_points
        self.display_name: str = display_name
        self.test_cases: list[TestCase] = []
        self.points_earned: int = 0

    def add_test_case(self, test_case: TestCase) -> None:
        """Append a test case to this question's list."""
        self.test_cases.append(test_case)

    def execute(self, grades: Any, module_dict: dict, solution_dicts: list[dict]) -> None:
        """Run every test case and assign points.

        Subclasses override this to implement the specific grading
        strategy (all-or-nothing vs. proportional).

        Parameters
        ----------
        grades : grading.grading.Grades
            Grade-book for reporting.
        module_dict : dict
            Student modules.
        solution_dicts : list[dict]
            One solution dict per test case, in the same order as
            ``self.test_cases``.
        """
        raise NotImplementedError

    def _run_tests(
        self, grades: Any, module_dict: dict, solution_dicts: list[dict]
    ) -> tuple[int, int]:
        """Run all test cases, return ``(passed, total)``."""
        import io, sys
        passed = 0
        total = len(self.test_cases)
        for i, tc in enumerate(self.test_cases):
            sol = solution_dicts[i] if i < len(solution_dicts) else {}
            # Capture stderr so util.raise_not_defined's print does
            # not interleave with the autograder's formatted output.
            old_err = sys.stderr
            sys.stderr = io.StringIO()
            try:
                result = tc.execute(grades, module_dict, sol)
                if result is True:
                    passed += 1
            except NotImplementedError as e:
                msg = str(e) if str(e) else ''
                if 'Method not implemented:' in msg:
                    detail = 'stub: ' + msg.replace(
                        '*** Method not implemented: ', '')
                else:
                    detail = 'student stub not implemented'
                tc.test_fail(grades, detail)
            except Exception as exc:
                tc.test_fail(grades, f'{type(exc).__name__}: {exc}')
            finally:
                sys.stderr = old_err
        return passed, total


class PassAllTestsQuestion(Question):
    """All test cases must pass for the student to earn any credit.

    If every test passes the student receives ``max_points``; if
    even one fails the score is 0.
    """

    def execute(self, grades: Any, module_dict: dict, solution_dicts: list[dict]) -> None:
        grades.add_message(f'{self.display_name}')
        passed, total = self._run_tests(grades, module_dict, solution_dicts)
        if passed == total:
            self.points_earned = self.max_points
            grades.assign_full_credit(self.max_points)
        else:
            self.points_earned = 0
            grades.add_message(
                f'  ({passed}/{total} tests passed -- need all for credit)'
            )


class NumberPassedQuestion(Question):
    """Each passing test earns a proportional share of max_points.

    ``points_earned = max_points * passed // total`` (integer
    division) so that the score is always an integer and never
    exceeds ``max_points``.
    """

    def execute(self, grades: Any, module_dict: dict, solution_dicts: list[dict]) -> None:
        grades.add_message(f'{self.display_name}')
        passed, total = self._run_tests(grades, module_dict, solution_dicts)
        if total > 0:
            self.points_earned = self.max_points * passed // total
        else:
            self.points_earned = self.max_points
        grades.add_points(self.points_earned)
        grades.add_message(
            f'  ({passed}/{total} tests passed -> {self.points_earned}/{self.max_points} pts)'
        )
