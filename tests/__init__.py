"""The V2 test suite is a regular package (not a namespace package).

Rationale (first surfaced on the authorized Colab T4, S-V2-16): Colab
runtimes ship a rogue top-level ``tests`` package under
``/usr/local/lib/python3.13/dist-packages`` (a third-party wheel leaks
its test directory). Per PEP 420 a regular package found anywhere on
``sys.path`` beats a repo-root *namespace* package, so
``from tests.test_supervised_dataset import ...`` resolved to the rogue
and collection failed. Declaring ``tests`` as a real package makes the
repository's own ``tests`` win at ``sys.path[0]`` (the rootdir) — the
same defensive pattern ``tests/suite_b/__init__.py`` already applies
against basename collisions.
"""
