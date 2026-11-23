# Running the tests

Run from the repository root in an isolated Python environment:

```sh
python -m pip install . -r requirements-test.txt
python -m pytest -q tests
```

The suite uses virtual CAN devices and does not require a connected robot.
`requirements-test.txt` includes the Nero tools' NumPy dependency, the
`dataclasses` backport for Python 3.6, and a compatible pytest version.
The CI matrix runs the same suite on Python 3.6 through 3.14.
