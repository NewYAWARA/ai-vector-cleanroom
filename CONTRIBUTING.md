# Contributing

Contributions are welcome.

Before opening a pull request:

- Do not add private, client-owned, trademarked, or unclear-license images.
- Prefer small focused changes with a short explanation.
- Test with at least one simple flat logo or synthetic sample.
- Keep output SVGs free of embedded bitmap images unless a feature explicitly
  documents why that is needed.
- Be careful with geometry regularization changes; a fix for circular logos can
  accidentally damage waves, fields, text strokes, or other curved details.

Useful checks:

```powershell
python -m py_compile vector_cleanroom.py clean_base.py trace_engine.py
python vector_cleanroom.py --help
python preflight_check.py
```
