# Documentation

## Local build

Install doc dependencies:

```bash
pip install -r docs/requirements.txt
```

Build HTML:

```bash
cd docs
make html
```

Open the output at `docs/build/html/index.html`.

## API reference generation

Generate API stubs from the Python packages:

```bash
cd docs
make apidoc
make html
```

Note: generated stubs are written to `docs/source/api_reference/_generated` and are not
included in the default navigation. Add them to a toctree if you want to expose them.
