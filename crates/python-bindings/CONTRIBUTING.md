# Python Binding Development Guide

This guide is for **project maintainers** who need to understand how the Python bindings work and what to do when making changes to the Rust codebase.

## Architecture Overview

### Component Structure

```
crates/python-bindings/
├── src/
│   └── lib.rs           # PyO3 bindings (Rust → Python bridge)
├── python/
│   └── deepseek_ocr/
│       ├── __init__.py  # Python package entry point
│       ├── _core.pyi    # Type stubs for Rust module
│       └── py.typed     # PEP 561 marker for type checking
├── tests/               # Python tests
├── examples/            # Usage examples
├── Cargo.toml           # Rust crate configuration
├── pyproject.toml       # Python package configuration
└── README.md            # User documentation
```

### Data Flow

```
Python Code
    ↓
deepseek_ocr (Python package)
    ↓
_core (PyO3 Rust extension)
    ↓
deepseek-ocr-core (Rust library)
    ↓
Model inference engines (Rust)
```

## When Rust Code Changes

### If You Modify Core Rust APIs

**What changed:**
- `crates/core/src/inference.rs` - Core traits and structs
- `crates/infer-*/` - Model implementations

**What you need to do:**

1. **Update Python bindings** (`crates/python-bindings/src/lib.rs`)
   - Add/modify PyO3 wrapper structs to match new Rust APIs
   - Update `From` trait implementations if data structures changed
   - Add new functions/methods as needed

2. **Update type stubs** (`crates/python-bindings/python/deepseek_ocr/_core.pyi`)
   - Keep Python type annotations in sync with Rust changes
   - Document new parameters and return types

3. **Test the changes**
   - Run `maturin develop` to rebuild
   - Run Python tests: `pytest`
   - Run type checker: `mypy python/deepseek_ocr`

4. **Update documentation**
   - Update `README.md` with new API usage
   - Add examples if new functionality is significant

**Example: Adding a new parameter to `DecodeParameters`**

If you add `min_new_tokens: usize` to Rust's `DecodeParameters`:

```rust
// In crates/python-bindings/src/lib.rs

#[pyclass]
pub struct DecodeParams {
    #[pyo3(get, set)]
    pub min_new_tokens: usize,  // ADD THIS

    // ... existing fields ...
}

#[pymethods]
impl DecodeParams {
    #[new]
    #[pyo3(signature = (
        min_new_tokens=0,  // ADD THIS
        // ... existing params ...
    ))]
    fn new(
        min_new_tokens: usize,  // ADD THIS
        // ... existing params ...
    ) -> Self {
        Self {
            min_new_tokens,  // ADD THIS
            // ... existing fields ...
        }
    }
}

impl From<&DecodeParams> for DecodeParameters {
    fn from(params: &DecodeParams) -> Self {
        DecodeParameters {
            min_new_tokens: params.min_new_tokens,  // ADD THIS
            // ... existing fields ...
        }
    }
}
```

```python
# In crates/python-bindings/python/deepseek_ocr/_core.pyi

class DecodeParams:
    min_new_tokens: int  # ADD THIS

    def __init__(
        self,
        min_new_tokens: int = 0,  # ADD THIS
        # ... existing params ...
    ) -> None: ...
```

### If You Add a New Model

**What changed:**
- Added new crate in `crates/infer-newmodel/`
- Added new `ModelKind` variant

**What you need to do:**

1. **Add model loading** in `lib.rs`:

```rust
// Import the new loader
use deepseek_ocr_infer_newmodel::load_model as load_newmodel_model;

// In OcrModel::new(), add new match arm
let model: Box<dyn OcrEngine> = match resources.kind {
    // ... existing models ...
    ModelKind::NewModel => Box::new(
        load_newmodel_model(load_args)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load NewModel: {}", e)))?
    ),
};
```

2. **Update `list_available_models()`**:

```rust
#[pyfunction]
fn list_available_models() -> Vec<String> {
    vec![
        // ... existing models ...
        "newmodel".to_string(),
        "newmodel-q4k".to_string(),
        "newmodel-q6k".to_string(),
        "newmodel-q8k".to_string(),
    ]
}
```

3. **Update documentation**:
   - Add model to README.md's model list
   - Document model characteristics and use cases
   - Add example if the model has unique features

### If You Change Configuration System

**What changed:**
- `crates/config/` - Configuration structs and loading

**What you need to do:**

1. **Review initialization** in `OcrModel::new()`
   - Ensure `AppConfig::load_or_init()` still works
   - Update resource path preparation if needed

2. **Test configuration loading**
   - Verify models still load with default config
   - Test custom config paths
   - Check error messages are clear

### If You Modify Vision Processing

**What changed:**
- `VisionSettings` struct or preprocessing logic

**What you need to do:**

1. **Update `VisionConfig`** if struct fields changed:

```rust
#[pyclass]
pub struct VisionConfig {
    #[pyo3(get, set)]
    pub new_field: Type,  // Add new field
    // ... existing fields ...
}

#[pymethods]
impl VisionConfig {
    #[new]
    #[pyo3(signature = (new_field=default, ...))]
    fn new(new_field: Type, ...) -> Self {
        Self { new_field, ... }
    }
}

impl From<&VisionConfig> for VisionSettings {
    fn from(config: &VisionConfig) -> Self {
        VisionSettings {
            new_field: config.new_field,
            // ... existing fields ...
        }
    }
}
```

2. **Update type stubs** (`_core.pyi`)
3. **Update default values** in README.md examples

## Building and Testing

### Development Workflow

```bash
# Navigate to python-bindings crate
cd crates/python-bindings

# Build and install in development mode
maturin develop

# Build with specific features
maturin develop --release --features metal  # macOS
maturin develop --release --features cuda   # CUDA

# Run tests
pytest

# Type checking
mypy python/deepseek_ocr

# Linting
ruff check python/
```

### Adding Tests

**Unit tests** (`tests/test_unit.py`):
- Test individual classes and functions
- Test error handling
- Mock heavy dependencies

**Integration tests** (`tests/test_integration.py`):
- Test actual model loading (if resources available)
- Test end-to-end OCR workflows
- Mark slow tests with `@pytest.mark.slow`

**Example test:**

```python
def test_new_parameter():
    """Test that new_parameter works correctly."""
    params = DecodeParams(new_parameter=42)
    assert params.new_parameter == 42
```

### Type Checking

This package uses **strict mypy** configuration. All code must pass:

```bash
mypy python/deepseek_ocr
```

Common issues:
- Missing type annotations on functions
- Using `Any` without proper narrowing
- Missing return type annotations

Fix by adding explicit types:

```python
def my_function(param: str) -> OcrResult:  # ← Add annotations
    result: OcrResult = model.infer(...)  # ← Annotate variables
    return result
```

## Release Checklist

When preparing a new release:

- [ ] Update version in `pyproject.toml`
- [ ] Update version in `Cargo.toml`
- [ ] Update version in `python/deepseek_ocr/__init__.py`
- [ ] Update `README.md` with any API changes
- [ ] Run full test suite: `pytest`
- [ ] Run type checker: `mypy python/deepseek_ocr`
- [ ] Run linter: `ruff check python/`
- [ ] Test examples work: `python examples/basic_ocr.py`
- [ ] Build wheels: `maturin build --release`
- [ ] Test wheel installation: `pip install target/wheels/*.whl`
- [ ] Update CHANGELOG.md
- [ ] Create git tag: `git tag -a python-v0.x.0 -m "Python bindings v0.x.0"`

## Common Pitfalls

### 1. Forgetting to Rebuild After Rust Changes

**Problem:** Python sees old version of Rust code after editing `lib.rs`

**Solution:** Always run `maturin develop` after Rust changes

### 2. Type Stub Mismatch

**Problem:** `.pyi` file doesn't match actual Rust implementation

**Solution:** Keep `.pyi` in sync manually, test with `mypy`

### 3. Missing Error Handling

**Problem:** Rust `Result<T, E>` not properly converted to Python exceptions

**Solution:** Always use `.map_err()` to convert to PyO3 exceptions:

```rust
some_rust_function()
    .map_err(|e| PyRuntimeError::new_err(format!("Error: {}", e)))?
```

### 4. Memory Safety with Image Data

**Problem:** Image data might be freed before Rust processes it

**Solution:** PyO3 handles this automatically with `&[u8]`, but be cautious with custom buffer protocols

### 5. GIL (Global Interpreter Lock) Issues

**Problem:** Long-running Rust operations block Python threads

**Solution:** Use `py.allow_threads()` for CPU-intensive operations:

```rust
py.allow_threads(|| {
    // Long-running Rust operation
    self.model.decode(...)
})
```

## Debugging

### Enable Rust Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Rust tracing logs will appear if built with tracing-subscriber
```

### Python Debugger

```python
import pdb; pdb.set_trace()  # Set breakpoint
```

### Inspect Rust Panics

If Rust code panics, you'll see:

```
thread '<unnamed>' panicked at 'message', src/lib.rs:123:4
```

This indicates a bug in the Rust bindings that should be fixed.

## FAQ

**Q: Do I need to rebuild Python bindings for every Rust change?**

A: Yes, if you change any code in `crates/python-bindings/src/` or any dependency crates used by the bindings. Run `maturin develop`.

**Q: How do I add a new Python-only helper function?**

A: Add it to `python/deepseek_ocr/__init__.py` or create a new `.py` file. No Rust changes needed.

**Q: Can I expose internal Rust functions directly to Python?**

A: Yes, but avoid it. Wrap them in a safe Python-friendly API first. Internal Rust functions may have unsafe preconditions or complex lifetimes.

**Q: How do I handle breaking changes to the Rust API?**

A:
1. Update Python bindings to match
2. Bump major version (0.x.0 → 0.y.0)
3. Document migration path in CHANGELOG.md
4. Consider adding compatibility shims for one release

**Q: What if CI fails on type checking?**

A: Run `mypy python/deepseek_ocr` locally, fix all errors. Strict mode is intentional.

## Resources

- [PyO3 User Guide](https://pyo3.rs/)
- [maturin Documentation](https://maturin.rs/)
- [Python Packaging Guide](https://packaging.python.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)

## Getting Help

- Open an issue on GitHub with `[python-bindings]` tag
- Check existing issues for similar problems
- Include error messages, Rust/Python versions, and OS
