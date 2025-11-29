#![allow(unsafe_op_in_unsafe_fn)]

use std::{
    io,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use deepseek_ocr_core::{
    normalize_text as normalize_text_impl, render_prompt as render_prompt_impl, DecodeOutcome,
    DecodeParameters, ModelKind, ModelLoadArgs, OcrEngine, VisionSettings,
};
use deepseek_ocr_assets as assets;
use deepseek_ocr_config::fs::{LocalFileSystem, VirtualPath};
use deepseek_ocr_config::VirtualFileSystem;
use deepseek_ocr_infer_deepseek::load_model as load_deepseek_model;
use deepseek_ocr_infer_dots::load_model as load_dots_model;
use deepseek_ocr_infer_paddleocr::load_model as load_paddle_model;
use image::DynamicImage;
use pyo3::{
    exceptions::{
        PyFileNotFoundError, PyPermissionError, PyRuntimeError, PyTypeError, PyValueError,
    },
    prelude::*,
    types::{PyAny, PyBytes, PyList, PySequence},
    Bound, PyErr, PyResult, Python,
};
use tokenizers::Tokenizer;

#[pyclass(module = "deepseek_ocr_rs._native")]
pub struct VisionSettingsInput {
    pub base_size: u32,
    pub image_size: u32,
    pub crop_mode: bool,
}

#[pymethods]
impl VisionSettingsInput {
    #[new]
    #[pyo3(signature = (base_size, image_size, crop_mode))]
    fn new(base_size: u32, image_size: u32, crop_mode: bool) -> Self {
        Self {
            base_size,
            image_size,
            crop_mode,
        }
    }
}

impl From<&VisionSettingsInput> for VisionSettings {
    fn from(value: &VisionSettingsInput) -> Self {
        Self {
            base_size: value.base_size,
            image_size: value.image_size,
            crop_mode: value.crop_mode,
        }
    }
}

#[pyclass(module = "deepseek_ocr_rs._native")]
pub struct DecodeParametersInput {
    pub max_new_tokens: usize,
    pub do_sample: bool,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: Option<usize>,
    pub seed: Option<u64>,
    pub use_cache: bool,
}

#[pymethods]
impl DecodeParametersInput {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        max_new_tokens,
        do_sample,
        temperature,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
        no_repeat_ngram_size=None,
        seed=None,
        use_cache=true,
    ))]
    fn new(
        max_new_tokens: usize,
        do_sample: bool,
        temperature: f64,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repetition_penalty: f32,
        no_repeat_ngram_size: Option<usize>,
        seed: Option<u64>,
        use_cache: bool,
    ) -> Self {
        Self {
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            no_repeat_ngram_size,
            seed,
            use_cache,
        }
    }
}

impl From<&DecodeParametersInput> for DecodeParameters {
    fn from(value: &DecodeParametersInput) -> Self {
        Self {
            max_new_tokens: value.max_new_tokens,
            do_sample: value.do_sample,
            temperature: value.temperature,
            top_p: value.top_p,
            top_k: value.top_k,
            repetition_penalty: value.repetition_penalty,
            no_repeat_ngram_size: value.no_repeat_ngram_size,
            seed: value.seed,
            use_cache: value.use_cache,
        }
    }
}

#[pyclass(module = "deepseek_ocr_rs._native")]
pub struct DecodeOutcomeHandle {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub prompt_tokens: usize,
    #[pyo3(get)]
    pub response_tokens: usize,
    #[pyo3(get)]
    pub generated_tokens: Vec<i64>,
}

impl From<DecodeOutcome> for DecodeOutcomeHandle {
    fn from(value: DecodeOutcome) -> Self {
        Self {
            text: value.text,
            prompt_tokens: value.prompt_tokens,
            response_tokens: value.response_tokens,
            generated_tokens: value.generated_tokens,
        }
    }
}

struct EngineState {
    tokenizer: Tokenizer,
    model: Mutex<Box<dyn OcrEngine>>,
}

impl EngineState {
    fn new(tokenizer: Tokenizer, model: Box<dyn OcrEngine>) -> Self {
        Self {
            tokenizer,
            model: Mutex::new(model),
        }
    }
}

#[pyclass(module = "deepseek_ocr_rs._native")]
pub struct EngineHandle {
    state: Arc<EngineState>,
}

#[pymethods]
impl EngineHandle {
    #[pyo3(signature = (prompt, images, vision, decode, stream=None))]
    fn decode(
        &self,
        py: Python<'_>,
        prompt: &str,
        images: Bound<'_, PyAny>,
        vision: &VisionSettingsInput,
        decode: &DecodeParametersInput,
        stream: Option<PyObject>,
    ) -> PyResult<DecodeOutcomeHandle> {
        let sequence: Bound<'_, PySequence> = images.downcast_into()?;
        let mut decoded_images: Vec<DynamicImage> = Vec::new();
        for item in sequence.iter()? {
            let obj = item?;
            let bytes = obj
                .downcast::<PyBytes>()
                .map_err(|_| PyTypeError::new_err("images must be a sequence of bytes"))?;
            let img = image::load_from_memory(bytes.as_bytes())
                .map_err(|err| PyValueError::new_err(format!("failed to decode image: {err}")))?;
            decoded_images.push(img);
        }

        let vision_settings: VisionSettings = vision.into();
        let decode_params: DecodeParameters = decode.into();

        let callback = stream.map(|callable| callable.into_py(py));
        // Keep the Python callback alive across `allow_threads` by storing it in an `Arc`.
        // The extra Send + Sync bounds placate the thread-safety requirement of
        // `allow_threads`, while the actual invocation re-acquires the GIL.
        let stream_holder: Option<Arc<dyn Fn(usize, &[i64]) + Send + Sync>> = callback.map(
            |py_obj| {
                Arc::new(move |count: usize, ids: &[i64]| {
                    Python::with_gil(|py| {
                        let list = PyList::new_bound(py, ids);
                        let result = py_obj.call1(py, (count, list));
                        if let Err(err) = result {
                            err.print(py);
                        }
                    });
                }) as Arc<dyn Fn(usize, &[i64]) + Send + Sync>
            },
        );

        let stream_ref: Option<&(dyn Fn(usize, &[i64]) + Send + Sync)> =
            stream_holder.as_deref();

        let outcome: Result<DecodeOutcome> = py.allow_threads(|| {
            let stream_cb = stream_ref.map(|cb| cb as &dyn Fn(usize, &[i64]));
            let model = self
                .state
                .model
                .lock()
                .map_err(|_| anyhow!("engine poisoned by previous panic"))?;

            model.decode(
                &self.state.tokenizer,
                prompt,
                &decoded_images,
                vision_settings,
                &decode_params,
                stream_cb,
            )
        });

        let outcome = outcome.map_err(to_pyerr)?;
        Ok(DecodeOutcomeHandle::from(outcome))
    }
}

fn to_pyerr(err: anyhow::Error) -> PyErr {
    let msg = err.to_string();

    for cause in err.chain() {
        if let Some(io_err) = cause.downcast_ref::<io::Error>() {
            return match io_err.kind() {
                io::ErrorKind::NotFound => PyFileNotFoundError::new_err(msg.clone()),
                io::ErrorKind::PermissionDenied => PyPermissionError::new_err(msg.clone()),
                io::ErrorKind::InvalidInput | io::ErrorKind::UnexpectedEof => {
                    PyValueError::new_err(msg.clone())
                }
                _ => PyRuntimeError::new_err(msg.clone()),
            };
        }
    }

    let lower = msg.to_lowercase();
    if lower.contains("invalid")
        || lower.contains("mismatch")
        || lower.contains("expected")
        || lower.contains("parse")
    {
        PyValueError::new_err(msg)
    } else {
        PyRuntimeError::new_err(msg)
    }
}

#[derive(Clone, Copy)]
enum DeviceRequest {
    Cpu,
    Metal,
    Cuda,
}

impl DeviceRequest {
    fn parse(value: Option<&str>) -> PyResult<Self> {
        match value.unwrap_or("cpu").to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            other => Err(PyValueError::new_err(format!(
                "unknown device `{}` (expected cpu|metal|cuda)",
                other
            ))),
        }
    }

    fn materialize(self) -> PyResult<(Device, DType)> {
        use deepseek_ocr_core::runtime::{
            default_dtype_for_device, prepare_device_and_dtype, DeviceKind, Precision,
        };
        let (device_kind, precision): (DeviceKind, Option<Precision>) = match self {
            Self::Cpu => (DeviceKind::Cpu, Some(Precision::F32)),
            Self::Metal => (DeviceKind::Metal, None),
            Self::Cuda => (DeviceKind::Cuda, None),
        };
        let (device, maybe_dtype) =
            prepare_device_and_dtype(device_kind, precision).map_err(to_pyerr)?;
        let dtype = maybe_dtype.unwrap_or_else(|| default_dtype_for_device(&device));
        Ok((device, dtype))
    }
}

fn parse_dtype(value: Option<&str>, device: &Device) -> PyResult<DType> {
    use deepseek_ocr_core::runtime::{default_dtype_for_device, dtype_from_precision, Precision};
    if let Some(label) = value {
        let precision = match label.to_lowercase().as_str() {
            "f32" => Precision::F32,
            "f16" => Precision::F16,
            "bf16" => Precision::Bf16,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported dtype `{}` (expected f32|f16|bf16)",
                    other
                )))
            }
        };
        Ok(dtype_from_precision(precision))
    } else {
        Ok(default_dtype_for_device(device))
    }
}

fn model_kind_to_label(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Deepseek => "deepseek",
        ModelKind::PaddleOcrVl => "paddle",
        ModelKind::DotsOcr => "dots",
    }
}

fn model_from_args(args: ModelArguments) -> Result<(Tokenizer, Box<dyn OcrEngine>)> {
    match args.kind {
        EngineKind::Deepseek => load_from_loader(args, ModelKind::Deepseek, load_deepseek_model),
        EngineKind::Paddle => load_from_loader(args, ModelKind::PaddleOcrVl, load_paddle_model),
        EngineKind::Dots => load_from_loader(args, ModelKind::DotsOcr, load_dots_model),
        #[cfg(feature = "mock-engine")]
        EngineKind::Mock => Ok((build_mock_tokenizer(), Box::new(MockEngine::default()))),
    }
}

fn load_from_loader(
    args: ModelArguments,
    kind: ModelKind,
    loader: fn(ModelLoadArgs<'_>) -> Result<Box<dyn OcrEngine>>,
) -> Result<(Tokenizer, Box<dyn OcrEngine>)> {
    let tokenizer_path = args
        .tokenizer_path
        .as_deref()
        .context("tokenizer_path is required for this engine")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|err| {
        anyhow!(
            "failed to load tokenizer from {}: {err}",
            tokenizer_path.display()
        )
    })?;

    let config_path = args
        .config_path
        .as_deref()
        .context("config_path is required for this engine")?;
    let load_args = ModelLoadArgs {
        kind,
        config_path: Some(config_path),
        weights_path: args.weights_path.as_deref(),
        snapshot_path: args.snapshot_path.as_deref(),
        device: args.device.clone(),
        dtype: args.dtype,
    };
    let model = loader(load_args)?;
    Ok((tokenizer, model))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EngineKind {
    Deepseek,
    Paddle,
    Dots,
    #[cfg(feature = "mock-engine")]
    Mock,
}

impl EngineKind {
    fn parse(raw: &str) -> PyResult<Self> {
        match raw.to_lowercase().as_str() {
            "deepseek" => Ok(Self::Deepseek),
            "paddle" | "paddleocr" | "paddleocr_vl" => Ok(Self::Paddle),
            "dots" | "dots_ocr" => Ok(Self::Dots),
            #[cfg(feature = "mock-engine")]
            "mock" => Ok(Self::Mock),
            other => Err(PyValueError::new_err(format!(
                "unknown model `{}` (expected deepseek|paddle|dots)",
                other
            ))),
        }
    }
}

struct ModelArguments {
    kind: EngineKind,
    config_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    weights_path: Option<PathBuf>,
    snapshot_path: Option<PathBuf>,
    device: Device,
    dtype: DType,
}

#[pyfunction]
#[pyo3(signature = (
    model_kind,
    *,
    config_path=None,
    tokenizer_path=None,
    weights_path=None,
    snapshot_path=None,
    device=None,
    dtype=None,
))]
fn create_engine(
    model_kind: &str,
    config_path: Option<&str>,
    tokenizer_path: Option<&str>,
    weights_path: Option<&str>,
    snapshot_path: Option<&str>,
    device: Option<&str>,
    dtype: Option<&str>,
) -> PyResult<EngineHandle> {
    let kind = EngineKind::parse(model_kind)?;
    let device_request = DeviceRequest::parse(device)?;
    let (device, _default_dtype) = device_request.materialize()?;
    let dtype = parse_dtype(dtype, &device)?;
    let args = ModelArguments {
        kind,
        config_path: config_path.map(PathBuf::from),
        tokenizer_path: tokenizer_path.map(PathBuf::from),
        weights_path: weights_path.map(PathBuf::from),
        snapshot_path: snapshot_path.map(PathBuf::from),
        device,
        dtype,
    };
    let (tokenizer, model) = model_from_args(args).map_err(to_pyerr)?;
    let state = EngineState::new(tokenizer, model);
    Ok(EngineHandle {
        state: Arc::new(state),
    })
}

/// Return the engine label (deepseek|paddle|dots) for a given model id if it is
/// registered in the Rust asset catalog. Unknown ids return `None` so the
/// Python layer can fall back to heuristics or user overrides.
#[pyfunction]
fn engine_for_model(model_id: &str) -> PyResult<Option<String>> {
    let baseline = assets::baseline_model_id(model_id);
    let engine = assets::MODEL_ASSETS
        .iter()
        .find(|asset| asset.id == baseline)
        .map(|asset| model_kind_to_label(asset.kind).to_string());
    Ok(engine)
}

/// Whether the given model id represents a quantized snapshot that requires a
/// `.dsq` file. Used by the Python binding to decide when to insist on
/// snapshot assets.
#[pyfunction]
fn model_requires_snapshot(model_id: &str) -> PyResult<bool> {
    let needs_snapshot = assets::QUANTIZED_MODEL_ASSETS
        .iter()
        .any(|asset| asset.id == model_id);
    Ok(needs_snapshot)
}

#[pyfunction]
#[pyo3(signature = (model_id, cache_dir=None))]
fn download_model(model_id: &str, cache_dir: Option<&str>) -> PyResult<DownloadResultHandle> {
    let cache_override = cache_dir.map(PathBuf::from);
    let paths = materialize_model_assets(model_id, cache_override).map_err(to_pyerr)?;
    Ok(DownloadResultHandle::from(paths))
}

#[pyfunction]
fn normalize_text(text: &str) -> String {
    normalize_text_impl(text)
}

#[pyfunction]
fn render_prompt(template: &str, system_prompt: &str, raw_prompt: &str) -> PyResult<String> {
    render_prompt_impl(template, system_prompt, raw_prompt).map_err(to_pyerr)
}

#[pymodule]
fn _native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<EngineHandle>()?;
    module.add_class::<VisionSettingsInput>()?;
    module.add_class::<DecodeParametersInput>()?;
    module.add_class::<DecodeOutcomeHandle>()?;
    module.add_class::<DownloadResultHandle>()?;
    module.add_function(wrap_pyfunction!(create_engine, module)?)?;
    module.add_function(wrap_pyfunction!(engine_for_model, module)?)?;
    module.add_function(wrap_pyfunction!(model_requires_snapshot, module)?)?;
    module.add_function(wrap_pyfunction!(download_model, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_text, module)?)?;
    module.add_function(wrap_pyfunction!(render_prompt, module)?)?;
    #[cfg(feature = "mock-engine")]
    module.add("MOCK_MODEL_KIND", "mock")?;
    Ok(())
}

#[cfg(feature = "mock-engine")]
mod mock {
    use super::*;
    use once_cell::sync::Lazy;
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    pub struct MockEngine;

    impl Default for MockEngine {
        fn default() -> Self {
            Self
        }
    }

    impl OcrEngine for MockEngine {
        fn kind(&self) -> ModelKind {
            ModelKind::Deepseek
        }

        fn device(&self) -> &Device {
            static DEVICE: Lazy<Device> = Lazy::new(|| Device::Cpu);
            &DEVICE
        }

        fn dtype(&self) -> DType {
            DType::F32
        }

        fn decode(
            &self,
            _tokenizer: &Tokenizer,
            prompt: &str,
            images: &[DynamicImage],
            vision: VisionSettings,
            params: &DecodeParameters,
            stream: deepseek_ocr_core::inference::StreamCallback,
        ) -> Result<DecodeOutcome> {
            let mut text = format!(
                "mock-response(prompt_bytes={}, images={}, base={}, size={}, max_tokens={})",
                prompt.len(),
                images.len(),
                vision.base_size,
                vision.image_size,
                params.max_new_tokens,
            );
            if params.do_sample {
                text.push_str("|sampled");
            }
            let generated_tokens: Vec<i64> = (0..text.len() as i64).collect();
            if let Some(cb) = stream {
                for idx in 1..=generated_tokens.len() {
                    cb(idx, &generated_tokens[..idx]);
                }
            }
            Ok(DecodeOutcome {
                text,
                prompt_tokens: 4,
                response_tokens: generated_tokens.len(),
                generated_tokens,
            })
        }
    }

    pub fn build_mock_tokenizer() -> Tokenizer {
        use ahash::AHashMap;
        let mut vocab = AHashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .expect("wordlevel model");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
        tokenizer
    }
}

#[cfg(feature = "mock-engine")]
use mock::{build_mock_tokenizer, MockEngine};

#[derive(Debug, Clone)]
struct DownloadResult {
    model_id: String,
    model_dir: PathBuf,
    baseline_dir: PathBuf,
    config_path: PathBuf,
    tokenizer_path: PathBuf,
    weights_path: PathBuf,
    snapshot_path: Option<PathBuf>,
    preprocessor_path: Option<PathBuf>,
}

#[pyclass(module = "deepseek_ocr_rs._native")]
pub struct DownloadResultHandle {
    #[pyo3(get)]
    pub model_id: String,
    #[pyo3(get)]
    pub model_dir: String,
    #[pyo3(get)]
    pub baseline_dir: String,
    #[pyo3(get)]
    pub config_path: String,
    #[pyo3(get)]
    pub tokenizer_path: String,
    #[pyo3(get)]
    pub weights_path: String,
    #[pyo3(get)]
    pub snapshot_path: Option<String>,
    #[pyo3(get)]
    pub preprocessor_path: Option<String>,
}

impl From<DownloadResult> for DownloadResultHandle {
    fn from(value: DownloadResult) -> Self {
        Self {
            model_id: value.model_id,
            model_dir: value.model_dir.display().to_string(),
            baseline_dir: value.baseline_dir.display().to_string(),
            config_path: value.config_path.display().to_string(),
            tokenizer_path: value.tokenizer_path.display().to_string(),
            weights_path: value.weights_path.display().to_string(),
            snapshot_path: value
                .snapshot_path
                .map(|p| p.display().to_string()),
            preprocessor_path: value
                .preprocessor_path
                .map(|p| p.display().to_string()),
        }
    }
}

fn default_model_dir(model_id: &str) -> Result<PathBuf> {
    let fs = LocalFileSystem::new("deepseek-ocr");
    let vpath = VirtualPath::model_dir(model_id.to_string());
    fs.with_physical_path(&vpath, |p| Ok(p.to_path_buf()))
}

fn materialize_model_assets(
    model_id: &str,
    cache_override: Option<PathBuf>,
) -> Result<DownloadResult> {
    let baseline_id = assets::baseline_model_id(model_id);
    let (config_name, tokenizer_name, weights_name, preprocessor, snapshot) =
        match baseline_id.as_str() {
            "deepseek-ocr" => (
                "config.json",
                "tokenizer.json",
                "model-00001-of-000001.safetensors",
                None,
                snapshot_for(model_id),
            ),
            "paddleocr-vl" => ("config.json", "tokenizer.json", "model.safetensors", None, snapshot_for(model_id)),
            "dots-ocr" => (
                "config.json",
                "tokenizer.json",
                "model.safetensors.index.json",
                Some("preprocessor_config.json"),
                snapshot_for(model_id),
            ),
            other => {
                return Err(anyhow::anyhow!("unknown model id `{}`", other));
            }
        };

    let baseline_dir = cache_override
        .clone()
        .map(|root| root.join(&baseline_id))
        .unwrap_or_else(|| default_model_dir(&baseline_id).unwrap_or_else(|_| PathBuf::from(".")));
    std::fs::create_dir_all(&baseline_dir)?;

    let config_path = assets::ensure_model_config_for(model_id, &baseline_dir.join(config_name))?;
    let tokenizer_path =
        assets::ensure_model_tokenizer_for(model_id, &baseline_dir.join(tokenizer_name))?;
    let weights_path =
        assets::ensure_model_weights_for(model_id, &baseline_dir.join(weights_name))?;

    let preprocessor_path = if let Some(_name) = preprocessor {
        assets::ensure_model_preprocessor_for(model_id, &config_path)?
    } else {
        None
    };

    let (model_dir, snapshot_path) = if let Some((dtype, snap_name)) = snapshot {
        let dir = cache_override
            .clone()
            .map(|root| root.join(model_id))
            .unwrap_or_else(|| default_model_dir(model_id).unwrap_or_else(|_| PathBuf::from(".")));
        std::fs::create_dir_all(&dir)?;
        let snap_path = assets::ensure_model_snapshot_for(model_id, dtype, &dir.join(snap_name))?;
        (dir, Some(snap_path))
    } else {
        (baseline_dir.clone(), None)
    };

    Ok(DownloadResult {
        model_id: model_id.to_string(),
        model_dir,
        baseline_dir,
        config_path,
        tokenizer_path,
        weights_path,
        snapshot_path,
        preprocessor_path,
    })
}

fn snapshot_for(model_id: &str) -> Option<(&'static str, &'static str)> {
    match model_id {
        "deepseek-ocr-q4k" => Some(("Q4_K", "DeepSeek-OCR.Q4_K.dsq")),
        "deepseek-ocr-q6k" => Some(("Q6_K", "DeepSeek-OCR.Q6_K.dsq")),
        "deepseek-ocr-q8k" => Some(("Q8_0", "DeepSeek-OCR.Q8_0.dsq")),
        "paddleocr-vl-q4k" => Some(("Q4_K", "PaddleOCR-VL.Q4_K.dsq")),
        "paddleocr-vl-q6k" => Some(("Q6_K", "PaddleOCR-VL.Q6_K.dsq")),
        "paddleocr-vl-q8k" => Some(("Q8_0", "PaddleOCR-VL.Q8_0.dsq")),
        "dots-ocr-q4k" => Some(("Q4_K", "dots.ocr.Q4_K.dsq")),
        "dots-ocr-q6k" => Some(("Q6_K", "dots.ocr.Q6_K.dsq")),
        "dots-ocr-q8k" => Some(("Q8_0", "dots.ocr.Q8_0.dsq")),
        _ => None,
    }
}
