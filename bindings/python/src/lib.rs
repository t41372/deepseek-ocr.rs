use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use deepseek_ocr_core::{
    normalize_text as normalize_text_impl, render_prompt as render_prompt_impl, DecodeOutcome,
    DecodeParameters, ModelKind, ModelLoadArgs, OcrEngine, VisionSettings,
};
use deepseek_ocr_infer_deepseek::load_model as load_deepseek_model;
use deepseek_ocr_infer_dots::load_model as load_dots_model;
use deepseek_ocr_infer_paddleocr::load_model as load_paddle_model;
use image::DynamicImage;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBytes, PyList, PySequence},
    Bound, PyErr, PyResult, Python,
};
use tokenizers::Tokenizer;

#[pyclass(module = "deepseek_ocr._native")]
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

#[pyclass(module = "deepseek_ocr._native")]
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

#[pyclass(module = "deepseek_ocr._native")]
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

#[pyclass(module = "deepseek_ocr._native")]
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
        let stream_holder = callback.map(|py_obj| {
            Box::new(move |count: usize, ids: &[i64]| {
                Python::with_gil(|py| {
                    let list = PyList::new_bound(py, ids);
                    let result = py_obj.call1(py, (count, list));
                    if let Err(err) = result {
                        err.print(py);
                    }
                });
            }) as Box<dyn Fn(usize, &[i64]) + Send + Sync>
        });

        let stream_ref = stream_holder
            .as_ref()
            .map(|cb| cb.as_ref() as &dyn Fn(usize, &[i64]));

        let model = self
            .state
            .model
            .lock()
            .map_err(|_| PyRuntimeError::new_err("engine poisoned by previous panic"))?;
        let outcome = model
            .decode(
                &self.state.tokenizer,
                prompt,
                &decoded_images,
                vision_settings,
                &decode_params,
                stream_ref,
            )
            .map_err(to_pyerr)?;
        Ok(DecodeOutcomeHandle::from(outcome))
    }
}

fn to_pyerr(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
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
    module.add_function(wrap_pyfunction!(create_engine, module)?)?;
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
