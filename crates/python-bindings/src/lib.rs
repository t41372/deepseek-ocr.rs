use std::path::PathBuf;

use anyhow::{Context, Result};
use deepseek_ocr_assets as assets;
use deepseek_ocr_config::{AppConfig, LocalFileSystem, ResourceLocation, VirtualFileSystem, VirtualPath};
use deepseek_ocr_core::{
    DecodeOutcome, DecodeParameters, ModelKind, ModelLoadArgs, OcrEngine, VisionSettings,
    normalize_text, render_prompt,
    runtime::default_dtype_for_device,
};
use image::DynamicImage;
use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use tokenizers::Tokenizer;

// Model loading functions
use deepseek_ocr_infer_deepseek::load_model as load_deepseek_model;
use deepseek_ocr_infer_dots::load_model as load_dots_model;
use deepseek_ocr_infer_paddleocr::load_model as load_paddle_model;

/// Initialize Python module
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OcrModel>()?;
    m.add_class::<DecodeParams>()?;
    m.add_class::<VisionConfig>()?;
    m.add_class::<OcrResult>()?;
    m.add_class::<Device>()?;
    m.add_class::<Precision>()?;
    m.add_function(wrap_pyfunction!(list_available_models, m)?)?;
    Ok(())
}

/// Represents the compute device for model inference.
#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Device {
    /// CPU device
    Cpu,
    /// CUDA device (NVIDIA GPU)
    Cuda,
    /// Metal device (Apple Silicon)
    Metal,
}

impl Device {
    fn to_candle_device(&self) -> Result<candle_core::Device> {
        match self {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda => candle_core::Device::new_cuda(0).context("Failed to initialize CUDA device"),
            Device::Metal => candle_core::Device::new_metal(0).context("Failed to initialize Metal device"),
        }
    }
}

/// Represents the precision/dtype for model weights.
#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Precision {
    /// 16-bit floating point
    F16,
    /// 32-bit floating point
    F32,
    /// Brain floating point 16
    BF16,
    /// Auto-select based on device
    Auto,
}

impl Precision {
    fn to_candle_dtype(&self, device: &candle_core::Device) -> candle_core::DType {
        match self {
            Precision::F16 => candle_core::DType::F16,
            Precision::F32 => candle_core::DType::F32,
            Precision::BF16 => candle_core::DType::BF16,
            Precision::Auto => default_dtype_for_device(device),
        }
    }
}

/// Vision preprocessing configuration.
#[pyclass]
#[derive(Clone, Debug)]
pub struct VisionConfig {
    /// Base resolution for image processing
    #[pyo3(get, set)]
    pub base_size: u32,

    /// Target image size
    #[pyo3(get, set)]
    pub image_size: u32,

    /// Enable crop mode
    #[pyo3(get, set)]
    pub crop_mode: bool,
}

#[pymethods]
impl VisionConfig {
    #[new]
    #[pyo3(signature = (base_size=1024, image_size=640, crop_mode=true))]
    fn new(base_size: u32, image_size: u32, crop_mode: bool) -> Self {
        Self {
            base_size,
            image_size,
            crop_mode,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "VisionConfig(base_size={}, image_size={}, crop_mode={})",
            self.base_size, self.image_size, self.crop_mode
        )
    }
}

impl From<&VisionConfig> for VisionSettings {
    fn from(config: &VisionConfig) -> Self {
        VisionSettings {
            base_size: config.base_size,
            image_size: config.image_size,
            crop_mode: config.crop_mode,
        }
    }
}

/// Decoding/generation parameters for OCR inference.
#[pyclass]
#[derive(Clone, Debug)]
pub struct DecodeParams {
    /// Maximum number of new tokens to generate
    #[pyo3(get, set)]
    pub max_new_tokens: usize,

    /// Enable sampling (vs greedy decoding)
    #[pyo3(get, set)]
    pub do_sample: bool,

    /// Sampling temperature
    #[pyo3(get, set)]
    pub temperature: f64,

    /// Top-p (nucleus) sampling threshold
    #[pyo3(get, set)]
    pub top_p: Option<f64>,

    /// Top-k sampling limit
    #[pyo3(get, set)]
    pub top_k: Option<usize>,

    /// Repetition penalty
    #[pyo3(get, set)]
    pub repetition_penalty: f32,

    /// No-repeat n-gram size
    #[pyo3(get, set)]
    pub no_repeat_ngram_size: Option<usize>,

    /// Random seed for sampling
    #[pyo3(get, set)]
    pub seed: Option<u64>,

    /// Use KV cache
    #[pyo3(get, set)]
    pub use_cache: bool,
}

#[pymethods]
impl DecodeParams {
    #[new]
    #[pyo3(signature = (
        max_new_tokens=512,
        do_sample=false,
        temperature=0.0,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
        no_repeat_ngram_size=None,
        seed=None,
        use_cache=true
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

    fn __repr__(&self) -> String {
        format!(
            "DecodeParams(max_new_tokens={}, do_sample={}, temperature={}, use_cache={})",
            self.max_new_tokens, self.do_sample, self.temperature, self.use_cache
        )
    }
}

impl From<&DecodeParams> for DecodeParameters {
    fn from(params: &DecodeParams) -> Self {
        DecodeParameters {
            max_new_tokens: params.max_new_tokens,
            do_sample: params.do_sample,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repetition_penalty: params.repetition_penalty,
            no_repeat_ngram_size: params.no_repeat_ngram_size,
            seed: params.seed,
            use_cache: params.use_cache,
        }
    }
}

/// OCR inference result.
#[pyclass]
#[derive(Clone, Debug)]
pub struct OcrResult {
    /// Extracted text
    #[pyo3(get)]
    pub text: String,

    /// Number of prompt tokens
    #[pyo3(get)]
    pub prompt_tokens: usize,

    /// Number of generated tokens
    #[pyo3(get)]
    pub response_tokens: usize,
}

#[pymethods]
impl OcrResult {
    fn __repr__(&self) -> String {
        format!(
            "OcrResult(text={:?}, prompt_tokens={}, response_tokens={})",
            &self.text[..self.text.len().min(50)],
            self.prompt_tokens,
            self.response_tokens
        )
    }

    fn __str__(&self) -> &str {
        &self.text
    }
}

impl From<DecodeOutcome> for OcrResult {
    fn from(outcome: DecodeOutcome) -> Self {
        Self {
            text: normalize_text(&outcome.text),
            prompt_tokens: outcome.prompt_tokens,
            response_tokens: outcome.response_tokens,
        }
    }
}

/// Main OCR model class for inference.
#[pyclass]
pub struct OcrModel {
    model: Box<dyn OcrEngine>,
    tokenizer: Tokenizer,
    model_name: String,
    device: candle_core::Device,
    dtype: candle_core::DType,
}

// Safety: OcrEngine trait implementations are Send, and we only access them from Python's GIL.
// The model is never shared across Python threads without proper synchronization.
unsafe impl Send for OcrModel {}
unsafe impl Sync for OcrModel {}

#[pymethods]
impl OcrModel {
    /// Load an OCR model.
    ///
    /// Args:
    ///     model_name: Name of the model (e.g., "deepseek-ocr", "paddleocr-vl", "dots-ocr")
    ///     device: Compute device (Device.Cpu, Device.Cuda, or Device.Metal)
    ///     precision: Model precision (Precision.F16, F32, BF16, or Auto)
    ///     config_path: Optional path to model config.json
    ///     weights_path: Optional path to model weights
    ///     snapshot_path: Optional path to quantized snapshot
    ///
    /// Returns:
    ///     Loaded OcrModel instance
    #[new]
    #[pyo3(signature = (
        model_name,
        device=Device::Cpu,
        precision=Precision::Auto,
        config_path=None,
        weights_path=None,
        snapshot_path=None
    ))]
    fn new(
        model_name: String,
        device: Device,
        precision: Precision,
        config_path: Option<PathBuf>,
        weights_path: Option<PathBuf>,
        snapshot_path: Option<PathBuf>,
    ) -> PyResult<Self> {
        // Initialize filesystem and config
        let fs = LocalFileSystem::new("deepseek-ocr");
        let (mut app_config, _) = AppConfig::load_or_init(&fs, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load config: {}", e)))?;

        // Override active model
        app_config.models.active = model_name.clone();
        app_config.normalise(&fs)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to normalize config: {}", e)))?;

        let resources = app_config.active_model_resources(&fs)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get model resources: {}", e)))?;

        // Prepare device and dtype
        let candle_device = device.to_candle_device()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize device: {}", e)))?;
        let candle_dtype = precision.to_candle_dtype(&candle_device);

        // Get resource paths
        let config_file = config_path.unwrap_or_else(|| {
            ensure_config_file(&fs, &resources.id, &resources.config)
                .expect("Failed to get config file")
        });

        let tokenizer_file = ensure_tokenizer_file(&fs, &resources.id, &resources.tokenizer)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get tokenizer: {}", e)))?;

        let weights_file = weights_path.unwrap_or_else(|| {
            prepare_weights_path(&fs, &resources.id, &resources.weights)
                .expect("Failed to get weights file")
        });

        let snapshot_file = snapshot_path.or_else(|| {
            prepare_snapshot_path(&fs, &resources.id, resources.snapshot.as_ref()).ok().flatten()
        });

        // Ensure preprocessor config for dots.ocr
        let _ = assets::ensure_model_preprocessor_for(&resources.id, &config_file);

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load tokenizer: {}", e)))?;

        // Load model
        let load_args = ModelLoadArgs {
            kind: resources.kind,
            config_path: Some(&config_file),
            weights_path: Some(&weights_file),
            snapshot_path: snapshot_file.as_deref(),
            device: candle_device.clone(),
            dtype: candle_dtype,
        };

        let model: Box<dyn OcrEngine> = match resources.kind {
            ModelKind::Deepseek =>
                load_deepseek_model(load_args)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load DeepSeek model: {}", e)))?,
            ModelKind::PaddleOcrVl =>
                load_paddle_model(load_args)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load PaddleOCR model: {}", e)))?,
            ModelKind::DotsOcr =>
                load_dots_model(load_args)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load DotsOCR model: {}", e)))?,
        };

        Ok(Self {
            model,
            tokenizer,
            model_name,
            device: candle_device,
            dtype: candle_dtype,
        })
    }

    /// Perform OCR on image(s) with a text prompt.
    ///
    /// Args:
    ///     prompt: Text prompt (use "<image>" as placeholder for images)
    ///     images: List of image paths or single image path
    ///     vision_config: Optional vision preprocessing config
    ///     decode_params: Optional decoding parameters
    ///     template: Conversation template name (default: "plain")
    ///     system_prompt: System prompt to prepend (default: "")
    ///
    /// Returns:
    ///     OcrResult with extracted text and token counts
    #[pyo3(signature = (prompt, images, vision_config=None, decode_params=None, template=None, system_prompt=None))]
    fn infer(
        &self,
        prompt: String,
        images: Vec<PathBuf>,
        vision_config: Option<VisionConfig>,
        decode_params: Option<DecodeParams>,
        template: Option<String>,
        system_prompt: Option<String>,
    ) -> PyResult<OcrResult> {
        let vision = vision_config.unwrap_or_else(|| VisionConfig::new(1024, 640, true));
        let params = decode_params.unwrap_or_else(|| DecodeParams::new(
            512, false, 0.0, None, None, 1.0, None, None, true
        ));

        let template_name = template.as_deref().unwrap_or("plain");
        let system_msg = system_prompt.as_deref().unwrap_or("");

        // Render prompt with template
        let rendered_prompt = render_prompt(template_name, system_msg, &prompt)
            .map_err(|e| PyValueError::new_err(format!("Failed to render prompt: {}", e)))?;

        // Verify image placeholder count
        let image_slots = rendered_prompt.matches("<image>").count();
        if image_slots != images.len() {
            return Err(PyValueError::new_err(format!(
                "Prompt contains {} <image> placeholders but {} images were provided",
                image_slots,
                images.len()
            )));
        }

        // Load images
        let loaded_images: Vec<DynamicImage> = images
            .iter()
            .map(|path| {
                image::open(path)
                    .with_context(|| format!("Failed to open image at {}", path.display()))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Run inference
        let vision_settings = VisionSettings::from(&vision);
        let decode_parameters = DecodeParameters::from(&params);

        let outcome = self.model.decode(
            &self.tokenizer,
            &rendered_prompt,
            &loaded_images,
            vision_settings,
            &decode_parameters,
            None,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Inference failed: {}", e)))?;

        Ok(OcrResult::from(outcome))
    }

    /// Get model information.
    fn info(&self) -> String {
        format!(
            "OcrModel(name={}, device={:?}, dtype={:?})",
            self.model_name, self.device, self.dtype
        )
    }

    fn __repr__(&self) -> String {
        self.info()
    }
}

/// List available OCR models.
#[pyfunction]
fn list_available_models() -> Vec<String> {
    vec![
        "deepseek-ocr".to_string(),
        "deepseek-ocr-q4k".to_string(),
        "deepseek-ocr-q6k".to_string(),
        "deepseek-ocr-q8k".to_string(),
        "paddleocr-vl".to_string(),
        "paddleocr-vl-q4k".to_string(),
        "paddleocr-vl-q6k".to_string(),
        "paddleocr-vl-q8k".to_string(),
        "dots-ocr".to_string(),
        "dots-ocr-q4k".to_string(),
        "dots-ocr-q6k".to_string(),
        "dots-ocr-q8k".to_string(),
    ]
}

// Helper functions (mirroring crates/cli/src/resources.rs)

fn ensure_config_file(
    fs: &LocalFileSystem,
    model_id: &str,
    location: &ResourceLocation,
) -> Result<PathBuf> {
    match location {
        ResourceLocation::Physical(path) => assets::ensure_model_config_for(model_id, path),
        ResourceLocation::Virtual(_) => {
            let baseline_id = assets::baseline_model_id(model_id);
            if baseline_id == model_id {
                ensure_resource(fs, location, |path| {
                    assets::ensure_model_config_for(&baseline_id, path)
                })
            } else {
                let vpath = VirtualPath::model_config(baseline_id.clone());
                fs.with_physical_path(&vpath, |physical| {
                    assets::ensure_model_config_for(&baseline_id, physical)
                })
            }
        }
    }
}

fn ensure_tokenizer_file(
    fs: &LocalFileSystem,
    model_id: &str,
    location: &ResourceLocation,
) -> Result<PathBuf> {
    match location {
        ResourceLocation::Physical(path) => assets::ensure_model_tokenizer_for(model_id, path),
        ResourceLocation::Virtual(_) => {
            let baseline_id = assets::baseline_model_id(model_id);
            if baseline_id == model_id {
                ensure_resource(fs, location, |path| {
                    assets::ensure_model_tokenizer_for(&baseline_id, path)
                })
            } else {
                let vpath = VirtualPath::model_tokenizer(baseline_id.clone());
                fs.with_physical_path(&vpath, |physical| {
                    assets::ensure_model_tokenizer_for(&baseline_id, physical)
                })
            }
        }
    }
}

fn prepare_weights_path(
    fs: &LocalFileSystem,
    model_id: &str,
    location: &ResourceLocation,
) -> Result<PathBuf> {
    let baseline_id = assets::baseline_model_id(model_id);
    if baseline_id == model_id {
        ensure_resource(fs, location, |path| {
            assets::ensure_model_weights_for(model_id, path)
        })
    } else {
        let vpath = VirtualPath::model_weights(baseline_id.clone());
        fs.with_physical_path(&vpath, |physical| {
            assets::ensure_model_weights_for(&baseline_id, physical)
        })
    }
}

fn prepare_snapshot_path(
    fs: &LocalFileSystem,
    model_id: &str,
    snapshot: Option<&deepseek_ocr_config::SnapshotResources>,
) -> Result<Option<PathBuf>> {
    let Some(entry) = snapshot else {
        return Ok(None);
    };
    ensure_resource(fs, &entry.location, |path| {
        assets::ensure_model_snapshot_for(model_id, &entry.dtype, path)
    })
    .map(Some)
}

fn ensure_resource<F>(
    fs: &LocalFileSystem,
    location: &ResourceLocation,
    ensure_fn: F,
) -> Result<PathBuf>
where
    F: Fn(&std::path::Path) -> Result<PathBuf>,
{
    match location {
        ResourceLocation::Physical(path) => ensure_fn(path),
        ResourceLocation::Virtual(vpath) => {
            fs.with_physical_path(vpath, |physical| ensure_fn(physical))
        }
    }
}
