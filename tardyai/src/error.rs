#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("io error: {0}")]
    IO(#[from] std::io::Error),

    #[error("homedir error: {0}")]
    Home(#[from] homedir::GetHomeError),

    #[error("tar entry error: {0}")]
    TarEntry(&'static str),

    #[error("image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("download name not specified: {0}")]
    DownloadNameNotSpecified(String),

    #[error("error with safetensors file: {0:?}")]
    Safetensors(dfdx::tensor::safetensors::Error),

    #[error("error with optimizer update: {0}")]
    OptimizerUpdate(#[from] dfdx::optim::OptimizerUpdateError<dfdx::tensor::CpuError>),

    #[error("not enough tensor names")]
    NotEnoughNames,

    #[error("error converting number formats")]
    NumberFormatException,

    #[error("error while decoding image '{0}': {1}")]
    DecodeImageError(std::path::PathBuf, image::ImageError),
}

impl From<dfdx::tensor::safetensors::Error> for Error {
    fn from(value: dfdx::tensor::safetensors::Error) -> Self {
        Self::Safetensors(value)
    }
}

impl From<safetensors::SafeTensorError> for Error {
    fn from(value: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(dfdx::tensor::safetensors::Error::SafeTensorError(value))
    }
}
