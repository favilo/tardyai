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

    #[error("error with dfdx: {0}")]
    Dfdx(dfdx::prelude::Error),

    #[error("error with dfdx tensors: {0:?}")]
    DfdxTensor(dfdx_core::tensor::Error),

    #[error("error with safetensors: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("error with optimizer update: {0}")]
    OptimizerUpdate(#[from] dfdx::optim::OptimizerUpdateError<dfdx::tensor::CpuError>),

    #[error("not enough tensor names")]
    NotEnoughNames,

    #[error("error converting number formats")]
    NumberFormatException,

    #[error("error while decoding image '{0}': {1}")]
    DecodeImageError(std::path::PathBuf, image::ImageError),

    #[error("no validation dataset defined")]
    NoValidationDataset,
}

// impl From<safetensors::SafeTensorError> for Error {
//     fn from(value: safetensors::SafeTensorError) -> Self {
//         Self::Safetensors(dfdx::tensor::safetensors::Error::SafeTensorError(value))
//     }
// }
