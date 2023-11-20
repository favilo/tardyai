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
}
