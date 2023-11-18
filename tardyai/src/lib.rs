use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf,
};

const S3_BASE: &str = "https://s3.amazonaws.com/fast-ai-";
const S3_IMAGE: &str = "imageclas/";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("io error: {0}")]
    IO(#[from] std::io::Error),

    #[error("homedir error: {0}")]
    Home(#[from] homedir::GetHomeError),
}

#[derive(Debug, Clone, Copy)]
pub enum Url {
    Pets,
}

impl Url {
    pub fn url(self) -> String {
        match self {
            Self::Pets => {
                format!("{}{}{}", S3_BASE, S3_IMAGE, "oxford-iiit-pet.tgz")
            }
        }
    }
}

fn ensure_dir(path: &PathBuf) -> Result<(), Error> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

pub fn untar_images(url: Url) -> Result<PathBuf, Error> {
    let dest_dir = homedir::get_my_home()?
        .expect("home directory needs to exist")
        .join(".tardyai")
        .join("archive");
    ensure_dir(&dest_dir)?;
    download_archive(url, &dest_dir)?;
    // TODO: untar the archive

    Ok(dest_dir)
}

fn download_archive(url: Url, dest_dir: &PathBuf) -> Result<(), Error> {
    let mut response = reqwest::blocking::get(url.url())?;
    let mut dest = {
        let archive_name = response
            .url()
            .path_segments()
            .and_then(|s| s.last())
            .and_then(|name| if name.is_empty() { None } else { Some(name) })
            .unwrap_or("tmp.tar.gz");

        let archive_file = dest_dir.join(archive_name);

        // TODO: check if the archive is valid and exists
        if archive_file.exists() {
            log::info!("Archive already exists: {}", archive_file.display());
            return Ok(());
        }

        log::info!(
            "Downloading {} to archive: {}",
            url.url(),
            archive_file.display()
        );
        File::create(&archive_file)?
    };
    response.copy_to(&mut dest)?;
    Ok(())
}
