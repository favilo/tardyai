use std::{
    fs::File,
    io::{self, Seek, Write},
    path::PathBuf,
};

use flate2::read::GzDecoder;
use tar::Archive;

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

    #[error("tar entry error: {0}")]
    TarEntry(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub enum Url {
    Pets,
}

impl Url {
    pub fn url(self) -> String {
        match self {
            Self::Pets => {
                format!("{S3_BASE}{S3_IMAGE}oxford-iiit-pet.tgz")
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
    let home = &homedir::get_my_home()?
        .expect("home directory needs to exist")
        .join(".tardyai");
    let dest_dir = home.join("archive");
    ensure_dir(&dest_dir)?;
    let archive_file = download_archive(url, &dest_dir)?;

    let dest_dir = home.join("data");
    let dir = extract_archive(&archive_file, &dest_dir)?;

    Ok(dir)
}

fn download_archive(url: Url, dest_dir: &PathBuf) -> Result<PathBuf, Error> {
    let mut response = reqwest::blocking::get(url.url())?;
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
        return Ok(archive_file);
    }

    log::info!(
        "Downloading {} to archive: {}",
        url.url(),
        archive_file.display()
    );
    let mut dest = File::create(&archive_file)?;
    response.copy_to(&mut dest)?;
    Ok(archive_file)
}

fn extract_archive(archive_file: &PathBuf, dest_dir: &PathBuf) -> Result<PathBuf, Error> {
    let tar_gz = File::open(archive_file)?;
    let tar = GzDecoder::new(tar_gz);
    let mut archive = Archive::new(tar);

    log::info!(
        "Extracting archive {} to: {}",
        archive_file.display(),
        dest_dir.display()
    );
    let dir = {
        let entry = &archive
            .entries()?
            .next()
            .ok_or(Error::TarEntry("No entries in archive"))??;
        entry.path()?.into_owned()
    };
    let archive_dir = dest_dir.join(dir);
    if archive_dir.exists() {
        log::info!("Archive already extracted to: {}", archive_dir.display());
        return Ok(archive_dir);
    }

    let tar = archive.into_inner();
    let mut tar_gz = tar.into_inner();
    tar_gz.seek(io::SeekFrom::Start(0))?;
    let tar = GzDecoder::new(tar_gz);
    let mut archive = Archive::new(tar);
    archive.unpack(dest_dir)?;

    Ok(archive_dir)
}
