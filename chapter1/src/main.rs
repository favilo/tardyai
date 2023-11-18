use std::path::PathBuf;

fn main() {
    let path: PathBuf = tardyai::untar_images(tardyai::Url::Pets).join("images");
}
