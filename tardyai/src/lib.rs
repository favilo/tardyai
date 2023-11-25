#![feature(generic_const_exprs)]
pub mod datasets;
pub mod download;
pub mod error;
pub mod learners;
pub mod models;

pub use self::{
    download::{untar_images, DatasetUrl},
    error::Error,
};
