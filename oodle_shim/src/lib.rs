//! API subset of repak's oodle_loader that repak actually calls.
#[derive(Debug)]
pub enum Error {
    CompressionUnsupported,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("oodle compression not supported (read-only shim)")
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

pub enum Compressor {
    Mermaid,
}
pub enum CompressionLevel {
    Normal,
}

pub struct Oodle;
static OODLE: Oodle = Oodle;

pub fn oodle() -> Result<&'static Oodle> {
    Ok(&OODLE)
}

impl Oodle {
    /// Returns bytes written, 0 on failure (same contract as the real loader).
    pub fn decompress(&self, input: &[u8], output: &mut [u8]) -> isize {
        oozextract::Extractor::new().read_from_slice(input, output).map_or(0, |n| n as isize)
    }
    pub fn compress(&self, _: &[u8], _: Compressor, _: CompressionLevel) -> Result<Vec<u8>> {
        Err(Error::CompressionUnsupported)
    }
}
