//! Minimal Unreal archive explorer: IoStore (.utoc/.ucas) and legacy .pak (via repak).
//! Format reference: UE FIoStoreTocHeader / CUE4Parse IoStoreReader.
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

mod anim;
mod bp;
mod global;
mod gui;
mod mesh;
mod material;
mod skeletal;
mod preview;
mod props;
mod texture;
mod zen;

/// Game archive for the `#[ignore]` data tests: `UCAS_TEST_UTOC=/path/to/pakchunk0.utoc cargo test -- --ignored`.
#[cfg(test)]
fn test_utoc() -> String {
    std::env::var("UCAS_TEST_UTOC").expect("set UCAS_TEST_UTOC to a .utoc to run the data tests")
}

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Cur<'a>(&'a [u8], usize);
impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let s = self.0.get(self.1..self.1 + n).ok_or("utoc truncated")?;
        self.1 += n;
        Ok(s)
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn i32(&mut self) -> Res<i32> {
        Ok(self.u32()? as i32)
    }
    fn fstring(&mut self) -> Res<String> {
        let len = self.i32()?;
        if len >= 0 {
            let b = self.take(len as usize)?;
            Ok(String::from_utf8_lossy(b.strip_suffix(&[0]).unwrap_or(b)).into())
        } else {
            let b = self.take(-len as usize * 2)?;
            let w: Vec<u16> = b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            Ok(String::from_utf16_lossy(w.strip_suffix(&[0]).unwrap_or(&w)))
        }
    }
}

fn be40(b: &[u8]) -> u64 {
    b.iter().fold(0, |a, &x| (a << 8) | x as u64)
}
fn le(b: &[u8]) -> u64 {
    b.iter().rev().fold(0, |a, &x| (a << 8) | x as u64)
}

struct Block {
    offset: u64,
    csize: u32,
    usize: u32,
    method: u8,
}

struct Toc {
    block_size: u64,
    ids: Vec<(u64, u8)>,     // FIoChunkId: (id, chunk type), index = toc entry
    chunks: Vec<(u64, u64)>, // (offset, length) in uncompressed space
    blocks: Vec<Block>,
    methods: Vec<String>,
    files: Vec<(String, u32)>, // (path, toc entry index)
    pak: Option<repak::PakReader>, // legacy .pak: entry index = position in `files`, read via repak
}

impl Toc {
    /// Uncompressed size; None for .pak entries (repak doesn't expose entry sizes).
    fn size(&self, entry: u32) -> Option<u64> {
        self.chunks.get(entry as usize).map(|c| c.1)
    }
}

fn is_pak(p: &Path) -> bool {
    p.extension().is_some_and(|e| e.eq_ignore_ascii_case("pak"))
}
/// File holding the index: .pak itself, else the .utoc (a .ucas has no index).
fn index_path(p: &Path) -> PathBuf {
    if is_pak(p) { p.into() } else { p.with_extension("utoc") }
}
/// File holding the data: .pak itself, else the .ucas.
fn data_path(p: &Path) -> PathBuf {
    if is_pak(p) { p.into() } else { p.with_extension("ucas") }
}

fn open_archive(p: &Path) -> Res<Toc> {
    if !is_pak(p) {
        return parse_toc(&std::fs::read(p)?);
    }
    let pak = repak::PakBuilder::new().reader(&mut io::BufReader::new(File::open(p)?))?;
    let mut names = pak.files();
    names.sort();
    let files = names.into_iter().enumerate().map(|(i, n)| (n, i as u32)).collect();
    Ok(Toc { block_size: 1, ids: vec![], chunks: vec![], blocks: vec![], methods: vec![], files, pak: Some(pak) })
}

const FLAG_ENCRYPTED: u8 = 0x2;
const FLAG_SIGNED: u8 = 0x4;
const FLAG_INDEXED: u8 = 0x8;

fn parse_toc(buf: &[u8]) -> Res<Toc> {
    if buf.get(..16) != Some(b"-==--==--==--==-") {
        return Err("not a .utoc (bad magic)".into());
    }
    let mut h = Cur(buf, 16);
    let version = h.take(4)?[0];
    let header_size = h.u32()? as usize;
    let entry_count = h.u32()? as usize;
    let block_count = h.u32()? as usize;
    let block_entry_size = h.u32()? as usize;
    let method_count = h.u32()? as usize;
    let method_len = h.u32()? as usize;
    let block_size = h.u32()? as u64;
    let dir_index_size = h.u32()? as usize;
    let partition_count = h.u32()?;
    h.take(8 + 16)?; // container id, encryption key guid
    let flags = h.take(4)?[0];
    let seeds = h.u32()? as usize;
    h.take(8)?; // partition size
    let no_hash = h.u32()? as usize;

    if flags & FLAG_ENCRYPTED != 0 {
        return Err("encrypted container: AES key support not implemented".into());
    }
    // ponytail: single partition only; add .ucas_s1.. lookup if a game ships multi-partition containers
    if partition_count > 1 {
        return Err("multi-partition containers not supported".into());
    }

    let mut c = Cur(buf, header_size);
    let ids = c
        .take(entry_count * 12)?
        .chunks(12)
        .map(|e| (le(&e[..8]), e[11]))
        .collect();
    let chunks = c
        .take(entry_count * 10)?
        .chunks(10)
        .map(|e| (be40(&e[..5]), be40(&e[5..])))
        .collect();
    if version >= 4 {
        c.take(seeds * 4)?;
    }
    if version >= 5 {
        c.take(no_hash * 4)?;
    }
    let blocks = c
        .take(block_count * block_entry_size)?
        .chunks(block_entry_size)
        .map(|e| Block {
            offset: le(&e[0..5]),
            csize: le(&e[5..8]) as u32,
            usize: le(&e[8..11]) as u32,
            method: e[11],
        })
        .collect();
    let methods = c
        .take(method_count * method_len)?
        .chunks(method_len)
        .map(|n| String::from_utf8_lossy(n).trim_end_matches('\0').to_string())
        .collect();
    if flags & FLAG_SIGNED != 0 {
        let hash_size = c.i32()? as usize;
        c.take(hash_size * 2 + block_count * 20)?;
    }
    let mut files = Vec::new();
    if flags & FLAG_INDEXED != 0 && dir_index_size > 0 {
        parse_dir_index(c.take(dir_index_size)?, &mut files)?;
    }
    Ok(Toc { block_size, ids, chunks, blocks, methods, files, pak: None })
}

fn parse_dir_index(buf: &[u8], out: &mut Vec<(String, u32)>) -> Res<()> {
    let mut c = Cur(buf, 0);
    let mount = c.fstring()?;
    let n = c.u32()? as usize;
    let dirs: Vec<[u32; 4]> = (0..n)
        .map(|_| Ok([c.u32()?, c.u32()?, c.u32()?, c.u32()?]))
        .collect::<Res<_>>()?;
    let n = c.u32()? as usize;
    let files: Vec<[u32; 3]> = (0..n)
        .map(|_| Ok([c.u32()?, c.u32()?, c.u32()?]))
        .collect::<Res<_>>()?;
    let n = c.u32()? as usize;
    let strings: Vec<String> = (0..n).map(|_| c.fstring()).collect::<Res<_>>()?;

    const NONE: u32 = u32::MAX;
    let name = |i: u32| strings.get(i as usize).map(String::as_str).ok_or("bad string index");
    // iterative DFS: (dir index, path prefix)
    let mut stack = vec![(0u32, mount.trim_start_matches("../").to_string())];
    while let Some((d, prefix)) = stack.pop() {
        let Some(&[dname, first_child, next_sibling, first_file]) = dirs.get(d as usize) else { continue };
        let _ = dname;
        let mut f = first_file;
        while f != NONE {
            let [fname, next, user] = *files.get(f as usize).ok_or("bad file index")?;
            out.push((format!("{prefix}{}", name(fname)?), user));
            f = next;
        }
        let mut ch = first_child;
        while ch != NONE {
            let cd = dirs.get(ch as usize).ok_or("bad dir index")?;
            stack.push((ch, format!("{prefix}{}/", name(cd[0])?)));
            ch = cd[2];
        }
        let _ = next_sibling;
    }
    out.sort();
    Ok(())
}

/// Reads up to `limit` bytes of a chunk (u64::MAX = whole chunk).
fn read_chunk(toc: &Toc, ucas: &mut File, entry: u32, limit: u64) -> Res<Vec<u8>> {
    if let Some(pak) = &toc.pak {
        let (path, _) = toc.files.get(entry as usize).ok_or("bad pak entry")?;
        let mut data = pak.get(path, ucas)?; // ponytail: reads whole entry even for previews; pak files are small
        data.truncate(limit.min(data.len() as u64) as usize);
        return Ok(data);
    }
    let (off, len) = *toc.chunks.get(entry as usize).ok_or("bad toc entry")?;
    let len = len.min(limit);
    let first = (off / toc.block_size) as usize;
    let last = ((off + len).div_ceil(toc.block_size)) as usize;
    let mut out = Vec::with_capacity((last - first) * toc.block_size as usize);
    let mut cbuf = Vec::new();
    for b in toc.blocks.get(first..last).ok_or("bad block range")? {
        cbuf.resize(b.csize as usize, 0);
        ucas.seek(SeekFrom::Start(b.offset))?;
        ucas.read_exact(&mut cbuf)?;
        let start = out.len();
        match b.method {
            0 => out.extend_from_slice(&cbuf),
            m => {
                let name = toc.methods.get(m as usize - 1).map(String::as_str).unwrap_or("?");
                if !name.eq_ignore_ascii_case("oodle") {
                    return Err(format!("unsupported compression: {name}").into());
                }
                out.resize(start + b.usize as usize, 0);
                oozextract::Extractor::new().read_from_slice(&cbuf, &mut out[start..])?;
            }
        }
    }
    let skip = (off % toc.block_size) as usize;
    Ok(out[skip..skip + len as usize].to_vec())
}

fn extract_one(toc: &Toc, ucas: &mut File, out_dir: &Path, path: &str, entry: u32) -> Res<()> {
    // trust boundary: paths come from the archive, refuse traversal
    if path.split('/').any(|s| s == ".." || s.is_empty()) {
        return Err("unsafe path".into());
    }
    let dst = out_dir.join(path);
    let data = read_chunk(toc, ucas, entry, u64::MAX)?;
    std::fs::create_dir_all(dst.parent().ok_or("no parent")?)?;
    Ok(std::fs::write(&dst, data)?)
}

/// Converts every export of class `class` ("Texture2D" -> .png, "StaticMesh" -> .obj) in a .uasset,
/// writing `<out_dir>/<path>[_<export>].<ext>`. Returns how many were written.
fn export_converted(toc: &Toc, res: &global::Resolver, ucas: &mut File, out_dir: &Path, path: &str, entry: u32, class: &str) -> Res<usize> {
    if path.split('/').any(|s| s == ".." || s.is_empty()) {
        return Err("unsafe path".into());
    }
    let pkg = zen::parse(read_chunk(toc, ucas, entry, u64::MAX)?)?;
    let stem = path.rsplit_once('.').map_or(path, |(s, _)| s);
    let ubulk_path = format!("{stem}.ubulk");
    let ubulk = match toc.files.iter().find(|(p, _)| *p == ubulk_path) {
        Some((_, e)) => Some(read_chunk(toc, ucas, *e, u64::MAX)?),
        None => None,
    };
    let hits: Vec<usize> = (0..pkg.exports.len()).filter(|&i| res.class_name(&pkg, i) == class).collect();
    // reads another package/bulk file of this container by path (materials, textures)
    // ponytail: linear path lookup per load; add a path->entry map if batch exports get slow
    let mut load = |p: &str| -> Option<Vec<u8>> {
        let (_, e) = toc.files.iter().find(|(q, _)| q == p)?;
        read_chunk(toc, ucas, *e, u64::MAX).ok()
    };
    for &i in &hits {
        let (data, ext) = match class {
            "SkeletalMesh" => {
                let m = skeletal::decode_skeletal_mesh(&pkg, i, ubulk.as_deref())?;
                let need = m.sections.iter().map(|s| s.0 as usize + 1).max().unwrap_or(1);
                let tex = material::slot_diffuse_pngs(res, &pkg, i, need, &mut load);
                (skeletal::to_glb(&m, &pkg.exports[i].name, &tex)?, "glb")
            }
            "StaticMesh" => {
                // OBJ + MTL + PNG textures, named after the export, in the asset's folder
                let dir = out_dir.join(Path::new(stem).parent().unwrap_or(Path::new("")));
                material::export_obj_with_materials(res, &pkg, i, ubulk.as_deref(), &mut load, &dir)?;
                continue;
            }
            _ => (texture::to_png(&texture::decode_texture2d(pkg.export_data(i), None, ubulk.as_deref())?)?, "png"),
        };
        let name = if hits.len() == 1 { format!("{stem}.{ext}") } else { format!("{stem}_{}.{ext}", pkg.exports[i].name) };
        let dst = out_dir.join(name);
        std::fs::create_dir_all(dst.parent().ok_or("no parent")?)?;
        std::fs::write(dst, data)?;
    }
    Ok(hits.len())
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 2 {
        return gui::run(args.get(1).cloned());
    }
    let usage = "usage (<archive> = .utoc, .ucas or .pak):\n  ucas-explorer [archive]              (GUI)\n  ucas-explorer <archive> list [filter]\n  ucas-explorer <archive> extract <filter> <out_dir>\n  ucas-explorer <archive> png <filter> <out_dir>      (Texture2D -> PNG)\n  ucas-explorer <archive> obj <filter> <out_dir>      (StaticMesh LOD0 -> OBJ+MTL+PNG)\n  ucas-explorer <archive> glb <filter> <out_dir>      (SkeletalMesh -> rigged, textured .glb)\n  ucas-explorer <archive> mat <filter> <out_dir>      (Material/MI -> PNG textures + .mtl + parameter list)\n  ucas-explorer <archive> anim <filter> <out_dir> [mesh_filter]  (AnimSequence -> .glb with skeleton, mesh, animation)\n  ucas-explorer <archive> props <filter> <out_dir> [usmap]  (properties of every export -> <asset>.json; usmap auto-detected in <archive dir>/Mappings/)\n  ucas-explorer <archive> bp <filter> <out_dir>       (Blueprint actor -> assembled OBJ+MTL+PNG of its mesh components)";
    let (Some(utoc), Some(cmd)) = (args.get(1), args.get(2)) else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    let utoc = &index_path(Path::new(utoc)).display().to_string(); // accept .ucas too
    let toc = open_archive(Path::new(utoc))?;
    let filter = args.get(3).map(|s| s.to_lowercase()).unwrap_or_default();
    let matches = toc.files.iter().filter(|(p, _)| p.to_lowercase().contains(&filter));

    match cmd.as_str() {
        "list" => {
            let mut out = io::stdout().lock();
            for (p, e) in matches {
                use io::Write;
                let len = toc.size(*e).map_or("-".into(), |n| n.to_string());
                if writeln!(out, "{len:>12}  {p}").is_err() {
                    break; // broken pipe (e.g. | head)
                }
            }
        }
        "extract" => {
            let out_dir = PathBuf::from(args.get(4).ok_or(usage)?);
            let mut ucas = File::open(data_path(Path::new(utoc)))?;
            let (mut ok, mut fail) = (0, 0);
            for (p, e) in matches {
                match extract_one(&toc, &mut ucas, &out_dir, p, *e) {
                    Ok(()) => ok += 1,
                    Err(err) => {
                        fail += 1;
                        eprintln!("fail {p}: {err}");
                    }
                }
            }
            eprintln!("extracted {ok}, failed {fail}");
        }
        "png" | "obj" | "glb" => {
            let class = match cmd.as_str() {
                "png" => "Texture2D",
                "obj" => "StaticMesh",
                _ => "SkeletalMesh",
            };
            let out_dir = PathBuf::from(args.get(4).ok_or(usage)?);
            let mut ucas = File::open(data_path(Path::new(utoc)))?;
            let res = global::Resolver::load(&Path::new(utoc).with_file_name("global.utoc"), &toc)?;
            let (mut ok, mut fail) = (0, 0);
            for (p, e) in matches.filter(|(p, _)| p.ends_with(".uasset")) {
                match export_converted(&toc, &res, &mut ucas, &out_dir, p, *e, class) {
                    Ok(n) => ok += n,
                    Err(err) => {
                        fail += 1;
                        eprintln!("fail {p}: {err}");
                    }
                }
            }
            eprintln!("{class} written {ok}, failed {fail}");
        }
        "mat" => {
            let out_dir = PathBuf::from(args.get(4).ok_or(usage)?);
            let mut ucas = File::open(data_path(Path::new(utoc)))?;
            let res = global::Resolver::load(&Path::new(utoc).with_file_name("global.utoc"), &toc)?;
            let (mut ok, mut fail) = (0, 0);
            for (p, e) in matches.filter(|(p, _)| p.ends_with(".uasset")) {
                let r = (|| -> Res<bool> {
                    let pkg = zen::parse(read_chunk(&toc, &mut ucas, *e, u64::MAX)?)?;
                    let is_mat = (0..pkg.exports.len()).any(|i| {
                        let c = res.class_name(&pkg, i);
                        c == "Material" || c.starts_with("MaterialInstance")
                    });
                    if !is_mat {
                        return Ok(false);
                    }
                    let mut load = |q: &str| -> Option<Vec<u8>> {
                        let (_, e) = toc.files.iter().find(|(f, _)| f == q)?;
                        read_chunk(&toc, &mut ucas, *e, u64::MAX).ok()
                    };
                    let dir = out_dir.join(Path::new(p).parent().unwrap_or(Path::new("")));
                    material::export_material(&res, &pkg.name, &mut load, &dir)?;
                    Ok(true)
                })();
                match r {
                    Ok(true) => ok += 1,
                    Ok(false) => {}
                    Err(err) => {
                        fail += 1;
                        eprintln!("fail {p}: {err}");
                    }
                }
            }
            eprintln!("materials written {ok}, failed {fail}");
        }
        "props" => {
            let out_dir = PathBuf::from(args.get(4).ok_or(usage)?);
            let game = props::game_name(&toc.files);
            let usmap = match args.get(5) {
                // an explicit usmap is also saved to the per-game library, so it auto-loads next time
                Some(p) => match &game {
                    Some(g) => {
                        let lib = props::Mappings::remember(Path::new(p), g)?;
                        eprintln!("remembered mappings for {g}: {}", lib.display());
                        lib
                    }
                    None => PathBuf::from(p),
                },
                None => props::Mappings::find(Path::new(utoc), game.as_deref()).ok_or(
                    "no .usmap found (<archive dir>/Mappings/, next to the archive, or ~/.ucas-explorer/mappings/<game>/); pass one once and it is remembered",
                )?,
            };
            let maps = props::Mappings::load(&usmap)?;
            eprintln!("mappings: {} ({} structs, {} enums)", usmap.display(), maps.structs.len(), maps.enums.len());
            let mut ucas = File::open(data_path(Path::new(utoc)))?;
            let res = global::Resolver::load(&Path::new(utoc).with_file_name("global.utoc"), &toc)?;
            let (mut ok, mut fail) = (0, 0);
            for (p, e) in matches.filter(|(p, _)| p.ends_with(".uasset") || p.ends_with(".umap")) {
                let r = (|| -> Res<()> {
                    if p.split('/').any(|s| s == ".." || s.is_empty()) {
                        return Err("unsafe path".into());
                    }
                    let pkg = zen::parse(read_chunk(&toc, &mut ucas, *e, u64::MAX)?)?;
                    let dst = out_dir.join(format!("{}.json", p.rsplit_once('.').map_or(p.as_str(), |(s, _)| s)));
                    std::fs::create_dir_all(dst.parent().ok_or("no parent")?)?;
                    Ok(std::fs::write(dst, props::package_json(&pkg, &maps, &res))?)
                })();
                match r {
                    Ok(()) => ok += 1,
                    Err(err) => {
                        fail += 1;
                        eprintln!("fail {p}: {err}");
                    }
                }
            }
            eprintln!("JSON written {ok}, failed {fail} (per-export property errors are inside the JSON)");
        }
        "bp" => {
            let out_dir = PathBuf::from(args.get(4).ok_or(usage)?);
            let game = props::game_name(&toc.files);
            let usmap = props::Mappings::find(Path::new(utoc), game.as_deref()).ok_or("bp needs a .usmap (see the props command)")?;
            let maps = props::Mappings::load(&usmap)?;
            let mut ucas = File::open(data_path(Path::new(utoc)))?;
            let res = global::Resolver::load(&Path::new(utoc).with_file_name("global.utoc"), &toc)?;
            let (mut ok, mut fail) = (0, 0);
            for (p, e) in matches.filter(|(p, _)| p.ends_with(".uasset")) {
                let r = (|| -> Res<bool> {
                    if p.split('/').any(|s| s == ".." || s.is_empty()) {
                        return Err("unsafe path".into());
                    }
                    let pkg = zen::parse(read_chunk(&toc, &mut ucas, *e, u64::MAX)?)?;
                    if !(0..pkg.exports.len()).any(|i| res.class_name(&pkg, i) == "BlueprintGeneratedClass") {
                        return Ok(false);
                    }
                    let mut load = |q: &str| -> Option<Vec<u8>> {
                        let (_, e) = toc.files.iter().find(|(f, _)| f == q)?;
                        read_chunk(&toc, &mut ucas, *e, u64::MAX).ok()
                    };
                    let asm = bp::assemble(&res, &maps, &pkg, &mut load)?;
                    eprint!("{p}: {} parts\n{}", asm.parts.len(), bp::summary(&asm));
                    let dir = out_dir.join(Path::new(p).parent().unwrap_or(Path::new("")));
                    let files = bp::export_obj(&res, &asm, &mut load, &dir)?;
                    eprintln!("  wrote {} files to {}", files.len(), dir.display());
                    Ok(true)
                })();
                match r {
                    Ok(true) => ok += 1,
                    Ok(false) => {}
                    Err(err) => {
                        fail += 1;
                        eprintln!("fail {p}: {err}");
                    }
                }
            }
            eprintln!("Blueprints written {ok}, failed {fail}");
        }
        "anim" => {
            let out_dir = PathBuf::from(args.get(4).ok_or(usage)?);
            let mesh_hint = args.get(5).map(String::as_str);
            let mut ucas = File::open(data_path(Path::new(utoc)))?;
            let res = global::Resolver::load(&Path::new(utoc).with_file_name("global.utoc"), &toc)?;
            let (mut ok, mut fail) = (0, 0);
            for (p, e) in matches.filter(|(p, _)| p.ends_with(".uasset")) {
                let r = (|| -> Res<bool> {
                    let pkg = zen::parse(read_chunk(&toc, &mut ucas, *e, u64::MAX)?)?;
                    let Some(i) = (0..pkg.exports.len()).find(|&i| res.class_name(&pkg, i) == "AnimSequence") else { return Ok(false) };
                    let mut load = |q: &str| -> Option<Vec<u8>> {
                        let (_, e) = toc.files.iter().find(|(f, _)| f == q)?;
                        read_chunk(&toc, &mut ucas, *e, u64::MAX).ok()
                    };
                    let (glb, mesh) = anim::anim_to_glb(&res, &pkg, i, mesh_hint, &mut load)?;
                    let name = p.rsplit('/').next().unwrap_or(p).trim_end_matches(".uasset");
                    std::fs::create_dir_all(&out_dir)?;
                    std::fs::write(out_dir.join(format!("{name}.glb")), glb)?;
                    eprintln!("{name}.glb (mesh: {})", mesh.as_deref().unwrap_or("none, skeleton only"));
                    Ok(true)
                })();
                match r {
                    Ok(true) => ok += 1,
                    Ok(false) => {}
                    Err(err) => {
                        fail += 1;
                        eprintln!("fail {p}: {err}");
                    }
                }
            }
            eprintln!("AnimSequence written {ok}, failed {fail}");
        }
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn be40_and_le() {
        assert_eq!(super::be40(&[0, 0, 0, 1, 2]), 0x0102);
        assert_eq!(super::le(&[2, 1, 0]), 0x0102);
    }
}
