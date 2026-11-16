//! Unity player data: SerializedFile (.assets, level*, globalgamemanagers; format 17-22).
//! Asset objects are listed as virtual files `<Class>/<name>.<ext>`; reading one converts it:
//! Texture2D -> PNG, Mesh -> OBJ, AudioClip -> WAV (PCM/ADPCM) or raw FSB5 (Vorbis etc.),
//! TextAsset -> its bytes, anything else -> the raw object bytes.
// ponytail: players strip the TypeTree, so Texture2D/Mesh/AudioClip layouts are hardcoded for Unity
// 2022+/6000 (Texture2D is checked to consume exactly the object's bytes). Other versions fail with a
// layout error. Upgrade: per-version layouts or embedded type trees (UnityPy's tpk). UnityFS
// .bundle files (Addressables) are not read yet.
use crate::{mesh::Mesh, texture::Image, Res};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub struct Obj {
    pub path_id: i64,
    pub offset: u64, // absolute, in the .assets file
    pub size: u32,
    pub class_id: i32,
}

pub struct Archive {
    pub path: PathBuf,
    pub unity: String,
    pub objects: Vec<Obj>, // toc entry index = position here
    /// referenced files (PPtr m_FileID k > 0 -> externals[k - 1]), as file names
    externals: Vec<String>,
    /// MeshFilter / MeshRenderer / SkinnedMeshRenderer objects: link meshes to their materials
    renderers: Vec<Obj>,
    /// (mesh file name lowercase, mesh path id) -> material PPtrs (relative to this file), built on first use
    mesh_materials: OnceLock<HashMap<(String, i64), Vec<(i32, i64)>>>,
    /// other SerializedFiles of the game, opened on demand for cross-file references
    siblings: Mutex<HashMap<String, Option<Arc<Archive>>>>,
}

/// Asset classes that get listed (scene objects and components are skipped).
fn class_name(id: i32) -> Option<&'static str> {
    Some(match id {
        21 => "Material",
        28 => "Texture2D",
        43 => "Mesh",
        48 => "Shader",
        49 => "TextAsset",
        74 => "AnimationClip",
        83 => "AudioClip",
        84 => "RenderTexture",
        89 => "Cubemap",
        90 => "Avatar",
        91 => "AnimatorController",
        115 => "MonoScript",
        117 => "Texture3D",
        128 => "Font",
        134 => "PhysicsMaterial",
        156 => "TerrainData",
        187 => "Texture2DArray",
        213 => "Sprite",
        221 => "AnimatorOverrideController",
        319 => "AvatarMask",
        329 => "VideoClip",
        _ => return None,
    })
}

/// True if `p` starts like a SerializedFile (big-endian header, format version 9..=50).
pub fn sniff(p: &Path) -> bool {
    let mut h = [0u8; 20];
    File::open(p).and_then(|mut f| f.read_exact(&mut h)).is_ok()
        && (9..=50).contains(&u32::from_be_bytes(h[8..12].try_into().unwrap()))
        && h[16] <= 1
}

struct R<'a>(&'a [u8], usize);
impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let s = self.0.get(self.1..self.1.checked_add(n).ok_or("unity: bad length")?).ok_or("unity: truncated")?;
        self.1 += n;
        Ok(s)
    }
    fn u8(&mut self) -> Res<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn i32(&mut self) -> Res<i32> {
        Ok(self.u32()? as i32)
    }
    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn f32(&mut self) -> Res<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn align(&mut self) {
        self.1 = self.1.next_multiple_of(4);
    }
    /// `count` then `count * elem` bytes, aligned (byte arrays, vectors of fixed-size structs).
    fn arr(&mut self, elem: usize) -> Res<&'a [u8]> {
        let n = self.u32()? as usize;
        let s = self.take(n.checked_mul(elem).ok_or("unity: bad array")?)?;
        self.align();
        Ok(s)
    }
    fn str(&mut self) -> Res<String> {
        Ok(String::from_utf8_lossy(self.arr(1)?).into())
    }
    fn cstr(&mut self) -> Res<String> {
        let rest = &self.0[self.1..];
        let n = rest.iter().position(|&b| b == 0).ok_or("unity: unterminated string")?;
        self.1 += n + 1;
        Ok(String::from_utf8_lossy(&rest[..n]).into())
    }
}

/// StreamingInfo / StreamedResource: (offset, size, file) of data stored next to the .assets.
struct Stream(u64, u64, String);

impl Archive {
    /// Parses the object table and names the listed objects. Returns the archive and its file list.
    pub fn open(path: &Path) -> Res<(Archive, Vec<(String, u32)>)> {
        let a = Archive::parse(path)?;
        let mut f = File::open(path)?;
        // name each object from its first bytes (m_Name is the first field of every listed class)
        let mut names: Vec<(String, u32)> = Vec::with_capacity(a.objects.len());
        let mut seen = std::collections::HashSet::new();
        for (i, o) in a.objects.iter().enumerate() {
            let head = a.read_at(&mut f, o.offset, o.size.min(4096) as usize)?;
            let name = R(&head, 0).str().map(|s| crate::material::sanitize(&s)).unwrap_or_default();
            let ext = match o.class_id {
                28 => "png",
                43 => "obj",
                49 => "txt",
                83 => match audio_header(&head) {
                    Ok(c) if c.compression == 0 || c.compression == 2 => "wav",
                    _ => "fsb",
                },
                _ => "dat",
            };
            let class = class_name(o.class_id).unwrap();
            let base = if name.is_empty() { format!("{class}/{}", o.path_id) } else { format!("{class}/{name}") };
            // duplicate names are common (e.g. many "Default" meshes): disambiguate by path id;
            // case-insensitively, since "Eye" and "eYE" are the same file on macOS/Windows
            let p = if seen.insert(base.to_lowercase()) { format!("{base}.{ext}") } else { format!("{base}_{}.{ext}", o.path_id) };
            names.push((p, i as u32));
        }
        names.sort();
        Ok((a, names))
    }

    /// Object table only (no names): used directly for sibling files.
    fn parse(path: &Path) -> Res<Archive> {
        let mut f = File::open(path)?;
        let mut h = [0u8; 48];
        f.read_exact(&mut h)?; // header is 20 bytes before format 22, 48 after; metadata follows either way
        let be = |o: usize, n: usize| h[o..o + n].iter().fold(0u64, |a, &b| (a << 8) | b as u64);
        let version = be(8, 4) as u32;
        let (meta_start, meta_size, data_offset) = if version >= 22 {
            (48, be(20, 4) as usize, be(32, 8))
        } else {
            (20, be(0, 4) as usize, be(12, 4))
        };
        if !(17..=22).contains(&version) {
            return Err(format!("unity: SerializedFile format {version} not supported (17..22 = Unity 5.5+)").into());
        }
        if h[16] != 0 {
            return Err("unity: big-endian SerializedFile not supported".into());
        }
        let mut meta = vec![0u8; meta_size];
        f.seek(SeekFrom::Start(meta_start))?;
        f.read_exact(&mut meta)?;
        let mut r = R(&meta, 0);
        let unity = r.cstr()?;
        r.i32()?; // platform
        let type_tree = r.u8()? != 0;
        let classes: Vec<i32> = (0..r.u32()?)
            .map(|_| {
                let id = r.i32()?;
                r.take(3)?; // isStrippedType, scriptTypeIndex
                r.take(if id == 114 { 32 } else { 16 })?; // [scriptID] + oldTypeHash
                if type_tree {
                    let (nodes, strings) = (r.u32()? as usize, r.u32()? as usize);
                    r.take(nodes * if version >= 19 { 32 } else { 24 } + strings)?;
                    if version >= 21 {
                        r.arr(4)?; // type dependencies
                    }
                }
                Ok(id)
            })
            .collect::<Res<_>>()?;
        let n = r.u32()? as usize;
        let (mut objects, mut renderers) = (Vec::new(), Vec::new());
        for _ in 0..n {
            r.align();
            let path_id = r.u64()? as i64;
            let start = if version >= 22 { r.u64()? } else { r.u32()? as u64 };
            let size = r.u32()?;
            let class_id = *classes.get(r.u32()? as usize).ok_or("unity: bad type index")?;
            let o = Obj { path_id, offset: data_offset + start, size, class_id };
            if class_name(class_id).is_some() {
                objects.push(o);
            } else if matches!(class_id, 23 | 33 | 137) {
                renderers.push(o);
            }
        }
        // script types (i32 file index, aligned i64 id), then externals (cstr, guid, i32 type, cstr path);
        // only needed for cross-file textures, so a short table is not an error
        let externals = (|| -> Res<Vec<String>> {
            for _ in 0..r.u32()? {
                r.i32()?;
                r.align();
                r.u64()?;
            }
            (0..r.u32()?)
                .map(|_| {
                    r.cstr()?;
                    r.take(20)?;
                    let p = r.cstr()?;
                    Ok(p.rsplit('/').next().unwrap_or(&p).to_string())
                })
                .collect()
        })()
        .unwrap_or_default();
        Ok(Archive { path: path.into(), unity, objects, externals, renderers, mesh_materials: OnceLock::new(), siblings: Mutex::new(HashMap::new()) })
    }


    fn file_name(&self) -> String {
        self.path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default()
    }

    /// Another SerializedFile of the game by file name (cached; None if missing or unreadable).
    fn sibling(&self, name: &str) -> Option<Arc<Archive>> {
        let mut cache = self.siblings.lock().ok()?;
        cache
            .entry(name.to_lowercase())
            .or_insert_with(|| {
                let p = self.path.with_file_name(name);
                (p != self.path && p.is_file()).then(|| Archive::parse(&p).ok().map(Arc::new)).flatten()
            })
            .clone()
    }

    /// Runs `f` on the object a PPtr (relative to this file) points to, in whichever file holds it.
    fn with_obj<T>(&self, file_id: i32, path_id: i64, f: impl FnOnce(&Archive, u32) -> Option<T>) -> Option<T> {
        let find = |a: &Archive| a.objects.iter().position(|o| o.path_id == path_id).map(|i| i as u32);
        if file_id == 0 {
            return f(self, find(self)?);
        }
        let a = self.sibling(self.externals.get(usize::try_from(file_id - 1).ok()?)?)?;
        let i = find(&a)?;
        f(&a, i)
    }

    /// (mesh file, mesh path id) -> material PPtrs of the renderers in this file that draw it.
    fn mesh_materials(&self) -> &HashMap<(String, i64), Vec<(i32, i64)>> {
        self.mesh_materials.get_or_init(|| {
            let Ok(mut f) = File::open(&self.path) else { return HashMap::new() };
            let pptr = |d: &[u8], o: usize| -> Option<(i32, i64)> {
                Some((i32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?), i64::from_le_bytes(d.get(o + 4..o + 12)?.try_into().ok()?)))
            };
            // Renderer (Unity 6000): m_Materials vector at 80; SkinnedMeshRenderer: m_Mesh 60 bytes after it
            // ponytail: fixed offsets for Unity 6000.x layouts (no type tree); other versions find no materials
            let materials = |d: &[u8]| -> Option<(Vec<(i32, i64)>, usize)> {
                let n = u32::from_le_bytes(d.get(80..84)?.try_into().ok()?) as usize;
                if n > 64 {
                    return None;
                }
                Some(((0..n).map(|k| pptr(d, 84 + k * 12)).collect::<Option<_>>()?, 84 + n * 12))
            };
            let mesh_key = |(fid, pid): (i32, i64)| -> Option<(String, i64)> {
                let file = if fid == 0 { self.file_name() } else { self.externals.get(usize::try_from(fid - 1).ok()?)?.to_lowercase() };
                Some((file, pid))
            };
            let (mut go_mesh, mut go_mats, mut out) = (HashMap::new(), HashMap::new(), HashMap::new());
            for o in &self.renderers {
                let Ok(d) = self.read_at(&mut f, o.offset, (o.size as usize).min(4096)) else { continue };
                let Some((_, go)) = pptr(&d, 0) else { continue };
                match o.class_id {
                    33 => {
                        if let Some(k) = pptr(&d, 12).and_then(mesh_key) {
                            go_mesh.insert(go, k);
                        }
                    }
                    23 => {
                        if let Some((m, _)) = materials(&d) {
                            go_mats.insert(go, m);
                        }
                    }
                    _ => {
                        if let Some((m, end)) = materials(&d) {
                            if let Some(k) = pptr(&d, end + 60).and_then(mesh_key) {
                                out.insert(k, m);
                            }
                        }
                    }
                }
            }
            for (go, k) in go_mesh {
                if let Some(m) = go_mats.remove(&go) {
                    out.entry(k).or_insert(m);
                }
            }
            out
        })
    }

    /// Diffuse image of each submesh (= material slot) of Mesh `entry`: from the renderer that draws it,
    /// in this file or, for sharedassetsN.assets, in levelN. Material: main texture, else its base color.
    pub fn mesh_textures(&self, entry: u32) -> Vec<Option<Image>> {
        let Some(o) = self.objects.get(entry as usize) else { return vec![] };
        let key = (self.file_name(), o.path_id);
        let level = key.0.strip_prefix("sharedassets").and_then(|n| n.strip_suffix(".assets")).map(|n| format!("level{n}"));
        let found = self.mesh_materials().get(&key).map(|m| (None, m.clone())).or_else(|| {
            let a = self.sibling(&level?)?;
            let m = a.mesh_materials().get(&key)?.clone();
            Some((Some(a), m))
        });
        let Some((holder, mats)) = found else { return vec![] };
        let holder: &Archive = holder.as_deref().unwrap_or(self);
        mats.iter().map(|&(fid, pid)| holder.with_obj(fid, pid, |a, i| a.material_image(i))).collect()
    }

    /// Mesh `entry` -> `dst` (.obj) + .mtl + one PNG per textured material slot, next to it.
    pub fn save_obj_textured(&self, entry: u32, dst: &Path) -> Res<()> {
        use std::fmt::Write;
        let m = self.mesh(entry)?;
        let tex = self.mesh_textures(entry);
        let stem = dst.file_stem().ok_or("bad output path")?.to_string_lossy().into_owned();
        let dir = dst.parent().ok_or("bad output path")?;
        std::fs::create_dir_all(dir)?;
        let slots = tex.len().max(m.sections.iter().map(|s| s.0 as usize + 1).max().unwrap_or(1));
        let mut mtl = String::new();
        for k in 0..slots {
            let _ = writeln!(mtl, "newmtl m{k}\nKd 1 1 1\nKs 0 0 0");
            if let Some(Some(img)) = tex.get(k) {
                let png = format!("{stem}_m{k}.png");
                std::fs::write(dir.join(&png), crate::texture::to_png(img)?)?;
                let _ = writeln!(mtl, "map_Kd {png}");
            }
        }
        // usemtl at each submesh's first triangle (to_obj writes one "f" line per triangle, in order)
        let mut obj = format!("mtllib {stem}.mtl\n");
        let (mut tri, mut sec) = (0u32, m.sections.iter().peekable());
        for line in m.to_obj(&stem).lines() {
            if line.starts_with("f ") {
                while let Some(s) = sec.next_if(|s| s.1 / 3 <= tri) {
                    let _ = writeln!(obj, "usemtl m{}", s.0);
                }
                tri += 1;
            }
            obj.push_str(line);
            obj.push('\n');
        }
        std::fs::write(dir.join(format!("{stem}.mtl")), mtl)?;
        Ok(std::fs::write(dst, obj)?)
    }

    /// Main texture of Material `entry` (Unity 6000 layout), or a solid image of its base color.
    fn material_image(&self, entry: u32) -> Option<Image> {
        let d = self.object(entry).ok()?;
        let mut r = R(&d, 0);
        let skip_strings = |r: &mut R| -> Res<()> {
            for _ in 0..r.u32()? {
                r.str()?;
            }
            Ok(())
        };
        r.str().ok()?;
        r.take(12).ok()?; // m_Shader
        skip_strings(&mut r).ok()?; // m_ValidKeywords
        skip_strings(&mut r).ok()?; // m_InvalidKeywords
        r.take(6).ok()?; // m_LightmapFlags, m_EnableInstancingVariants, m_DoubleSidedGI
        r.align();
        r.i32().ok()?; // m_CustomRenderQueue
        for _ in 0..r.u32().ok()? {
            r.str().ok()?; // stringTagMap
            r.str().ok()?;
        }
        skip_strings(&mut r).ok()?; // disabledShaderPasses
        let mut tex = vec![];
        for _ in 0..r.u32().ok()? {
            let name = r.str().ok()?.to_lowercase();
            let (fid, pid) = (r.i32().ok()?, r.u64().ok()? as i64);
            r.take(16).ok()?; // scale, offset
            if pid != 0 {
                tex.push((name, fid, pid));
            }
        }
        let mut colors = vec![];
        for (kind, size) in [(0, 4), (1, 4), (2, 16)] {
            for _ in 0..r.u32().ok()? {
                let name = r.str().ok()?.to_lowercase();
                let v = r.take(size).ok()?;
                if kind == 2 {
                    let f = |k: usize| f32::from_le_bytes(v[k * 4..k * 4 + 4].try_into().unwrap());
                    colors.push((name, [f(0), f(1), f(2)]));
                }
            }
        }
        // ponytail: shader property-name heuristic (built-in, URP, HDRP and common custom names)
        let bad = ["normal", "bump", "mask", "detail", "emiss", "metal", "occlusion", "height", "parallax", "noise", "rough", "spec"];
        let rank = |n: &str| ["_maintex", "_basemap", "_basecolormap", "_albedo", "_diffuse", "_basecolortexture"].iter().position(|k| n == *k)
            .or_else(|| (["main", "base", "albedo", "diffuse", "color", "tex"].iter().any(|k| n.contains(k)) && !bad.iter().any(|k| n.contains(k))).then_some(10));
        let mut cand: Vec<_> = tex.iter().filter_map(|t| Some((rank(&t.0)?, t))).collect();
        cand.sort_by_key(|c| c.0);
        for (_, &(_, fid, pid)) in cand {
            if let Some(img) = self.with_obj(fid, pid, |a, i| (a.class(i) == 28).then(|| a.texture(i).ok()).flatten()) {
                return Some(img);
            }
        }
        let c = ["_basecolor", "_color", "_maincolor", "_tint"].iter().find_map(|k| colors.iter().find(|c| c.0 == *k)).map(|c| c.1)?;
        let s = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8; // material colors are stored as authored (sRGB)
        Some(Image { width: 4, height: 4, rgba: [s(c[0]), s(c[1]), s(c[2]), 255].repeat(16) })
    }

    fn read_at(&self, f: &mut File, off: u64, n: usize) -> Res<Vec<u8>> {
        let mut buf = vec![0u8; n];
        f.seek(SeekFrom::Start(off))?;
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Raw serialized bytes of object `entry`.
    pub fn object(&self, entry: u32) -> Res<Vec<u8>> {
        let o = self.objects.get(entry as usize).ok_or("unity: bad entry")?;
        self.read_at(&mut File::open(&self.path)?, o.offset, o.size as usize)
    }

    pub fn class(&self, entry: u32) -> i32 {
        self.objects.get(entry as usize).map_or(0, |o| o.class_id)
    }

    /// Bytes stored outside the object (.resS / .resource next to the .assets file).
    fn stream(&self, s: &Stream) -> Res<Vec<u8>> {
        // "sharedassets0.assets.resS" or "archive:/CAB-x/CAB-x.resS": only the file name matters
        let name = s.2.rsplit('/').next().unwrap_or(&s.2);
        if name.is_empty() || name == ".." {
            return Err("unity: bad stream path".into());
        }
        let p = self.path.with_file_name(name);
        let mut f = File::open(&p).map_err(|e| format!("unity: {}: {e}", p.display()))?;
        self.read_at(&mut f, s.0, s.1 as usize)
    }

    /// Converted file contents for `entry`, matching the extension it is listed with.
    pub fn export(&self, entry: u32) -> Res<Vec<u8>> {
        let d = self.object(entry)?;
        match self.class(entry) {
            28 => crate::texture::to_png(&self.texture(entry)?),
            43 => Ok(self.mesh(entry)?.to_obj("mesh").into_bytes()),
            49 => {
                let mut r = R(&d, 0);
                r.str()?;
                Ok(r.arr(1)?.to_vec())
            }
            83 => self.audio(&d),
            _ => Ok(d),
        }
    }

    pub fn texture(&self, entry: u32) -> Res<Image> {
        let d = self.object(entry)?;
        // Unity 6 dropped m_ForcedFallbackFormat + m_DownscaleFallback; the exact-size check picks the layout
        let t = parse_texture(&d, false).or_else(|e| parse_texture(&d, true).map_err(|_| e))?;
        let data = if t.image.is_empty() { self.stream(&t.stream)? } else { t.image };
        let (w, h) = (t.width as usize, t.height as usize);
        let mut rgba = decode_pixels(t.format, &data, w, h)?;
        // Unity stores rows bottom-up
        let row = w * 4;
        for y in 0..h / 2 {
            let (a, b) = rgba.split_at_mut((h - 1 - y) * row);
            a[y * row..y * row + row].swap_with_slice(&mut b[..row]);
        }
        Ok(Image { width: w as u32, height: h as u32, rgba })
    }

    pub fn mesh(&self, entry: u32) -> Res<Mesh> {
        let d = self.object(entry)?;
        let m = parse_mesh(&d)?;
        let vdata = if m.stream.1 > 0 { self.stream(&m.stream)? } else { m.vertex_data };
        build_mesh(&m.submeshes, m.index_u32, &m.indices, m.vertex_count, &m.channels, &vdata)
    }

    /// Short description of an AudioClip ("ADPCM  ·  2 ch  ·  44100 Hz  ·  1.22 s").
    pub fn audio_info(&self, entry: u32) -> Res<String> {
        let c = audio_header(&self.object(entry)?)?;
        let codec = ["PCM", "Vorbis", "ADPCM", "MP3", "VAG", "HEVAG", "XMA", "AAC", "GCADPCM", "ATRAC9"];
        let codec = codec.get(c.compression as usize).copied().unwrap_or("?");
        Ok(format!("{codec}  ·  {} ch  ·  {} Hz  ·  {:.2} s", c.channels, c.frequency, c.length))
    }

    fn audio(&self, d: &[u8]) -> Res<Vec<u8>> {
        let c = audio_header(d)?;
        let fsb = self.stream(&c.resource)?;
        if c.compression != 0 && c.compression != 2 {
            return Ok(fsb); // ponytail: Vorbis FSB needs vgmstream's setup-header table to rebuild .ogg
        }
        fsb_to_wav(&fsb)
    }
}

struct Tex {
    width: u32,
    height: u32,
    format: i32,
    image: Vec<u8>,
    stream: Stream,
}

fn parse_texture(d: &[u8], legacy: bool) -> Res<Tex> {
    let mut r = R(d, 0);
    r.str()?; // m_Name
    if legacy {
        r.take(5)?; // m_ForcedFallbackFormat, m_DownscaleFallback
    }
    r.u8()?; // m_IsAlphaChannelOptional
    r.align();
    let (width, height) = (r.u32()?, r.u32()?);
    r.take(8)?; // m_CompleteImageSize, m_MipsStripped
    let format = r.i32()?;
    r.take(7)?; // m_MipCount, m_IsReadable, m_IsPreProcessed, m_IgnoreMipmapLimit
    r.align();
    r.str()?; // m_MipmapLimitGroupName
    r.u8()?; // m_StreamingMipmaps
    r.align();
    r.take(4 * 3 + 24 + 4 * 2)?; // priority, image count, dimension, GLTextureSettings, lightmap fmt, color space
    r.arr(1)?; // m_PlatformBlob
    let image = r.arr(1)?.to_vec();
    let stream = Stream(r.u64()?, r.u32()? as u64, r.str()?);
    if r.1 == d.len() && (width == 0 || height == 0) {
        return Err("unity: empty 0x0 texture (e.g. a dynamic font atlas)".into());
    }
    if r.1 != d.len() || width > 16384 || height > 16384 {
        return Err("unity: Texture2D layout not understood (unsupported Unity version?)".into());
    }
    Ok(Tex { width, height, format, image, stream })
}

/// (bytes per block, block width, block height) of Unity TextureFormat `f`; None for crunched.
fn block_info(f: i32) -> Option<(usize, usize, usize)> {
    Some(match f {
        1 | 63 => (1, 1, 1),
        2 | 7 | 9 | 13 | 15 | 62 => (2, 1, 1),
        3 => (3, 1, 1),
        4 | 5 | 14 | 16 | 18 => (4, 1, 1),
        17 => (8, 1, 1),
        20 => (16, 1, 1),
        10 | 26 | 34 | 45 | 46 => (8, 4, 4),
        12 | 24 | 25 | 27 | 47 => (16, 4, 4),
        48..=59 => {
            let b = [4, 5, 6, 8, 10, 12][((f - 48) % 6) as usize];
            (16, b, b)
        }
        _ => return None,
    })
}

/// Mip 0 of `data` in Unity TextureFormat `f` -> RGBA8 (rows as stored, bottom-up).
fn decode_pixels(f: i32, data: &[u8], w: usize, h: usize) -> Res<Vec<u8>> {
    use crate::texture::half_to_f32;
    let crunched = matches!(f, 28 | 29 | 64 | 65);
    let mip = match block_info(f) {
        Some((bb, bw, bh)) => {
            let n = w.div_ceil(bw) * h.div_ceil(bh) * bb;
            data.get(..n).ok_or("unity: texture data truncated")?
        }
        None if crunched => data,
        None => return Err(format!("unity: TextureFormat {f} not supported").into()),
    };
    let u16s = || mip.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]));
    let f32s = || mip.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()));
    let unorm = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let n4 = |v: u16, s: u32| ((v >> s) & 15) as u8 * 17;
    let rgba: Vec<u8> = match f {
        1 => mip.iter().flat_map(|&a| [255, 255, 255, a]).collect(),
        63 => mip.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        2 => u16s().flat_map(|v| [n4(v, 8), n4(v, 4), n4(v, 0), n4(v, 12)]).collect(),
        13 => u16s().flat_map(|v| [n4(v, 12), n4(v, 8), n4(v, 4), n4(v, 0)]).collect(),
        7 => u16s()
            .flat_map(|v| [((v >> 11) * 255 / 31) as u8, (((v >> 5) & 63) * 255 / 63) as u8, ((v & 31) * 255 / 31) as u8, 255])
            .collect(),
        9 => u16s().flat_map(|v| [(v >> 8) as u8, (v >> 8) as u8, (v >> 8) as u8, 255]).collect(),
        62 => mip.chunks_exact(2).flat_map(|c| [c[0], c[1], 0, 255]).collect(),
        3 => mip.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        4 => mip.to_vec(),
        5 => mip.chunks_exact(4).flat_map(|c| [c[1], c[2], c[3], c[0]]).collect(),
        14 => mip.chunks_exact(4).flat_map(|c| [c[2], c[1], c[0], c[3]]).collect(),
        15 => u16s().flat_map(|v| { let g = unorm(half_to_f32(v)); [g, g, g, 255] }).collect(),
        16 => u16s().map(|v| unorm(half_to_f32(v))).collect::<Vec<_>>().chunks_exact(2).flat_map(|c| [c[0], c[1], 0, 255]).collect(),
        17 => u16s().map(|v| unorm(half_to_f32(v))).collect(),
        18 => f32s().flat_map(|v| { let g = unorm(v); [g, g, g, 255] }).collect(),
        20 => f32s().map(unorm).collect(),
        _ => {
            use texture2ddecoder as t;
            let mut buf = vec![0u32; w * h];
            match f {
                10 => t::decode_bc1(mip, w, h, &mut buf),
                12 => t::decode_bc3(mip, w, h, &mut buf),
                24 => t::decode_bc6_unsigned(mip, w, h, &mut buf),
                25 => t::decode_bc7(mip, w, h, &mut buf),
                26 => t::decode_bc4(mip, w, h, &mut buf),
                27 => t::decode_bc5(mip, w, h, &mut buf),
                34 => t::decode_etc1(mip, w, h, &mut buf),
                45 => t::decode_etc2_rgb(mip, w, h, &mut buf),
                46 => t::decode_etc2_rgba1(mip, w, h, &mut buf),
                47 => t::decode_etc2_rgba8(mip, w, h, &mut buf),
                48..=59 => {
                    let (_, bw, bh) = block_info(f).unwrap();
                    t::decode_astc(mip, w, h, bw, bh, &mut buf)
                }
                _ => t::decode_unity_crunch(mip, w, h, &mut buf).or_else(|_| t::decode_crunch(mip, w, h, &mut buf)),
            }
            .map_err(|e| format!("unity: TextureFormat {f} decode failed: {e}"))?;
            // texture2ddecoder packs little-endian BGRA
            buf.iter()
                .flat_map(|v| {
                    let [b, g, r, a] = v.to_le_bytes();
                    if f == 26 { [r, r, r, 255] } else { [r, g, b, a] }
                })
                .collect()
        }
    };
    if rgba.len() != w * h * 4 {
        return Err(format!("unity: TextureFormat {f} {w}x{h}: wrong data size").into());
    }
    Ok(rgba)
}

struct MeshData {
    submeshes: Vec<(u32, u32, u32, u32)>, // firstByte, indexCount, topology, baseVertex
    index_u32: bool,
    indices: Vec<u8>,
    vertex_count: u32,
    channels: Vec<[u8; 4]>, // stream, offset, format, dimension
    vertex_data: Vec<u8>,
    stream: Stream,
}

fn parse_mesh(d: &[u8]) -> Res<MeshData> {
    let mut r = R(d, 0);
    r.str()?;
    let submeshes = r
        .arr(48)?
        .chunks_exact(48)
        .map(|s| {
            let u = |o: usize| u32::from_le_bytes(s[o..o + 4].try_into().unwrap());
            (u(0), u(4), u(8), u(12))
        })
        .collect();
    // m_Shapes: vertices, shapes (10 bytes + align each), channels, fullWeights
    r.arr(40)?;
    r.arr(12)?;
    for _ in 0..r.u32()? {
        r.str()?;
        r.take(12)?;
    }
    r.arr(4)?;
    r.arr(64)?; // m_BindPose
    r.arr(4)?; // m_BoneNameHashes
    r.u32()?; // m_RootBoneNameHash
    r.arr(24)?; // m_BonesAABB
    r.arr(4)?; // m_VariableBoneCountWeights
    let compression = r.u8()?;
    r.take(3)?;
    r.align();
    if compression != 0 {
        // ponytail: none seen in Unity 6 players; port AssetStudio's CompressedMesh unpacking if needed
        return Err("unity: compressed Mesh not supported".into());
    }
    let index_u32 = r.u32()? == 1;
    let indices = r.arr(1)?.to_vec();
    let vertex_count = r.u32()?;
    let channels = r.arr(4)?.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
    let vertex_data = r.arr(1)?.to_vec();
    // m_CompressedMesh: 10 PackedBitVectors (4 with range/start), m_UVInfo
    for ranged in [true, true, true, true, false, false, false, true, false, false] {
        r.u32()?;
        if ranged {
            r.take(8)?;
        }
        r.arr(1)?;
        r.u8()?;
        r.align();
    }
    r.u32()?;
    r.take(24 + 8)?; // m_LocalAABB, m_MeshUsageFlags, m_CookingOptions
    r.arr(1)?;
    r.arr(1)?; // baked collision meshes
    r.take(8)?; // m_MeshMetrics
    let stream = Stream(r.u64()?, r.u32()? as u64, r.str()?);
    Ok(MeshData { submeshes, index_u32, indices, vertex_count, channels, vertex_data, stream })
}

/// Vertex channel `ch` as floats (`dim` per vertex), from Unity 2019+ VertexData.
fn channel(chs: &[[u8; 4]], data: &[u8], n: usize, ch: usize) -> Res<Option<(Vec<f32>, usize)>> {
    const SIZE: [usize; 12] = [4, 2, 1, 1, 2, 2, 1, 1, 2, 2, 4, 4];
    let Some(&[stream, offset, format, dim]) = chs.get(ch) else { return Ok(None) };
    let dim = (dim & 15) as usize;
    if dim == 0 {
        return Ok(None);
    }
    let size = |c: &[u8; 4]| SIZE.get(c[2] as usize).copied().unwrap_or(0) * (c[3] & 15) as usize;
    let stride = |s: u8| chs.iter().filter(|c| c[0] == s).map(size).sum::<usize>();
    // streams are stored one after another, each 16-byte aligned
    let start: usize = (0..stream).map(|s| (stride(s) * n).next_multiple_of(16)).sum();
    let (stride, es) = (stride(stream), *SIZE.get(format as usize).ok_or("unity: bad vertex format")?);
    let mut out = Vec::with_capacity(n * dim);
    for v in 0..n {
        for k in 0..dim {
            let o = start + v * stride + offset as usize + k * es;
            let b = data.get(o..o + es).ok_or("unity: vertex data truncated")?;
            let u = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
            out.push(match format {
                0 => f32::from_le_bytes(b.try_into()?),
                1 => crate::texture::half_to_f32(u(b)),
                2 => b[0] as f32 / 255.0,
                3 => (b[0] as i8 as f32 / 127.0).max(-1.0),
                4 => u(b) as f32 / 65535.0,
                5 => (u(b) as i16 as f32 / 32767.0).max(-1.0),
                _ => return Err(format!("unity: vertex format {format} not supported for channel {ch}").into()),
            });
        }
    }
    Ok(Some((out, dim)))
}

/// Unity (Y-up, left-handed, meters) -> mesh::Mesh (UE space: Z-up, cm), laid out so to_obj writes
/// Unity's (-x, y, z) like AssetStudio/UnityPy.
fn build_mesh(subs: &[(u32, u32, u32, u32)], u32i: bool, ib: &[u8], vc: u32, chs: &[[u8; 4]], vd: &[u8]) -> Res<Mesh> {
    let n = vc as usize;
    let (pos, pd) = channel(chs, vd, n, 0)?.ok_or("unity: mesh has no positions")?;
    let v3 = |a: &[f32], d: usize, i: usize, s: f32| [-a[i * d] * s, a[i * d + 2] * s, a[i * d + 1] * s];
    let positions = (0..n).map(|i| v3(&pos, pd, i, 100.0)).collect();
    let normals = match channel(chs, vd, n, 1)? {
        Some((a, d)) if d >= 3 => (0..n).map(|i| v3(&a, d, i, 1.0)).collect(),
        _ => vec![],
    };
    let uvs = match channel(chs, vd, n, 4)? {
        Some((a, d)) if d >= 2 => (0..n).map(|i| [a[i * d], 1.0 - a[i * d + 1]]).collect(),
        _ => vec![],
    };
    let idx: Vec<u32> = if u32i {
        ib.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    } else {
        ib.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect()
    };
    let (mut indices, mut sections) = (vec![], vec![]);
    for (slot, &(first_byte, count, topology, base)) in subs.iter().enumerate() {
        if topology != 0 {
            continue; // ponytail: triangles only; quads/lines/strips are rare in shipped meshes
        }
        let first = (first_byte / if u32i { 4 } else { 2 }) as usize;
        let s = idx.get(first..first + count as usize).ok_or("unity: submesh out of index range")?;
        sections.push((slot as u32, indices.len() as u32, count / 3));
        // the X mirror flips handedness, so reverse each triangle to keep faces agreeing with normals
        for t in s.chunks_exact(3) {
            for i in [t[0], t[2], t[1]] {
                let i = i + base;
                if i >= vc {
                    return Err("unity: index out of vertex range".into());
                }
                indices.push(i);
            }
        }
    }
    Ok(Mesh { positions, normals, uvs, indices, sections })
}

struct Audio {
    channels: u32,
    frequency: u32,
    length: f32,
    resource: Stream,
    compression: i32,
}

fn audio_header(d: &[u8]) -> Res<Audio> {
    let mut r = R(d, 0);
    r.str()?;
    r.i32()?; // m_LoadType
    let (channels, frequency) = (r.u32()?, r.u32()?);
    r.i32()?; // m_BitsPerSample
    let length = r.f32()?;
    r.take(2)?; // m_IsTrackerFormat, m_Ambisonic
    r.align();
    r.i32()?; // m_SubsoundIndex
    r.take(3)?; // preload, load in background, legacy 3D
    r.align();
    let src = r.str()?;
    let resource = Stream(r.u64()?, r.u64()?, src);
    Ok(Audio { channels, frequency, length, resource, compression: r.i32()? })
}

/// FSB5 with one PCM8/PCM16/IMA-ADPCM sample -> 16-bit PCM WAV.
fn fsb_to_wav(fsb: &[u8]) -> Res<Vec<u8>> {
    if fsb.get(..4) != Some(b"FSB5") || fsb.len() < 0x48 {
        return Err("unity: audio is not FSB5".into());
    }
    let u = |o: usize| u32::from_le_bytes(fsb[o..o + 4].try_into().unwrap());
    let (ver, shs, nts, ds, mode) = (u(4), u(12) as usize, u(16) as usize, u(20) as usize, u(24));
    let hs = if ver == 0 { 0x40 } else { 0x3C };
    let sh = u64::from_le_bytes(fsb[hs..hs + 8].try_into()?);
    let channels = (((sh >> 5) & 3) + 1) as usize;
    let rate = [4000, 8000, 11000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000];
    let mut rate = *rate.get(((sh >> 1) & 15) as usize).unwrap_or(&44100);
    let samples = ((sh >> 34) & 0x3FFF_FFFF) as usize;
    // extra chunks may override channels (1) / frequency (2)
    let mut o = hs + 8;
    let mut more = sh & 1 != 0;
    let mut channels = channels;
    while more {
        let c = u(o);
        let (size, kind) = (((c >> 1) & 0xFF_FFFF) as usize, c >> 25);
        match kind {
            1 => channels = fsb[o + 4] as usize,
            2 => rate = u(o + 4),
            _ => {}
        }
        more = c & 1 != 0;
        o += 4 + size;
    }
    let start = hs + shs + nts;
    let data = fsb.get(start..start + ds).ok_or("unity: FSB truncated")?;
    let pcm: Vec<i16> = match mode {
        1 => data.iter().map(|&b| (b as i8 as i16) << 8).collect(),
        2 => data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect(),
        7 => ima_adpcm(data, channels),
        m => return Err(format!("unity: FSB5 codec {m} not supported").into()),
    };
    let pcm = &pcm[..(samples * channels).min(pcm.len())];
    Ok(wav(pcm, channels as u16, rate))
}

/// FMOD/Xbox IMA ADPCM: blocks of 36 bytes per channel (4-byte header, then 4-byte words interleaved
/// per channel), 64 samples per channel each: the header sample + 63 nibbles.
fn ima_adpcm(data: &[u8], ch: usize) -> Vec<i16> {
    const STEP: [i32; 89] = [
        7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157,
        173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707,
        1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635,
        13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
    ];
    const IDX: [i32; 8] = [-1, -1, -1, -1, 2, 4, 6, 8];
    let mut out = Vec::new();
    for block in data.chunks_exact(36 * ch) {
        let mut frame = vec![0i16; 64 * ch];
        for c in 0..ch {
            let h = &block[c * 4..c * 4 + 4];
            let (mut hist, mut idx) = (i16::from_le_bytes([h[0], h[1]]) as i32, (h[2] as i32).min(88));
            frame[c] = hist as i16;
            for k in 0..63 {
                let byte = block[4 * ch + (k / 8) * 4 * ch + c * 4 + (k % 8) / 2];
                let nib = (if k % 2 == 0 { byte } else { byte >> 4 } & 15) as i32;
                let step = STEP[idx as usize];
                let mut diff = step >> 3;
                if nib & 1 != 0 { diff += step >> 2; }
                if nib & 2 != 0 { diff += step >> 1; }
                if nib & 4 != 0 { diff += step; }
                hist = if nib & 8 != 0 { hist - diff } else { hist + diff }.clamp(-32768, 32767);
                idx = (idx + IDX[(nib & 7) as usize]).clamp(0, 88);
                frame[(k + 1) * ch + c] = hist as i16;
            }
        }
        out.extend(frame);
    }
    out
}

fn wav(pcm: &[i16], ch: u16, rate: u32) -> Vec<u8> {
    let n = (pcm.len() * 2) as u32;
    let mut w = Vec::with_capacity(44 + n as usize);
    w.extend(b"RIFF");
    w.extend((36 + n).to_le_bytes());
    w.extend(b"WAVEfmt ");
    w.extend(16u32.to_le_bytes());
    w.extend(1u16.to_le_bytes());
    w.extend(ch.to_le_bytes());
    w.extend(rate.to_le_bytes());
    w.extend((rate * ch as u32 * 2).to_le_bytes());
    w.extend((ch * 2).to_le_bytes());
    w.extend(16u16.to_le_bytes());
    w.extend(b"data");
    w.extend(n.to_le_bytes());
    w.extend(pcm.iter().flat_map(|s| s.to_le_bytes()));
    w
}

#[cfg(test)]
mod tests {
    /// Builds a format-22 SerializedFile holding one 2x2 RGBA32 Texture2D and reads it back.
    #[test]
    fn texture_roundtrip() {
        let mut obj = Vec::new();
        let s = |o: &mut Vec<u8>, t: &[u8]| {
            o.extend((t.len() as u32).to_le_bytes());
            o.extend(t);
            o.resize(o.len().next_multiple_of(4), 0);
        };
        s(&mut obj, b"tex");
        obj.extend([0, 0, 0, 0]); // alpha optional + align
        for v in [2u32, 2, 16, 0, 4, 1] { obj.extend(v.to_le_bytes()); } // w h size stripped fmt=RGBA32 mips
        obj.extend([0, 0, 0, 0]); // readable, preprocessed, ignore limit, align
        s(&mut obj, b"");
        obj.extend([0, 0, 0, 0]); // streaming mips + align
        obj.extend([0u8; 12 + 24 + 8]);
        s(&mut obj, b""); // platform blob
        let px: Vec<u8> = (0..16).collect(); // rows bottom-up: row0 = bytes 0..8
        s(&mut obj, &px);
        obj.extend([0u8; 12]);
        s(&mut obj, b"");

        let mut meta = b"6000.3.17f1\0".to_vec();
        meta.extend(19u32.to_le_bytes());
        meta.push(0); // no type tree
        meta.extend(1u32.to_le_bytes());
        meta.extend(28i32.to_le_bytes());
        meta.extend([0, 0xff, 0xff]);
        meta.extend([0u8; 16]);
        meta.extend(1u32.to_le_bytes());
        meta.resize(meta.len().next_multiple_of(4), 0);
        meta.extend(7u64.to_le_bytes()); // path id
        meta.extend(0u64.to_le_bytes());
        meta.extend((obj.len() as u32).to_le_bytes());
        meta.extend(0u32.to_le_bytes());
        let data_off = (48 + meta.len()).next_multiple_of(16);
        let mut f = vec![0u8; 8];
        f.extend(22u32.to_be_bytes());
        f.extend([0u8; 8]);
        f.extend((meta.len() as u32).to_be_bytes());
        f.extend(((data_off + obj.len()) as u64).to_be_bytes());
        f.extend((data_off as u64).to_be_bytes());
        f.extend(0u64.to_be_bytes());
        f.extend(&meta);
        f.resize(data_off, 0);
        f.extend(&obj);

        let p = std::env::temp_dir().join(format!("unity_test_{}.assets", std::process::id()));
        std::fs::write(&p, &f).unwrap();
        assert!(super::sniff(&p));
        let (a, files) = super::Archive::open(&p).unwrap();
        let img = a.texture(0);
        std::fs::remove_file(&p).unwrap();
        assert_eq!(files, vec![("Texture2D/tex.png".to_string(), 0)]);
        assert_eq!(a.unity, "6000.3.17f1");
        let img = img.unwrap();
        assert_eq!((img.width, img.height), (2, 2));
        assert_eq!(&img.rgba[..8], &px[8..16]); // flipped: top row = last stored row
    }

    /// Material/texture coverage of every Mesh: `UNITY_TEST_ASSETS=/path/x.assets cargo test unity -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn mesh_texture_coverage() {
        let p = std::path::PathBuf::from(std::env::var("UNITY_TEST_ASSETS").expect("set UNITY_TEST_ASSETS"));
        let (a, files) = super::Archive::open(&p).unwrap();
        let t = std::time::Instant::now();
        let (mut meshes, mut with_mat, mut textured, mut colored, mut slots) = (0, 0, 0, 0, 0);
        for (path, e) in files.iter().filter(|f| f.0.starts_with("Mesh/")) {
            meshes += 1;
            let tex = a.mesh_textures(*e);
            slots += tex.len();
            with_mat += !tex.is_empty() as usize;
            let big = tex.iter().flatten().filter(|i| i.width > 4).count();
            textured += (big > 0) as usize;
            colored += (big == 0 && tex.iter().any(|t| t.is_some())) as usize;
            if meshes <= 3 {
                println!("{path}: {:?}", tex.iter().map(|t| t.as_ref().map(|i| (i.width, i.height))).collect::<Vec<_>>());
            }
        }
        println!("{meshes} meshes: {with_mat} with materials ({slots} slots), {textured} textured, {colored} color only ({:?})", t.elapsed());
        assert!(with_mat > 0);
    }

    #[test]
    fn adpcm_to_wav() {
        // mono block: header sample 1000, index 0, positive nibbles (4) -> rising samples
        let mut b = vec![0xE8, 0x03, 0, 0];
        b.extend([0x44u8; 32]);
        let pcm = super::ima_adpcm(&b, 1);
        assert_eq!(pcm.len(), 64);
        assert_eq!(pcm[0], 1000);
        assert!(pcm[63] > 1000);
        let w = super::wav(&pcm, 1, 44100);
        assert_eq!((&w[..4], w.len()), (&b"RIFF"[..], 44 + 128));
    }
}
