//! .usmap mappings + unversioned property deserialization (reference: CUE4Parse UsmapParser,
//! FUnversionedHeader/FIterator, *Property.cs, FScriptStruct).
use crate::global::Resolver;
use crate::zen::ZenPackage;
use crate::Res;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// usmap property type tree (EPropertyType).
#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    Byte,
    Bool,
    Int8,
    Int16,
    Int,
    Int64,
    UInt16,
    UInt32,
    UInt64,
    Float,
    Double,
    Name,
    Str,
    Text,
    Object, // also Class, Interface, WeakObject
    SoftObject, // also SoftClass, AssetObject
    Lazy,
    Delegate,
    MulticastDelegate,
    FieldPath,
    Struct(String),
    Enum(Box<Ty>, String),
    Array(Box<Ty>),
    Set(Box<Ty>),
    Map(Box<Ty>, Box<Ty>),
    Optional(Box<Ty>),
    Unknown(u8),
}

pub struct Prop {
    pub name: String, // "Foo", or "Foo[i]" for fixed-size C arrays
    pub ty: Ty,
}

pub struct Struct {
    pub super_name: Option<String>,
    /// schema slots of this struct alone (super's slots follow, see `prop`)
    pub props: Vec<Option<Prop>>,
}

#[derive(Default)]
pub struct Mappings {
    pub file: PathBuf,
    pub enums: HashMap<String, Vec<(i64, String)>>,
    pub structs: HashMap<String, Struct>,
}

struct Rd<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let s = self.p.checked_add(n).and_then(|e| self.d.get(self.p..e));
        let s = s.ok_or_else(|| format!("read of {n} bytes at 0x{:x} past end (0x{:x})", self.p, self.d.len()))?;
        self.p += n;
        Ok(s)
    }
    fn left(&self) -> usize {
        self.d.len().saturating_sub(self.p)
    }
    fn arr<const N: usize>(&mut self) -> Res<[u8; N]> {
        Ok(self.take(N)?.try_into()?)
    }
    fn u8(&mut self) -> Res<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Res<u16> {
        Ok(u16::from_le_bytes(self.arr()?))
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn i32(&mut self) -> Res<i32> {
        Ok(i32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn i64(&mut self) -> Res<i64> {
        Ok(i64::from_le_bytes(self.arr()?))
    }
    fn f32(&mut self) -> Res<f64> {
        Ok(f32::from_le_bytes(self.arr()?) as f64)
    }
    fn f64(&mut self) -> Res<f64> {
        Ok(f64::from_le_bytes(self.arr()?))
    }
    /// i32 count, refusing counts that can't fit in the remaining bytes (min `elem` bytes each).
    fn count(&mut self, elem: usize) -> Res<usize> {
        let n = self.i32()?;
        if n < 0 || (n as usize).saturating_mul(elem.max(1)) > self.left() {
            return Err(format!("implausible count {n} at 0x{:x} ({} bytes left)", self.p - 4, self.left()).into());
        }
        Ok(n as usize)
    }
    /// UE bool (u32 0/1).
    fn bool32(&mut self) -> Res<bool> {
        match self.u32()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(format!("bad bool32 {v} at 0x{:x}", self.p - 4).into()),
        }
    }
    /// FString: i32 len (negative = UTF-16), null terminated.
    fn fstring(&mut self) -> Res<String> {
        let len = self.i32()?;
        if len >= 0 {
            let b = self.take(len as usize)?;
            Ok(String::from_utf8_lossy(b.strip_suffix(&[0]).unwrap_or(b)).into())
        } else {
            let n = (len as i64).unsigned_abs() as usize;
            let b = self.take(n.checked_mul(2).ok_or("bad string length")?)?;
            let w: Vec<u16> = b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            Ok(String::from_utf16_lossy(w.strip_suffix(&[0]).unwrap_or(&w)))
        }
    }
}

// ---------------------------------------------------------------- usmap

impl Mappings {
    pub fn load(usmap: &Path) -> Res<Mappings> {
        let mut m = Mappings::parse(&std::fs::read(usmap)?)?;
        m.file = usmap.to_path_buf();
        Ok(m)
    }

    /// Auto-detects mappings, newest file first in each place, searching in order:
    /// `<archive dir>/Mappings/`, next to the archive, then the per-game library
    /// `~/.ucas-explorer/mappings/<game>/` (filled by [`Mappings::remember`]).
    pub fn find(archive: &Path, game: Option<&str>) -> Option<PathBuf> {
        let dir = archive.parent().unwrap_or(Path::new("."));
        let lib = game.and_then(|g| Some(library_dir()?.join(g)));
        [Some(dir.join("Mappings")), Some(dir.to_path_buf()), lib].into_iter().flatten().find_map(|d| newest_usmap(&d))
    }

    /// Copies `usmap` into the per-game library so it auto-loads for this game from any archive
    /// location. Returns the library path (unchanged if the file already lives there).
    pub fn remember(usmap: &Path, game: &str) -> Res<PathBuf> {
        let dir = library_dir().ok_or("no HOME directory for the mappings library")?.join(game);
        let dst = dir.join(usmap.file_name().ok_or("usmap path has no file name")?);
        if usmap.canonicalize().ok() != dst.canonicalize().ok() {
            std::fs::create_dir_all(&dir)?;
            std::fs::copy(usmap, &dst)?;
        }
        Ok(dst)
    }
}

/// `~/.ucas-explorer/mappings` (per-game subfolders of .usmap files).
pub fn library_dir() -> Option<PathBuf> {
    Some(PathBuf::from(std::env::var_os("HOME")?).join(".ucas-explorer").join("mappings"))
}

/// Most recently modified `*.usmap` in `dir`.
fn newest_usmap(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("usmap")))
        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
}

/// Game name of an archive: the most common top-level folder of its files that isn't "Engine"
/// (e.g. "Palia" for "Palia/Content/..."). None for engine-only archives.
pub fn game_name(files: &[(String, u32)]) -> Option<String> {
    let mut count: HashMap<&str, usize> = HashMap::new();
    for (p, _) in files {
        let top = p.split('/').next().unwrap_or("");
        if !top.is_empty() && top != "Engine" {
            *count.entry(top).or_default() += 1;
        }
    }
    // ties broken by name so the result is deterministic
    count.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(a.0))).map(|(g, _)| g.to_string())
}

impl Mappings {

    pub fn parse(file: &[u8]) -> Res<Mappings> {
        let mut r = Rd { d: file, p: 0 };
        if r.u16()? != 0x30C4 {
            return Err("usmap: bad magic".into());
        }
        let ver = r.u8()?; // 0 Initial, 1 PackageVersioning, 2 LongFName, 3 LargeEnums, 4 ExplicitEnumValues, 5 EngineVersioning
        if ver > 5 {
            return Err(format!("usmap: unsupported version {ver}").into());
        }
        // ponytail: versions >= 1 and compressed files follow CUE4Parse but are untested (only a v0/None file at hand)
        if ver >= 1 && r.bool32()? {
            if ver >= 5 {
                r.take(2 + 2 + 2 + 4)?; // FEngineVersion major/minor/patch/changelist
                r.fstring()?; // branch
            }
            r.take(8)?; // FileVersionUE4/UE5
            let n = r.count(20)?;
            r.take(n * 20)?; // custom versions (guid, i32)
            r.u32()?; // NetCL
        }
        let method = r.u8()?;
        let (csize, dsize) = (r.u32()? as usize, r.u32()? as usize);
        let comp = r.take(csize)?;
        let data = match method {
            0 if csize == dsize => comp.to_vec(),
            0 => return Err("usmap: uncompressed but sizes differ".into()),
            1 => {
                let mut out = vec![0; dsize];
                oozextract::Extractor::new().read_from_slice(comp, &mut out)?;
                out
            }
            m => return Err(format!("usmap: compression {} not supported", if m == 2 { "Brotli" } else if m == 3 { "ZStandard" } else { "?" }).into()),
        };
        let mut r = Rd { d: &data, p: 0 };
        let n = r.count(1)?;
        let names = (0..n)
            .map(|_| {
                let len = if ver >= 2 { r.u16()? as usize } else { r.u8()? as usize };
                Ok(String::from_utf8_lossy(r.take(len)?).into_owned())
            })
            .collect::<Res<Vec<_>>>()?;
        let name = |r: &mut Rd| -> Res<Option<String>> {
            match r.i32()? {
                -1 => Ok(None),
                i => Ok(Some(names.get(i as usize).ok_or_else(|| format!("usmap: bad name index {i}"))?.clone())),
            }
        };
        let mut m = Mappings::default();
        for _ in 0..r.count(5)? {
            let ename = name(&mut r)?.ok_or("usmap: unnamed enum")?;
            let n = if ver >= 3 { r.u16()? as usize } else { r.u8()? as usize };
            let vals = (0..n)
                .map(|j| {
                    let v = if ver >= 4 { r.i64()? } else { j as i64 };
                    Ok((v, name(&mut r)?.unwrap_or_default()))
                })
                .collect::<Res<Vec<_>>>()?;
            m.enums.entry(ename).or_insert(vals); // duplicates: first wins (as CUE4Parse)
        }
        for _ in 0..r.count(12)? {
            let sname = name(&mut r)?.ok_or("usmap: unnamed struct")?;
            let super_name = name(&mut r)?;
            let count = r.u16()? as usize;
            let mut props: Vec<Option<Prop>> = (0..count).map(|_| None).collect();
            for _ in 0..r.u16()? {
                let (idx, dim) = (r.u16()? as usize, r.u8()? as usize);
                let pname = name(&mut r)?.unwrap_or_default();
                let ty = parse_ty(&mut r, &name, 0)?;
                for j in 0..dim {
                    let slot = props.get_mut(idx + j).ok_or_else(|| format!("usmap: {sname}.{pname} schema index {} >= {count}", idx + j))?;
                    let name = if dim > 1 { format!("{pname}[{j}]") } else { pname.clone() };
                    *slot = Some(Prop { name, ty: ty.clone() });
                }
            }
            m.structs.insert(sname, Struct { super_name, props });
        }
        Ok(m)
    }

    fn enum_name(&self, e: &str, v: i64) -> Res<String> {
        let Some(vals) = self.enums.get(e) else { return Ok(format!("{e}({v})")) };
        let n = vals.iter().find(|x| x.0 == v).ok_or_else(|| format!("enum {e} has no value {v} (mappings know {})", vals.len()))?;
        Ok(n.1.clone())
    }
}

fn parse_ty(r: &mut Rd, name: &dyn Fn(&mut Rd) -> Res<Option<String>>, depth: u32) -> Res<Ty> {
    if depth > 16 {
        return Err("usmap: property type nested too deep".into());
    }
    let inner = |r: &mut Rd| parse_ty(r, name, depth + 1).map(Box::new);
    Ok(match r.u8()? {
        0 => Ty::Byte,
        1 => Ty::Bool,
        2 => Ty::Int,
        3 => Ty::Float,
        4 | 12 | 14 | 31 => Ty::Object, // Object, Interface, WeakObject, Class
        5 => Ty::Name,
        6 => Ty::Delegate,
        7 => Ty::Double,
        8 => Ty::Array(inner(r)?),
        9 => Ty::Struct(name(r)?.unwrap_or_default()),
        10 | 29 | 30 => Ty::Str, // Str, Utf8Str, AnsiStr (ponytail: utf8/ansi read as FString)
        11 => Ty::Text,
        13 | 32 => Ty::MulticastDelegate,
        15 => Ty::Lazy,
        16 | 17 | 33 => Ty::SoftObject,
        18 => Ty::UInt64,
        19 => Ty::UInt32,
        20 => Ty::UInt16,
        21 => Ty::Int64,
        22 => Ty::Int16,
        23 => Ty::Int8,
        24 => {
            let k = inner(r)?;
            Ty::Map(k, inner(r)?)
        }
        25 => Ty::Set(inner(r)?),
        26 => {
            let i = inner(r)?;
            Ty::Enum(i, name(r)?.unwrap_or_default())
        }
        27 => Ty::FieldPath,
        28 => Ty::Optional(inner(r)?),
        t => Ty::Unknown(t),
    })
}

// ---------------------------------------------------------------- values

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String), // StrProperty / TextProperty display string
    Name(String),
    Enum(String),
    Object(String),     // resolved object path or "None"
    SoftObject(String), // "/Game/X.Y[:sub]"
    Struct(String, Vec<(String, Value)>), // struct type, fields
    Array(Vec<Value>),
    Map(Vec<(Value, Value)>),
}

struct Ctx<'a> {
    pkg: &'a ZenPackage,
    maps: &'a Mappings,
    res: &'a Resolver,
}

/// Prefixes an error with the property path segment it happened in.
fn at<T>(seg: impl std::fmt::Display, r: Res<T>) -> Res<T> {
    r.map_err(|e| {
        let e = e.to_string();
        let sep = if e.starts_with('[') || e.starts_with('.') { "" } else { ": " };
        format!("{seg}{sep}{e}").into()
    })
}

impl Ctx<'_> {
    fn name(&self, r: &mut Rd) -> Res<String> {
        let (i, n) = (r.u32()?, r.u32()?);
        let base = self.pkg.names.get(i as usize).ok_or_else(|| format!("FName index {i} out of range ({} names) at 0x{:x}", self.pkg.names.len(), r.p - 8))?;
        Ok(if n == 0 { base.clone() } else { format!("{base}_{}", n - 1) })
    }

    /// FPackageIndex: >0 export i-1, <0 import -i-1.
    fn object(&self, r: &mut Rd) -> Res<String> {
        let v = r.i32()?;
        let bad = || format!("object index {v} out of range at 0x{:x}", r.p - 4);
        Ok(match v {
            0 => "None".into(),
            v if v > 0 => {
                let i = v as usize - 1;
                self.pkg.exports.get(i).ok_or_else(bad)?;
                self.res.object_path(i as u64, self.pkg).unwrap_or_else(|| format!("export#{i}"))
            }
            v => {
                let imp = *self.pkg.imports.get((-(v as i64) - 1) as usize).ok_or_else(bad)?;
                if imp >> 62 == 3 {
                    return Ok("None".into());
                }
                let p = self.res.object_path(imp, self.pkg).unwrap_or_else(|| format!("import#{imp:x}"));
                pretty_import(&p)
            }
        })
    }

    /// FSoftObjectPath (UE5.1+, no soft path list in cooked zen): FTopLevelAssetPath + FString subpath.
    fn soft(&self, r: &mut Rd) -> Res<String> {
        let (pkg, asset, sub) = (self.name(r)?, self.name(r)?, r.fstring()?);
        Ok(match (pkg.as_str(), asset.as_str(), sub.as_str()) {
            ("None", _, _) => "None".into(),
            (_, "None", _) => pkg,
            (_, _, "") => format!("{pkg}.{asset}"),
            _ => format!("{pkg}.{asset}:{sub}"),
        })
    }

    fn text(&self, r: &mut Rd, depth: u32) -> Res<String> {
        if depth > 8 {
            return Err("FText nested too deep".into());
        }
        r.u32()?; // flags
        let hist = r.u8()? as i8;
        Ok(match hist {
            -1 => if r.bool32()? { r.fstring()? } else { String::new() },
            0 => {
                let (_ns, _key) = (r.fstring()?, r.fstring()?);
                r.fstring()?
            }
            1..=3 => {
                let fmt = self.text(r, depth + 1)?;
                for _ in 0..r.count(1)? {
                    if hist != 2 {
                        r.fstring()?; // argument name
                    }
                    self.format_arg(r, depth)?;
                }
                fmt
            }
            4..=6 => {
                if hist == 6 {
                    r.fstring()?; // currency code
                }
                let v = self.format_arg(r, depth)?;
                if r.bool32()? {
                    r.take(4 + 4 + 1 + 16)?; // FNumberFormattingOptions
                }
                r.fstring()?; // target culture
                v
            }
            10 => {
                let s = self.text(r, depth + 1)?;
                r.u8()?;
                s
            }
            11 => {
                let table = self.name(r)?;
                format!("{table}:{}", r.fstring()?)
            }
            h => return Err(format!("FText history type {h} not supported").into()),
        })
    }

    fn format_arg(&self, r: &mut Rd, depth: u32) -> Res<String> {
        Ok(match r.u8()? {
            0 => r.i64()?.to_string(),
            1 => r.u64()?.to_string(),
            2 => r.f32()?.to_string(),
            3 => r.f64()?.to_string(),
            4 => self.text(r, depth + 1)?,
            5 => r.u64()?.to_string(), // gender (as CUE4Parse)
            t => return Err(format!("FText argument type {t} not supported").into()),
        })
    }

    /// Unversioned property block of struct/class `sname` (+ its super chain).
    fn read_struct(&self, r: &mut Rd, sname: &str, depth: u32) -> Res<Vec<(String, Value)>> {
        if depth > 32 {
            return Err("structs nested too deep".into());
        }
        let mut chain = vec![];
        let mut s = Some(sname);
        while let Some(n) = s {
            let st = self.maps.structs.get(n).ok_or_else(|| format!("no mappings for struct {n}"))?;
            chain.push((n, st));
            s = st.super_name.as_deref();
            if chain.len() > 64 {
                return Err(format!("super chain of {sname} loops").into());
            }
        }
        let schema: usize = chain.iter().map(|c| c.1.props.len()).sum();
        let mut out = vec![];
        for (idx, zero) in header(r)? {
            if !zero {
                let (owner, p) = slot(&chain, idx);
                let p = p.ok_or_else(|| match idx >= schema {
                    true => format!("value in schema slot {idx} but mappings of {sname} have {schema} slots: class changed since the mappings"),
                    false => format!("{sname}: schema slot {idx} ({owner}) has no property in the mappings"),
                })?;
                let v = at(format_args!(".{}", p.name), self.value(r, &p.ty, true, depth))?;
                out.push((p.name.clone(), v));
            }
        }
        Ok(out)
    }

    /// `top`: value is a direct member of an unversioned struct (enums are numeric there, FName inside containers).
    fn value(&self, r: &mut Rd, ty: &Ty, top: bool, depth: u32) -> Res<Value> {
        Ok(match ty {
            Ty::Bool => match r.u8()? {
                b @ (0 | 1) => Value::Bool(b == 1),
                b => return Err(format!("bool byte {b} at 0x{:x}", r.p - 1).into()),
            },
            Ty::Byte => Value::UInt(r.u8()? as u64),
            Ty::Int8 => Value::Int(r.u8()? as i8 as i64),
            Ty::Int16 => Value::Int(r.u16()? as i16 as i64),
            Ty::Int => Value::Int(r.i32()? as i64),
            Ty::Int64 => Value::Int(r.i64()?),
            Ty::UInt16 => Value::UInt(r.u16()? as u64),
            Ty::UInt32 => Value::UInt(r.u32()? as u64),
            Ty::UInt64 => Value::UInt(r.u64()?),
            Ty::Float => Value::Float(sane(r.f32()?, f32::MIN_POSITIVE as f64, r.p - 4)?),
            Ty::Double => Value::Float(sane(r.f64()?, f64::MIN_POSITIVE, r.p - 8)?),
            Ty::Name => Value::Name(self.name(r)?),
            Ty::Str => Value::Str(r.fstring()?),
            Ty::Text => Value::Str(self.text(r, 0)?),
            Ty::Object => Value::Object(self.object(r)?),
            Ty::SoftObject => Value::SoftObject(self.soft(r)?),
            Ty::Lazy => Value::Str(guid(r)?),
            Ty::Delegate => self.delegate(r)?,
            Ty::MulticastDelegate => Value::Array((0..r.count(12)?).map(|_| self.delegate(r)).collect::<Res<_>>()?),
            Ty::FieldPath => {
                let path = (0..r.count(8)?).map(|_| self.name(r).map(Value::Name)).collect::<Res<_>>()?;
                Value::Struct("FieldPath".into(), vec![("Path".into(), Value::Array(path)), ("Owner".into(), Value::Object(self.object(r)?))])
            }
            Ty::Enum(inner, e) if top => {
                let v = match self.value(r, inner, true, depth)? {
                    Value::Int(i) => i,
                    Value::UInt(u) => u as i64,
                    v => return Err(format!("enum {e} underlying value {v:?}").into()),
                };
                Value::Enum(self.maps.enum_name(e, v)?)
            }
            Ty::Enum(..) => Value::Enum(self.name(r)?),
            Ty::Array(inner) => {
                let n = r.count(1)?;
                Value::Array((0..n).map(|i| at(format_args!("[{i}]"), self.value(r, inner, false, depth))).collect::<Res<_>>()?)
            }
            Ty::Set(inner) => {
                for _ in 0..r.count(1)? {
                    self.value(r, inner, false, depth)?; // elements to remove
                }
                let n = r.count(1)?;
                Value::Array((0..n).map(|i| at(format_args!("[{i}]"), self.value(r, inner, false, depth))).collect::<Res<_>>()?)
            }
            Ty::Map(k, v) => {
                for _ in 0..r.count(1)? {
                    self.value(r, k, false, depth)?; // keys to remove
                }
                let n = r.count(2)?;
                let entries = (0..n).map(|i| {
                    let key = at(format_args!("[{i}].key"), self.value(r, k, false, depth))?;
                    Ok((key, at(format_args!("[{i}].value"), self.value(r, v, false, depth))?))
                });
                Value::Map(entries.collect::<Res<_>>()?)
            }
            Ty::Optional(inner) => match r.bool32()? {
                true => self.value(r, inner, false, depth)?,
                false => Value::Str("None".into()),
            },
            Ty::Struct(s) => match self.native(r, s)? {
                Some(v) => v,
                None => Value::Struct(s.clone(), self.read_struct(r, s, depth + 1)?),
            },
            Ty::Unknown(t) => return Err(format!("unknown usmap property type {t}").into()),
        })
    }

    fn delegate(&self, r: &mut Rd) -> Res<Value> {
        let o = self.object(r)?;
        Ok(Value::Struct("Delegate".into(), vec![("Object".into(), Value::Object(o)), ("Function".into(), Value::Name(self.name(r)?))]))
    }

    /// Structs with a native binary layout (FScriptStruct in CUE4Parse; UE5 large-world doubles).
    fn native(&self, r: &mut Rd, s: &str) -> Res<Option<Value>> {
        let f = |names: &[&str], vals: Vec<f64>| -> Value {
            Value::Struct(s.into(), names.iter().zip(vals).map(|(n, v)| (n.to_string(), Value::Float(v))).collect())
        };
        let i = |names: &[&str], vals: Vec<i64>| -> Value {
            Value::Struct(s.into(), names.iter().zip(vals).map(|(n, v)| (n.to_string(), Value::Int(v))).collect())
        };
        let d = |r: &mut Rd, n: usize| (0..n).map(|_| r.f64()).collect::<Res<Vec<_>>>();
        let fl = |r: &mut Rd, n: usize| (0..n).map(|_| r.f32()).collect::<Res<Vec<_>>>();
        let int = |r: &mut Rd, n: usize| (0..n).map(|_| r.i32().map(|v| v as i64)).collect::<Res<Vec<_>>>();
        const XYZ: &[&str] = &["X", "Y", "Z", "W"];
        let bx = |r: &mut Rd, n: usize, dbl: bool| -> Res<Value> {
            let v = if dbl { d(r, n * 2)? } else { fl(r, n * 2)? };
            let valid = r.u8()? != 0;
            let (a, b) = v.split_at(n);
            Ok(Value::Struct(s.into(), vec![
                ("Min".into(), f(&XYZ[..n], a.to_vec())),
                ("Max".into(), f(&XYZ[..n], b.to_vec())),
                ("IsValid".into(), Value::Bool(valid)),
            ]))
        };
        Ok(Some(match s {
            "Vector" | "Vector_NetQuantize" | "Vector_NetQuantize10" | "Vector_NetQuantize100" | "Vector_NetQuantizeNormal" | "Vector3d" => f(XYZ, d(r, 3)?),
            "Vector2D" => f(XYZ, d(r, 2)?),
            "Vector4" | "Vector4d" | "Quat" | "Quat4d" | "Plane" | "Plane4d" | "Sphere" => f(XYZ, d(r, 4)?),
            "Rotator" => f(&["Pitch", "Yaw", "Roll"], d(r, 3)?),
            "Vector3f" => f(XYZ, fl(r, 3)?),
            "Vector2f" | "DeprecateSlateVector2D" => f(XYZ, fl(r, 2)?),
            "Vector4f" | "Quat4f" | "Plane4f" => f(XYZ, fl(r, 4)?),
            "Rotator3f" => f(&["Pitch", "Yaw", "Roll"], fl(r, 3)?),
            "TwoVectors" => f(&["v1.X", "v1.Y", "v1.Z", "v2.X", "v2.Y", "v2.Z"], d(r, 6)?),
            "Matrix" => Value::Array(d(r, 16)?.into_iter().map(Value::Float).collect()),
            "Box" => bx(r, 3, true)?,
            "Box2D" => bx(r, 2, true)?,
            "Box3f" => bx(r, 3, false)?,
            "Box2f" => bx(r, 2, false)?,
            "LinearColor" => f(&["R", "G", "B", "A"], fl(r, 4)?),
            "Color" => {
                let [b, g, rr, a] = r.arr()?;
                i(&["R", "G", "B", "A"], vec![rr as i64, g as i64, b as i64, a as i64])
            }
            "Guid" | "GUID" => Value::Str(guid(r)?),
            "IntPoint" | "Int32Point" | "IntVector2" | "Int32Vector2" => i(XYZ, int(r, 2)?),
            "IntVector" => i(XYZ, int(r, 3)?),
            "IntVector4" => i(XYZ, int(r, 4)?),
            "FrameNumber" | "NavAgentSelector" => Value::Int(r.i32()? as i64),
            "DateTime" | "Timespan" => Value::Int(r.i64()?),
            "SoftObjectPath" | "SoftClassPath" | "StringAssetReference" | "StringClassReference" => Value::SoftObject(self.soft(r)?),
            "GameplayTagContainer" => Value::Array((0..r.count(8)?).map(|_| self.name(r).map(Value::Name)).collect::<Res<_>>()?),
            "PerPlatformFloat" | "PerPlatformInt" | "PerPlatformBool" | "PerPlatformFrameRate" => {
                let cooked = r.bool32()?;
                let one = |r: &mut Rd| -> Res<Value> {
                    Ok(match s {
                        "PerPlatformFloat" => Value::Float(r.f32()?),
                        "PerPlatformInt" => Value::Int(r.i32()? as i64),
                        "PerPlatformBool" => Value::Bool(r.bool32()?),
                        _ => i(&["Numerator", "Denominator"], int(r, 2)?),
                    })
                };
                let default = one(r)?;
                if !cooked {
                    for _ in 0..r.count(12)? {
                        self.name(r)?;
                        one(r)?;
                    }
                }
                default
            }
            "PerQualityLevelInt" | "PerQualityLevelFloat" => {
                r.bool32()?; // bCooked
                let one = |r: &mut Rd| if s.ends_with("Int") { r.i32().map(|v| Value::Int(v as i64)) } else { r.f32().map(Value::Float) };
                let default = one(r)?;
                let mut m = vec![];
                for _ in 0..r.count(8)? {
                    let k = Value::Int(r.i32()? as i64);
                    m.push((k, one(r)?));
                }
                Value::Struct(s.into(), vec![("Default".into(), default), ("PerQuality".into(), Value::Map(m))])
            }
            "RichCurveKey" => {
                let [interp, tangent, weight] = r.arr()?;
                let v = fl(r, 6)?;
                let mut fields = vec![("InterpMode".into(), Value::UInt(interp as u64)), ("TangentMode".into(), Value::UInt(tangent as u64)), ("TangentWeightMode".into(), Value::UInt(weight as u64))];
                let names = ["Time", "Value", "ArriveTangent", "ArriveTangentWeight", "LeaveTangent", "LeaveTangentWeight"];
                fields.extend(names.iter().zip(v).map(|(n, v)| (n.to_string(), Value::Float(v))));
                Value::Struct(s.into(), fields)
            }
            "SimpleCurveKey" => f(&["Time", "Value"], fl(r, 2)?),
            // native layouts we don't decode: fail loudly instead of reading them as tagged structs
            "InstancedStruct" | "InstancedStructContainer" | "InstancedPropertyBag" | "InstancedOverridablePropertyBag" | "ExpressionInput"
            | "ColorMaterialInput" | "ScalarMaterialInput" | "VectorMaterialInput" | "Vector2MaterialInput" | "MaterialAttributesInput"
            | "MovieSceneFloatChannel" | "MovieSceneDoubleChannel" | "MovieSceneEvaluationKey" | "MovieSceneSegment"
            | "MovieSceneFrameRange" | "MovieSceneEvalTemplatePtr" | "MovieSceneTrackImplementationPtr" | "MovieSceneSubSequenceTree"
            | "MovieSceneEvaluationFieldEntityTree" | "SectionEvaluationDataTree" | "LevelSequenceObjectReferenceMap"
            | "NiagaraVariable" | "NiagaraVariableBase" | "NiagaraVariableWithOffset" | "NiagaraDataInterfaceGPUParamInfo"
            | "FontData" | "FontCharacter" | "SmartName" | "UniqueNetIdRepl" | "MaterialOverrideNanite" | "MaterialLayersFunctionsTree"
            | "SkeletalMeshSamplingLODBuiltData" | "SkeletalMeshSamplingRegionBuiltData" | "ClothLODDataCommon" | "ClothTetherData"
            | "StateTreeInstanceData" | "WorldConditionQueryDefinition" | "VAL_CharacterCustomizationVariantOptionsArray"
            | "CompressedRichCurve" | "KeyHandleMap" | "Timecode" => {
                return Err(format!("native struct {s} not supported").into());
            }
            _ => return Ok(None),
        }))
    }
}

/// ponytail: misaligned reads tend to produce NaN/subnormal floats; real asset data almost never has them.
/// Only checked for top-level Float/Double properties (native struct floats are left alone).
fn sane(v: f64, min_normal: f64, at: usize) -> Res<f64> {
    if v.is_nan() || (v != 0.0 && v.abs() < min_normal) || v.abs() > 1e300 {
        return Err(format!("implausible float {v:e} at 0x{at:x}").into());
    }
    Ok(v)
}

fn guid(r: &mut Rd) -> Res<String> {
    let g: Vec<u32> = (0..4).map(|_| r.u32()).collect::<Res<_>>()?;
    Ok(format!("{:08X}{:08X}{:08X}{:08X}", g[0], g[1], g[2], g[3]))
}

/// Schema slot `idx` across the class chain: own slots first, then the super's (as CUE4Parse Struct.TryGetValue).
fn slot<'a>(chain: &[(&'a str, &'a Struct)], mut idx: usize) -> (&'a str, Option<&'a Prop>) {
    for (n, s) in chain {
        if idx < s.props.len() {
            return (n, s.props[idx].as_ref());
        }
        idx -= s.props.len();
    }
    ("?", None)
}

/// FUnversionedHeader: u16 fragments [SkipNum:7 | HasAnyZeroes:1 | IsLast:1 | ValueNum:7], then the
/// zero mask. Returns (schema slot, zero-masked) for every slot that has a value or a zero bit.
fn header(r: &mut Rd) -> Res<Vec<(usize, bool)>> {
    let mut frags = vec![];
    let mut zeroes = 0;
    loop {
        let f = r.u16()?;
        let (skip, has_zero, last, num) = ((f & 0x7f) as usize, f & 0x80 != 0, f & 0x100 != 0, (f >> 9) as usize);
        frags.push((skip, has_zero, num));
        if has_zero {
            zeroes += num;
        }
        if last {
            break;
        }
    }
    let mask = match zeroes {
        0 => vec![],
        1..=8 => r.take(1)?.to_vec(),
        9..=16 => r.take(2)?.to_vec(),
        n => r.take(n.div_ceil(32) * 4)?.to_vec(),
    };
    let (mut idx, mut zi, mut out) = (0usize, 0usize, vec![]);
    for (skip, has_zero, num) in frags {
        idx += skip;
        for _ in 0..num {
            out.push((idx, has_zero && mask.get(zi / 8).is_some_and(|b| b >> (zi % 8) & 1 != 0)));
            zi += has_zero as usize;
            idx += 1;
        }
    }
    Ok(out)
}

/// "/Game/X/MI_Foo.<public export hash>" -> "/Game/X/MI_Foo.MI_Foo" when the hash is that of the
/// package's main asset (CityHash64 of the lowercased UTF-16 name, "mi_foo", verified on Palia).
fn pretty_import(p: &str) -> String {
    let Some((pkg, hash)) = p.rsplit_once('.') else { return p.into() };
    let (Ok(h), true) = (u64::from_str_radix(hash, 16), hash.len() == 16) else { return p.into() };
    let short = pkg.rsplit('/').next().unwrap_or(pkg);
    if crate::global::export_hash(short) == h { format!("{pkg}.{short}") } else { p.into() }
}

// ---------------------------------------------------------------- export entry points

const UNVERSIONED: u32 = 0x2000;

/// "engine" (/Script/ module outside the game), "game" (/Script/Valeria*, S6*, Palia*), or "bp".
pub fn class_kind(pkg: &ZenPackage, i: usize, res: &Resolver) -> &'static str {
    let ci = pkg.exports[i].class_index;
    if ci >> 62 != 1 {
        return "bp";
    }
    let module = res.object_path(ci, pkg).unwrap_or_default();
    let module = module.trim_start_matches("/Script/").split('.').next().unwrap_or("");
    if ["Valeria", "S6", "Palia"].iter().any(|g| module.starts_with(g)) { "game" } else { "engine" }
}

/// Parses the unversioned property block at the start of export `i`'s data.
/// `class` = short class name (from Resolver::class_name). Returns properties and bytes consumed,
/// so callers (e.g. texture decoding) can continue reading native data after the properties.
pub fn read_properties(pkg: &ZenPackage, i: usize, class: &str, maps: &Mappings, res: &Resolver) -> Res<(Vec<(String, Value)>, usize)> {
    if pkg.flags & UNVERSIONED == 0 {
        return Err("package uses tagged (versioned) properties: not supported".into());
    }
    let e = pkg.exports.get(i).ok_or("bad export index")?;
    if e.class_index >> 62 != 1 {
        // ponytail: BP classes add their own (unmapped) properties in front of the native parent's,
        // so without parsing the BlueprintGeneratedClass's FProperty list the schema is unknown
        let parent = match e.class_index >> 62 {
            0 => pkg.exports.get((e.class_index & ((1 << 62) - 1)) as usize).and_then(|c| res.object_path(c.super_index, pkg)),
            _ => None,
        };
        let parent = parent.map(|p| format!(", parent {p}")).unwrap_or_default();
        return Err(format!("BP class {class}{parent}: schema unknown").into());
    }
    let ctx = Ctx { pkg, maps, res };
    let mut r = Rd { d: pkg.export_data(i), p: 0 };
    let props = ctx.read_struct(&mut r, class, 0)?;
    Ok((props, r.p))
}

/// Like read_properties, but keeps the properties read before an error, together with the error.
/// Values after the error point are missing; values before it are only trustworthy as far as the
/// schema matched up to there (the usual sanity checks still apply).
pub fn read_properties_partial(pkg: &ZenPackage, i: usize, class: &str, maps: &Mappings, res: &Resolver) -> (Vec<(String, Value)>, Option<String>) {
    let p = read_properties_drift(pkg, i, class, maps, res, &[], &[]);
    (p.props, p.error)
}

/// How the build's schema of struct `name` differs from the mappings: the build has `slots` slots for
/// it, of which the first `known` still match the mappings (the rest have unknown types).
pub struct Drift {
    pub name: &'static str,
    pub slots: usize,
    pub known: usize,
}

/// A partial property read.
pub struct Partial {
    /// values read, in stream order
    pub props: Vec<(String, Value)>,
    /// why reading stopped (values after that point are missing), or which values were skipped
    pub error: Option<String>,
    /// (build schema slot, zero-masked) of every slot in the header
    pub header: Vec<(usize, bool)>,
    /// where the values end (native data follows), if all of them were read or located
    pub end: Option<usize>,
}

enum Look<'a> {
    Known(&'a Prop),
    Unknown(&'a str),
    Missing(&'a str),
}

/// Class chain with build slot counts: (struct, mappings, build slots, known leading slots).
fn drift_chain<'a>(maps: &'a Mappings, class: &str, drift: &[Drift]) -> Res<Vec<(&'a str, &'a Struct, usize, usize)>> {
    let mut chain = vec![];
    let mut s = maps.structs.get_key_value(class);
    while let Some((n, st)) = s {
        let d = drift.iter().find(|d| d.name == n);
        chain.push((n.as_str(), st, d.map_or(st.props.len(), |d| d.slots), d.map_or(st.props.len(), |d| d.known.min(st.props.len()))));
        s = st.super_name.as_deref().and_then(|n| maps.structs.get_key_value(n));
        if chain.len() > 64 {
            return Err(format!("super chain of {class} loops").into());
        }
    }
    if chain.is_empty() {
        return Err(format!("no mappings for struct {class}").into());
    }
    Ok(chain)
}

fn look<'a>(chain: &[(&'a str, &'a Struct, usize, usize)], mut idx: usize) -> Look<'a> {
    for &(n, s, slots, known) in chain {
        if idx < slots {
            return match idx < known {
                true => s.props[idx].as_ref().map_or(Look::Missing(n), Look::Known),
                false => Look::Unknown(n),
            };
        }
        idx -= slots;
    }
    Look::Missing("?")
}

/// Build schema slot of top-level property `prop` of `class` under `drift` (None if not found or in a
/// changed range).
pub fn build_slot(maps: &Mappings, class: &str, drift: &[Drift], prop: &str) -> Option<usize> {
    let mut base = 0;
    for (_, s, slots, known) in drift_chain(maps, class, drift).ok()? {
        if let Some(k) = s.props.iter().take(known).position(|p| p.as_ref().is_some_and(|p| p.name == prop)) {
            return Some(base + k);
        }
        base += slots;
    }
    None
}

/// No NaN/subnormal/huge floats anywhere (native struct floats included).
fn plausible(v: &Value) -> bool {
    match v {
        Value::Float(f) => f.is_finite() && f.abs() < 1e7 && (*f == 0.0 || f.abs() >= f64::MIN_POSITIVE),
        Value::Struct(_, f) => f.iter().all(|x| plausible(&x.1)),
        Value::Array(a) => a.iter().all(plausible),
        Value::Map(m) => m.iter().all(|(k, v)| plausible(k) && plausible(v)),
        _ => true,
    }
}

/// Partial read of export `i` for a build whose schema drifted from the mappings (`drift`, see
/// [Drift]). Values in a changed range (or that fail to parse) have unknown sizes; if `tails` is
/// given and only known slots follow them, where those start is searched for instead: `tails` =
/// byte strings known to follow the properties (exact = they end the export, else a prefix).
// ponytail: anchor search = every offset after the unknown values where all remaining values parse
// sanely and a known tail follows (else: a zero u32); accepted only if exactly one offset fits.
// Ceiling: a remainder that parses at two offsets is reported as missing, not guessed.
pub fn read_properties_drift(pkg: &ZenPackage, i: usize, class: &str, maps: &Mappings, res: &Resolver, drift: &[Drift], tails: &[(&[u8], bool)]) -> Partial {
    let mut out = Partial { props: vec![], error: None, header: vec![], end: None };
    if pkg.flags & UNVERSIONED == 0 || pkg.exports.get(i).is_none_or(|e| e.class_index >> 62 != 1) {
        out.error = read_properties(pkg, i, class, maps, res).err().map(|e| e.to_string()); // tagged / BP class
        return out;
    }
    let chain = match drift_chain(maps, class, drift) {
        Ok(c) => c,
        Err(e) => {
            out.error = Some(e.to_string());
            return out;
        }
    };
    let ctx = Ctx { pkg, maps, res };
    let d = pkg.export_data(i);
    let mut r = Rd { d, p: 0 };
    match header(&mut r) {
        Ok(h) => out.header = h,
        Err(e) => {
            out.error = Some(format!("header: {e}"));
            return out;
        }
    }
    let vals: Vec<usize> = out.header.iter().filter(|s| !s.1).map(|s| s.0).collect();
    for (k, &idx) in vals.iter().enumerate() {
        let why = match look(&chain, idx) {
            Look::Missing(n) => {
                out.error = Some(format!("schema slot {idx} ({n}) has no property in the mappings"));
                return out;
            }
            Look::Known(p) => match at(format_args!(".{}", p.name), ctx.value(&mut r, &p.ty, true, 0)) {
                Ok(v) => {
                    out.props.push((p.name.clone(), v));
                    continue;
                }
                Err(e) => e.to_string(),
            },
            Look::Unknown(n) => format!("slot {idx} ({n}) has an unknown type (class changed since the mappings)"),
        };
        if tails.is_empty() {
            out.error = Some(why);
            return out;
        }
        // skip every value up to the last one of unknown type, then find where the rest starts
        let last = vals.iter().rposition(|&j| matches!(look(&chain, j), Look::Unknown(_))).unwrap_or(k).max(k);
        let rest: Vec<&Prop> = vals[last + 1..].iter().filter_map(|&j| match look(&chain, j) {
            Look::Known(p) => Some(p),
            _ => None,
        }).collect();
        let skipped = format!("{why}; {} value(s) skipped", last + 1 - k);
        out.error = Some(if rest.len() != vals.len() - last - 1 {
            format!("{skipped}; a later slot has no property in the mappings")
        } else if rest.is_empty() {
            skipped
        } else {
            // strong hit: what follows is a known trailer; weak: just a zero u32
            let (mut strong, mut weak) = (vec![], vec![]);
            for o in r.p + 1..d.len() {
                let mut rr = Rd { d, p: o };
                let Ok(v) = rest.iter().map(|p| Ok((p.name.clone(), ctx.value(&mut rr, &p.ty, true, 0)?))).collect::<Res<Vec<_>>>() else { continue };
                let tail = &d[rr.p..];
                if !v.iter().all(|x| plausible(&x.1)) {
                    continue;
                }
                if tails.iter().any(|&(t, exact)| if exact { tail == t } else { tail.starts_with(t) }) {
                    strong.push((v, rr.p));
                } else if tail.starts_with(&[0; 4]) && weak.len() < 2 {
                    weak.push((v, rr.p));
                }
                if strong.len() > 1 {
                    break;
                }
            }
            let hits = if strong.is_empty() { weak } else { strong };
            match <[_; 1]>::try_from(hits) {
                Ok([(v, end)]) => {
                    out.props.extend(v);
                    out.end = Some(end);
                    format!("{skipped}, the {} value(s) after them located by search", rest.len())
                }
                Err(h) if h.is_empty() => format!("{skipped}; the {} value(s) after them could not be located", rest.len()),
                Err(_) => format!("{skipped}; the {} value(s) after them are ambiguous", rest.len()),
            }
        });
        return out;
    }
    out.end = Some(r.p);
    out
}

/// Plausibility check after the properties: UObject::Serialize writes a u32 "has guid" bool right
/// after them (not for CDOs/archetypes). Anything else means the schema didn't fit.
/// An all-default object's header is one skip-only fragment whose SkipNum is the build's slot count
/// for the whole class chain (UE FUnversionedHeaderBuilder): direct evidence of a changed class.
pub fn check_slots(pkg: &ZenPackage, i: usize, class: &str, maps: &Mappings) -> Option<String> {
    let f = u16::from_le_bytes(pkg.export_data(i).get(..2)?.try_into().ok()?);
    if f & 0x100 == 0 || f >> 9 != 0 || f & 0x7f == 0x7f {
        return None;
    }
    let mut schema = 0;
    let mut s = Some(class);
    for _ in 0..64 {
        let Some(st) = s.and_then(|n| maps.structs.get(n)) else { break };
        schema += st.props.len();
        s = st.super_name.as_deref();
    }
    let build = (f & 0x7f) as usize;
    (build != schema).then(|| format!("build has {build} property slots for {class}, mappings {schema}: class changed since the mappings"))
}

pub fn check_trailer(pkg: &ZenPackage, i: usize, consumed: usize) -> Option<String> {
    if pkg.exports[i].object_flags & 0x30 != 0 {
        return None;
    }
    let d = pkg.export_data(i);
    match d.get(consumed..consumed + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())) {
        Some(0 | 1) => None,
        Some(v) => Some(format!("u32 after properties at 0x{consumed:x} is {v}, expected guid flag 0/1: likely schema mismatch")),
        None => Some(format!("export ends at 0x{:x}, {} bytes after properties: expected a guid flag", d.len(), d.len() - consumed.min(d.len()))),
    }
}

/// UDataTable rows after properties + guid flag: i32 count, (FName row, unversioned RowStruct) each.
pub fn read_datatable(pkg: &ZenPackage, i: usize, props: &[(String, Value)], consumed: usize, maps: &Mappings, res: &Resolver) -> Res<Vec<(String, Value)>> {
    let row = props.iter().find(|p| p.0 == "RowStruct").map(|p| &p.1);
    let Some(Value::Object(path)) = row else { return Err("DataTable without RowStruct".into()) };
    if !path.starts_with("/Script/") {
        return Err(format!("RowStruct {path} is a user-defined struct: schema unknown").into());
    }
    let sname = path.rsplit('.').next().unwrap_or(path);
    let ctx = Ctx { pkg, maps, res };
    let mut r = Rd { d: pkg.export_data(i), p: consumed };
    if r.bool32()? {
        r.take(16)?;
    }
    let n = r.count(10)?;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let name = ctx.name(&mut r)?;
        let v = at(format_args!("rows[{name}]"), ctx.read_struct(&mut r, sname, 1))?;
        rows.push((name, Value::Struct(sname.into(), v)));
    }
    if r.left() != 0 {
        return Err(format!("{} bytes left after {n} rows: row struct {sname} changed since the mappings", r.left()).into());
    }
    Ok(rows)
}

/// Everything known about one export's properties.
pub struct Report {
    pub class: String,
    pub kind: &'static str,
    pub size: usize,
    pub props: Result<(Vec<(String, Value)>, usize), String>,
    pub warning: Option<String>,
    pub rows: Option<Result<Vec<(String, Value)>, String>>,
}

pub fn report(pkg: &ZenPackage, i: usize, maps: &Mappings, res: &Resolver) -> Report {
    let class = res.class_name(pkg, i);
    let props = read_properties(pkg, i, &class, maps, res).map_err(|e| e.to_string());
    let (mut warning, mut rows) = (None, None);
    if let Ok((p, n)) = &props {
        warning = check_slots(pkg, i, &class, maps).or_else(|| check_trailer(pkg, i, *n));
        if class == "DataTable" || class == "CompositeDataTable" {
            rows = Some(read_datatable(pkg, i, p, *n, maps, res).map_err(|e| e.to_string()));
        }
    }
    Report { kind: class_kind(pkg, i, res), size: pkg.exports[i].data_size, class, props, warning, rows }
}

// ---------------------------------------------------------------- JSON

fn json_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn json_obj(fields: &[(String, Value)], ind: usize, out: &mut String) {
    if fields.is_empty() {
        out.push_str("{}");
        return;
    }
    out.push_str("{\n");
    for (n, (k, v)) in fields.iter().enumerate() {
        out.push_str(&"  ".repeat(ind + 1));
        json_str(k, out);
        out.push_str(": ");
        json_value(v, ind + 1, out);
        out.push_str(if n + 1 < fields.len() { ",\n" } else { "\n" });
    }
    out.push_str(&"  ".repeat(ind));
    out.push('}');
}

fn json_list<'a>(items: impl ExactSizeIterator<Item = Box<dyn FnOnce(&mut String) + 'a>>, ind: usize, out: &mut String) {
    let len = items.len();
    if len == 0 {
        out.push_str("[]");
        return;
    }
    out.push_str("[\n");
    for (n, item) in items.enumerate() {
        out.push_str(&"  ".repeat(ind + 1));
        item(out);
        out.push_str(if n + 1 < len { ",\n" } else { "\n" });
    }
    out.push_str(&"  ".repeat(ind));
    out.push(']');
}

fn json_value(v: &Value, ind: usize, out: &mut String) {
    match v {
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(i) => out.push_str(&i.to_string()),
        Value::UInt(u) => out.push_str(&u.to_string()),
        Value::Float(f) if f.is_finite() => out.push_str(&format!("{f:?}")),
        Value::Float(f) => json_str(&f.to_string(), out),
        Value::Str(s) | Value::Name(s) | Value::Enum(s) | Value::Object(s) | Value::SoftObject(s) => json_str(s, out),
        Value::Struct(_, fields) => json_obj(fields, ind, out),
        Value::Array(a) if !a.is_empty() && a.iter().all(|v| !matches!(v, Value::Struct(..) | Value::Array(_) | Value::Map(_))) => {
            out.push('[');
            for (n, v) in a.iter().enumerate() {
                if n > 0 {
                    out.push_str(", ");
                }
                json_value(v, ind, out);
            }
            out.push(']');
        }
        Value::Array(a) => json_list(a.iter().map(|v| Box::new(move |o: &mut String| json_value(v, ind + 1, o)) as Box<dyn FnOnce(&mut String)>), ind, out),
        Value::Map(m) => json_list(
            m.iter().map(|(k, v)| {
                Box::new(move |o: &mut String| json_obj(&[("key".into(), k.clone()), ("value".into(), v.clone())], ind + 1, o)) as Box<dyn FnOnce(&mut String)>
            }),
            ind,
            out,
        ),
    }
}

/// Pretty JSON for the GUI / "Save JSON".
pub fn to_json(props: &[(String, Value)]) -> String {
    let mut s = String::new();
    json_obj(props, 0, &mut s);
    s
}

/// Whole package as JSON: one entry per export with class, properties (or the error) and DataTable rows.
pub fn package_json(pkg: &ZenPackage, maps: &Mappings, res: &Resolver) -> String {
    let s = |v: &str| Value::Str(v.into());
    let exports: Vec<Value> = (0..pkg.exports.len())
        .map(|i| {
            let r = report(pkg, i, maps, res);
            let mut f = vec![("name".into(), s(&pkg.exports[i].name)), ("class".into(), s(&r.class)), ("class_kind".into(), s(r.kind)), ("size".into(), Value::UInt(r.size as u64))];
            match r.props {
                Ok((p, n)) => {
                    f.push(("consumed".into(), Value::UInt(n as u64)));
                    f.push(("properties".into(), Value::Struct(String::new(), p)));
                }
                Err(e) => f.push(("error".into(), s(&e))),
            }
            if let Some(w) = r.warning {
                f.push(("warning".into(), s(&w)));
            }
            match r.rows {
                Some(Ok(rows)) => f.push(("rows".into(), Value::Struct(String::new(), rows))),
                Some(Err(e)) => f.push(("rows_error".into(), s(&e))),
                None => {}
            }
            Value::Struct(String::new(), f)
        })
        .collect();
    let file = maps.file.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    to_json(&[("package".into(), s(&pkg.name)), ("mappings".into(), s(&file)), ("exports".into(), Value::Array(exports))])
}

#[cfg(test)]
mod tests {
    #[test]
    fn game_name_from_paths() {
        let f = |v: &[&str]| super::game_name(&v.iter().map(|p| (p.to_string(), 0)).collect::<Vec<_>>());
        assert_eq!(f(&["Engine/Content/a", "Palia/Content/b", "Palia/Plugins/c"]), Some("Palia".into()));
        assert_eq!(f(&["Engine/Content/a"]), None);
    }

    use super::*;

    /// Synthetic usmap (v0, uncompressed) with struct Base{A:int} <- Thing{B:bool, C:enum E, D:array<name>},
    /// and an unversioned blob setting C, D and A (B zero-masked).
    #[test]
    fn usmap_and_unversioned() {
        let names = ["E", "E_A", "E_B", "Base", "A", "Thing", "B", "C", "D"];
        let mut d = vec![];
        d.extend((names.len() as u32).to_le_bytes());
        for n in names {
            d.push(n.len() as u8);
            d.extend(n.as_bytes());
        }
        let nm = |n: &str| (names.iter().position(|x| *x == n).unwrap() as i32).to_le_bytes();
        d.extend(1u32.to_le_bytes()); // enums
        d.extend(nm("E"));
        d.push(2);
        d.extend(nm("E_A"));
        d.extend(nm("E_B"));
        d.extend(2u32.to_le_bytes()); // structs
        d.extend(nm("Base"));
        d.extend((-1i32).to_le_bytes());
        d.extend([1, 0, 1, 0]); // 1 slot, 1 serializable
        d.extend([0, 0, 1]);
        d.extend(nm("A"));
        d.push(2); // IntProperty
        d.extend(nm("Thing"));
        d.extend(nm("Base"));
        d.extend([3, 0, 3, 0]);
        d.extend([0, 0, 1]);
        d.extend(nm("B"));
        d.push(1); // Bool
        d.extend([1, 0, 1]);
        d.extend(nm("C"));
        d.extend([26, 0]); // Enum(Byte)
        d.extend(nm("E"));
        d.extend([2, 0, 1]);
        d.extend(nm("D"));
        d.extend([8, 5]); // Array(Name)
        let mut file = vec![0xC4, 0x30, 0, 0];
        file.extend((d.len() as u32).to_le_bytes());
        file.extend((d.len() as u32).to_le_bytes());
        file.extend(&d);
        let maps = Mappings::parse(&file).unwrap();
        assert_eq!(maps.structs["Thing"].props.len(), 3);
        assert_eq!(maps.enums["E"][1].1, "E_B");

        // 1 fragment: skip 0, 4 values, has zeroes, last -> mask bit0 (B) zero
        let frag: u16 = 4 << 9 | 0x100 | 0x80;
        let mut blob = frag.to_le_bytes().to_vec();
        blob.push(0b0001);
        blob.push(1); // C = E_B
        blob.extend(1i32.to_le_bytes()); // D: 1 name
        blob.extend([1, 0, 0, 0, 0, 0, 0, 0]); // name #1 "Bar"
        blob.extend(42i32.to_le_bytes()); // A (super)
        let pkg = ZenPackage {
            name: "/Game/T".into(),
            flags: UNVERSIONED,
            names: vec!["Foo".into(), "Bar".into()],
            imported_packages: vec![],
            imported_package_names: vec![],
            imported_public_export_hashes: vec![],
            imports: vec![],
            exports: vec![],
            data: vec![],
        };
        let res = Resolver::default();
        let ctx = Ctx { pkg: &pkg, maps: &maps, res: &res };
        let mut r = Rd { d: &blob, p: 0 };
        let v = ctx.read_struct(&mut r, "Thing", 0).unwrap();
        assert_eq!(r.p, blob.len());
        assert_eq!(v[0], ("C".into(), Value::Enum("E_B".into())));
        assert_eq!(v[1], ("D".into(), Value::Array(vec![Value::Name("Bar".into())])));
        assert_eq!(v[2], ("A".into(), Value::Int(42)));
        // out-of-range enum and truncated data must error with the property path
        blob[3] = 7;
        let e = ctx.read_struct(&mut Rd { d: &blob, p: 0 }, "Thing", 0).unwrap_err().to_string();
        assert!(e.starts_with(".C: enum E has no value 7"), "{e}");
        assert!(ctx.read_struct(&mut Rd { d: &blob[..6], p: 0 }, "Thing", 0).is_err());
    }
}
#[cfg(test)]
mod real {
    use std::collections::{BTreeMap, HashMap};
    use std::path::Path;
    static UTOC: std::sync::LazyLock<String> = std::sync::LazyLock::new(crate::test_utoc);

    /// STRUCT=Texture2D cargo test --release dump_schema -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_schema() {
        let m = super::Mappings::load(&super::Mappings::find(Path::new(UTOC.as_str()), None).unwrap()).unwrap();
        let mut s = Some(std::env::var("STRUCT").unwrap());
        while let Some(n) = s {
            let st = &m.structs[&n];
            println!("== {n} ({} slots)", st.props.len());
            for (i, p) in st.props.iter().enumerate() {
                println!("  {i}: {:?}", p.as_ref().map(|p| (&p.name, &p.ty)));
            }
            s = st.super_name.clone();
        }
    }

    /// Mappings fit over every STEP-th package (default 20), nothing written:
    /// STEP=20 cargo test --release batch_fit -- --ignored --nocapture
    #[test]
    #[ignore]
    fn batch_fit() {
        let toc = crate::open_archive(Path::new(UTOC.as_str())).unwrap();
        let res = crate::global::Resolver::load(&Path::new(UTOC.as_str()).with_file_name("global.utoc"), &toc).unwrap();
        let maps = super::Mappings::load(&super::Mappings::find(Path::new(UTOC.as_str()), None).unwrap()).unwrap();
        let mut ucas = std::fs::File::open(crate::data_path(Path::new(UTOC.as_str()))).unwrap();
        let step: usize = std::env::var("STEP").map(|s| s.parse().unwrap()).unwrap_or(20);
        // (kind, class) -> [total, ok, suspect(warning), failed, ok with >=1 value]; (kind, error kind) -> count
        let mut stats: BTreeMap<(&str, String), [usize; 5]> = BTreeMap::new();
        let mut errs: HashMap<(&str, String), usize> = HashMap::new();
        let (mut pkgs, mut dt) = (0, [0usize; 3]);
        for (p, e) in toc.files.iter().filter(|(p, _)| p.ends_with(".uasset")).step_by(step) {
            let Ok(pkg) = crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).and_then(crate::zen::parse) else { continue };
            pkgs += 1;
            for i in 0..pkg.exports.len() {
                let r = super::report(&pkg, i, &maps, &res);
                let s = stats.entry((r.kind, r.class.clone())).or_default();
                s[0] += 1;
                match (&r.props, &r.warning) {
                    (Ok((p, _)), None) => {
                        s[1] += 1;
                        s[4] += !p.is_empty() as usize;
                    }
                    (Ok(_), Some(w)) => {
                        s[2] += 1;
                        if std::env::var("WARN").is_ok() {
                            *errs.entry(("warn", format!("{}: {}", r.class, w.split(" at 0x").next().unwrap_or(w)))).or_default() += 1;
                        }
                    }
                    (Err(e), _) => {
                        s[3] += 1;
                        // error kind: strip offsets/numbers so similar failures group
                        let k: String = e.split(": ").last().unwrap_or(e).chars().map(|c| if c.is_ascii_digit() { '#' } else { c }).collect();
                        let k = k.split(" at 0x").next().unwrap_or(&k).to_string();
                        *errs.entry((r.kind, k)).or_default() += 1;
                        if std::env::var("SHOW").is_ok_and(|c| c == r.class) {
                            println!("  {p} [{}]: {e}", pkg.exports[i].name);
                        }
                    }
                }
                if let Some(rows) = &r.rows {
                    dt[0] += 1;
                    match rows {
                        Ok(_) => dt[1] += 1,
                        Err(e) => {
                            dt[2] += 1;
                            if std::env::var("SHOW").is_ok_and(|c| c == "rows") {
                                println!("  {p}: {e}");
                            }
                        }
                    }
                }
            }
        }
        println!("{pkgs} packages (every {step}th .uasset)");
        for kind in ["engine", "game", "bp", "warn"].into_iter().filter(|k| *k != "warn" || std::env::var("WARN").is_ok()) {
            let rows: Vec<_> = stats.iter().filter(|((k, _), _)| *k == kind).collect();
            let sum = rows.iter().fold([0; 5], |a, (_, s)| [a[0] + s[0], a[1] + s[1], a[2] + s[2], a[3] + s[3], a[4] + s[4]]);
            println!("\n== {kind}: {} exports, ok {} ({:.1}%, {} with values), suspect {}, failed {}", sum[0], sum[1], 100.0 * sum[1] as f64 / sum[0].max(1) as f64, sum[4], sum[2], sum[3]);
            let mut rows = rows;
            rows.sort_by_key(|(_, s)| std::cmp::Reverse(s[0]));
            for ((_, c), s) in rows.iter().take(40) {
                println!("  {c:<44} total {:>5}  ok {:>5} (values {:>5})  suspect {:>4}  failed {:>5}", s[0], s[1], s[4], s[2], s[3]);
            }
            let mut e: Vec<_> = errs.iter().filter(|((k, _), _)| *k == kind).collect();
            e.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
            for ((_, k), n) in e.iter().take(12) {
                println!("    {n:>5}x {k}");
            }
        }
        println!("\nDataTables: {} with rows parsed {}, failed {}", dt[0], dt[1], dt[2]);
    }
}
