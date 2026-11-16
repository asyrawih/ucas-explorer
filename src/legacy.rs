//! Legacy (pre-IoStore) cooked package: .uasset summary/name/import/export maps, UE4.27 layout.
//! Input = .uasset bytes with the .uexp appended (export SerialOffsets address that combined file).
//! The result is a ZenPackage so every converter works on it unchanged; imports are encoded like zen
//! ones: ScriptImport (1<<62 | i) for /Script/ objects, PackageImport (2<<62 | pkg<<32 | i) otherwise,
//! with `import_paths[i]` holding the full object path that Resolver::object_path returns.
// ponytail: unversioned UE4.27 summaries only (FileVersionUE4 = 0 or >= 507, LegacyFileVersion -7..-8);
// export entries are assumed 104 bytes. Upgrade: version-gated fields for older UE4 games.
use crate::zen::{ObjIndex, ZenExport, ZenPackage};
use crate::Res;

pub const MAGIC: u32 = 0x9E2A83C1;
const NULL: ObjIndex = 3 << 62;
const FILTER_EDITOR_ONLY: u32 = 0x8000_0000;

struct Cur<'a>(&'a [u8], usize);
impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let s = self.1.checked_add(n).and_then(|e| self.0.get(self.1..e)).ok_or("legacy package truncated")?;
        self.1 += n;
        Ok(s)
    }
    fn i32(&mut self) -> Res<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u(&mut self) -> Res<usize> {
        usize::try_from(self.i32()?).map_err(|_| "legacy package: negative count/offset".into())
    }
    fn i64(&mut self) -> Res<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn fstring(&mut self) -> Res<String> {
        let len = self.i32()?;
        Ok(if len >= 0 {
            let b = self.take(len as usize)?;
            String::from_utf8_lossy(b.strip_suffix(&[0]).unwrap_or(b)).into()
        } else {
            let n = (len as i64).unsigned_abs() as usize;
            let w: Vec<u16> = self.take(n.checked_mul(2).ok_or("bad string")?)?.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            String::from_utf16_lossy(w.strip_suffix(&[0]).unwrap_or(&w))
        })
    }
}

const PATH_TAG: &[u8; 8] = b"\0UCASPKG";

/// Appends the package's container path after the data (read_chunk does this): the summary doesn't
/// store the package name, so parse derives it from the path. Export offsets are unaffected.
pub fn append_path(data: &mut Vec<u8>, path: &str) {
    data.extend(PATH_TAG);
    data.extend(path.as_bytes());
    data.extend((path.len() as u32).to_le_bytes());
}

/// Splits off an append_path trailer, if present.
fn take_path(data: &mut Vec<u8>) -> Option<String> {
    let n = data.len();
    let len = u32::from_le_bytes(data.get(n.checked_sub(4)?..)?.try_into().ok()?) as usize;
    let start = n.checked_sub(4 + len + PATH_TAG.len())?;
    if &data[start..start + PATH_TAG.len()] != PATH_TAG {
        return None;
    }
    let path = String::from_utf8(data[start + PATH_TAG.len()..n - 4].to_vec()).ok()?;
    data.truncate(start);
    Some(path)
}

pub fn is_legacy(data: &[u8]) -> bool {
    data.get(..4).is_some_and(|m| u32::from_le_bytes(m.try_into().unwrap()) == MAGIC)
}

pub fn parse(mut data: Vec<u8>) -> Res<ZenPackage> {
    let path = take_path(&mut data).and_then(|p| crate::global::container_path_to_package(&p));
    let mut c = Cur(&data, 4);
    let legacy = c.i32()?;
    if !(-8..=-7).contains(&legacy) {
        return Err(format!("legacy package: LegacyFileVersion {legacy} not supported (UE4.27 = -7)").into());
    }
    c.i32()?; // LegacyUE3Version
    let ue4 = c.i32()?;
    if legacy == -8 {
        c.i32()?; // FileVersionUE5
    }
    c.i32()?; // licensee version
    let custom = c.u()?;
    c.take(custom.checked_mul(20).ok_or("bad custom versions")?)?;
    if ue4 != 0 && ue4 < 507 {
        return Err(format!("legacy package: UE4 version {ue4} not supported (4.25+)").into());
    }
    c.i32()?; // TotalHeaderSize
    c.fstring()?; // FolderName
    let flags = c.i32()? as u32;
    let (name_count, name_off) = (c.u()?, c.u()?);
    if flags & FILTER_EDITOR_ONLY == 0 {
        c.fstring()?; // LocalizationId
    }
    c.take(8)?; // GatherableTextData count/offset
    let (export_count, export_off, import_count, import_off) = (c.u()?, c.u()?, c.u()?, c.u()?);

    let mut c = Cur(&data, name_off);
    let names: Vec<String> = (0..name_count)
        .map(|_| {
            let s = c.fstring()?;
            c.take(4)?; // case-(in)sensitive hashes
            Ok(s)
        })
        .collect::<Res<_>>()?;
    let fname = |c: &mut Cur| -> Res<String> {
        let (i, n) = (c.i32()?, c.i32()?);
        let base = names.get(i as usize).ok_or("legacy package: bad name index")?;
        Ok(if n == 0 { base.clone() } else { format!("{base}_{}", n - 1) })
    };

    // imports: (object name, outer FPackageIndex)
    let mut c = Cur(&data, import_off);
    let raw: Vec<(String, i32)> = (0..import_count)
        .map(|_| {
            fname(&mut c)?; // ClassPackage
            fname(&mut c)?; // ClassName
            let outer = c.i32()?;
            Ok((fname(&mut c)?, outer))
        })
        .collect::<Res<_>>()?;
    // full path and top-level package of each import (outer chain: package, object, subobjects)
    let chain = |mut i: usize| -> Vec<&str> {
        let mut parts = vec![];
        for _ in 0..64 {
            let Some((name, outer)) = raw.get(i) else { break };
            parts.push(name.as_str());
            if *outer >= 0 {
                break; // 0 = package; > 0 (export outer) does not happen for imports
            }
            i = (-outer - 1) as usize;
        }
        parts.reverse();
        parts
    };
    let mut import_paths = vec![];
    let mut imported_package_names: Vec<String> = vec![];
    let mut imports = vec![];
    for i in 0..raw.len() {
        let parts = chain(i);
        let path = join(&parts);
        let package = parts.first().copied().unwrap_or("");
        imports.push(if package.starts_with("/Script/") {
            (1 << 62) | i as u64
        } else {
            let k = imported_package_names.iter().position(|p| p == package).unwrap_or_else(|| {
                imported_package_names.push(package.to_string());
                imported_package_names.len() - 1
            });
            (2 << 62) | ((k as u64) << 32) | i as u64
        });
        import_paths.push(path);
    }
    let index = |v: i32| -> ObjIndex {
        match v {
            0 => NULL,
            v if v > 0 => (v - 1) as u64,
            v => imports.get((-(v as i64) - 1) as usize).copied().unwrap_or(NULL),
        }
    };

    let mut c = Cur(&data, export_off);
    let mut exports = vec![];
    for _ in 0..export_count {
        let start = c.1;
        let (class, sup, template, outer) = (c.i32()?, c.i32()?, c.i32()?, c.i32()?);
        let name = fname(&mut c)?;
        let object_flags = c.i32()? as u32;
        let (size, off) = (c.i64()?, c.i64()?);
        let (data_offset, data_size) = (usize::try_from(off)?, usize::try_from(size)?);
        if data_offset.checked_add(data_size).is_none_or(|e| e > data.len()) {
            return Err("legacy package: export data out of bounds (missing .uexp?)".into());
        }
        c.1 = start + 104;
        exports.push(ZenExport {
            name,
            outer_index: index(outer),
            class_index: index(class),
            super_index: index(sup),
            template_index: index(template),
            public_export_hash: 0,
            object_flags,
            data_offset,
            data_size,
        });
    }
    // the package's own path is in its name map: a "/..." name that is not an imported package and whose
    // last segment names one of the exports (assets "X", Blueprints "X_C"). The name map holds FNames
    // without their number, so "/Game/SM_Foo_10" appears as "/Game/SM_Foo": the export supplies the "_10".
    // ponytail: heuristic fallback for data without the read_chunk path trailer (64 of 52654 wrong in DRG).
    let own: Vec<&String> = names.iter().filter(|n| n.starts_with('/') && !n.starts_with("/Script/") && !imported_package_names.contains(n)).collect();
    let numbered = |rest: &str| rest.len() > 1 && rest.starts_with('_') && rest[1..].bytes().all(|b| b.is_ascii_digit());
    let pick = |exact: bool| {
        own.iter().find_map(|n| {
            let (dir, short) = n.rsplit_once('/')?;
            exports.iter().find_map(|e| {
                let en = e.name.strip_suffix("_C").unwrap_or(&e.name);
                let rest = en.strip_prefix(short)?;
                (if exact { rest.is_empty() } else { numbered(rest) }).then(|| format!("{dir}/{en}"))
            })
        })
    };
    let main = exports.first().map(|e| e.name.as_str()).unwrap_or("");
    let name = path.or_else(|| pick(true)).or_else(|| pick(false)).or_else(|| own.first().map(|n| n.to_string())).unwrap_or_else(|| format!("/Unknown/{main}"));
    let imported_packages = imported_package_names.iter().map(|n| crate::global::package_id(n)).collect();
    Ok(ZenPackage {
        name,
        flags,
        names,
        imported_packages,
        imported_package_names,
        imported_public_export_hashes: vec![],
        imports,
        exports,
        data,
        import_paths,
    })
}

fn join(parts: &[&str]) -> String {
    let mut s = parts.first().map_or(String::new(), |p| p.to_string());
    for (n, p) in parts.iter().skip(1).enumerate() {
        s.push(if n == 0 { '.' } else { ':' });
        s.push_str(p);
    }
    s
}

#[cfg(test)]
mod tests {
    /// Parses every .uasset/.umap of a legacy .pak: `UCAS_TEST_PAK=/path/x.pak cargo test legacy -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn parse_all_packages() {
        let pak = std::path::PathBuf::from(std::env::var("UCAS_TEST_PAK").expect("set UCAS_TEST_PAK"));
        let toc = crate::open_archive(&pak).unwrap();
        let res = crate::global::Resolver::load(&pak.with_file_name("global.utoc"), &toc).unwrap();
        let mut f = std::fs::File::open(&pak).unwrap();
        let (mut ok, mut fail, mut bad_names, mut classes) = (0, 0, 0, std::collections::BTreeMap::<String, usize>::new());
        let mut main = std::collections::BTreeMap::<String, usize>::new();
        for (p, e) in toc.files.iter().filter(|(p, _)| p.ends_with(".uasset") || p.ends_with(".umap")) {
            match crate::read_chunk(&toc, &mut f, *e, u64::MAX).and_then(crate::zen::parse) {
                Ok(pkg) => {
                    ok += 1;
                    let want = crate::global::container_path_to_package(p).unwrap_or_default();
                    if !pkg.name.eq_ignore_ascii_case(&want) {
                        bad_names += 1;
                        if bad_names < 10 {
                            println!("name mismatch {p}: {} vs {want}", pkg.name);
                        }
                    }
                    for i in 0..pkg.exports.len() {
                        *classes.entry(res.class_name(&pkg, i)).or_default() += 1;
                    }
                    // main export: named like the package (Blueprints: "<name>_C"); maps count as "umap"
                    let short = pkg.name.rsplit('/').next().unwrap_or("").to_string();
                    let m = if p.ends_with(".umap") { Some("umap".to_string()) } else {
                        (0..pkg.exports.len()).find(|&i| pkg.exports[i].name == short || pkg.exports[i].name == format!("{short}_C")).map(|i| res.class_name(&pkg, i))
                    };
                    *main.entry(m.unwrap_or_else(|| "?".into())).or_default() += 1;
                }
                Err(err) => {
                    fail += 1;
                    if fail < 10 {
                        println!("FAIL {p}: {err}");
                    }
                }
            }
        }
        let mut top: Vec<_> = main.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        println!("main classes: {:?}", &top[..top.len().min(60)]);
        let mut top: Vec<_> = classes.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        println!("ok {ok} fail {fail} wrong names {bad_names}\n{:?}", &top[..top.len().min(12)]);
        assert_eq!((fail, bad_names), (0, 0));
    }
}
