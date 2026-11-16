//! Name resolution across containers. OWNER: agent B.
//! global.utoc ScriptObjects chunk (type 5) -> script object index -> "/Script/Engine.Texture2D".
//! Package ids: CityHash64 of lowercased package path ("/game/maps/foo") -> file in a container.
#![allow(dead_code)]
use crate::zen::{ObjIndex, ZenPackage};
use crate::{Cur, Res, Toc};
use std::collections::HashMap;
use std::path::Path;

#[derive(Default)]
pub struct Resolver {
    /// FPackageObjectIndex (ScriptImport, raw u64) -> "/Script/Engine.Texture2D"
    pub script: HashMap<u64, String>,
    /// package id -> container file path ("Palia/Content/Maps/Foo.umap")
    pub packages: HashMap<u64, String>,
}

const KIND_SHIFT: u32 = 62;
const INDEX_MASK: u64 = (1 << KIND_SHIFT) - 1;

impl Resolver {
    /// `global_utoc` = path to global.utoc; `game` = the opened game container's toc (for package ids).
    pub fn load(global_utoc: &Path, game: &Toc) -> Res<Resolver> {
        let toc = crate::parse_toc(&std::fs::read(global_utoc)?)?;
        let entry = toc.ids.iter().position(|&(_, t)| t == 5).ok_or("no ScriptObjects chunk")?;
        let mut ucas = std::fs::File::open(global_utoc.with_extension("ucas"))?;
        let script = parse_script_objects(&crate::read_chunk(&toc, &mut ucas, entry as u32, u64::MAX)?)?;
        let packages = game
            .files
            .iter()
            .filter(|(p, _)| p.ends_with(".uasset") || p.ends_with(".umap"))
            .filter_map(|(p, _)| Some((package_id(&container_path_to_package(p)?), p.clone())))
            .collect();
        Ok(Resolver { script, packages })
    }

    /// Full object path for an import/class index, e.g. "/Script/Engine.Texture2D".
    /// For Export indices, resolves within `pkg`. PackageImports -> "<package>.<export hash hex>".
    pub fn object_path(&self, idx: ObjIndex, pkg: &ZenPackage) -> Option<String> {
        match idx >> KIND_SHIFT {
            0 => {
                let mut parts = vec![];
                let mut i = idx;
                // bounded walk: corrupt outer chains can't loop forever
                for _ in 0..64 {
                    let e = pkg.exports.get((i & INDEX_MASK) as usize)?;
                    parts.push(e.name.as_str());
                    if e.outer_index >> KIND_SHIFT != 0 {
                        break;
                    }
                    i = e.outer_index;
                }
                parts.push(&pkg.name);
                parts.reverse();
                Some(join_path(&parts))
            }
            1 => self.script.get(&idx).cloned(),
            2 => {
                // UE5.3+: low 32 bits = public export hash index, next 30 bits = imported package index
                let pkg_i = ((idx & INDEX_MASK) >> 32) as usize;
                let hash_i = (idx & 0xFFFF_FFFF) as usize;
                let id = *pkg.imported_packages.get(pkg_i)?;
                let hash = pkg.imported_public_export_hashes.get(hash_i)?;
                let name = self
                    .packages
                    .get(&id)
                    .and_then(|p| container_path_to_package(p))
                    .unwrap_or_else(|| format!("{id:016X}"));
                Some(format!("{name}.{hash:016X}"))
            }
            _ => None,
        }
    }

    /// Short class name of export `i` ("Texture2D"), best effort.
    pub fn class_name(&self, pkg: &ZenPackage, i: usize) -> String {
        self.object_path(pkg.exports[i].class_index, pkg)
            .map(|p| p.rsplit(['.', '/', ':']).next().unwrap_or(&p).to_string())
            .unwrap_or_else(|| "?".into())
    }
}

/// UE object path joining: "/Pkg" + "Obj" -> "/Pkg.Obj", deeper subobjects use ':'.
fn join_path(parts: &[&str]) -> String {
    let mut s = parts[0].to_string();
    for (n, p) in parts[1..].iter().enumerate() {
        s.push(if n == 0 { '.' } else { ':' });
        s.push_str(p);
    }
    s
}

/// FNameEntrySerialized::LoadNameBatch (UE5).
fn load_name_batch(c: &mut Cur) -> Res<Vec<String>> {
    let n = c.u32()? as usize;
    if n == 0 {
        return Ok(vec![]);
    }
    c.take(4 + 8)?; // string bytes, hash version
    c.take(n.checked_mul(8).ok_or("bad name count")?)?; // hashes
    let headers = c.take(n * 2)?;
    headers
        .chunks(2)
        .map(|h| {
            let len = (((h[0] & 0x7f) as usize) << 8) | h[1] as usize;
            Ok(if h[0] & 0x80 != 0 {
                let w: Vec<u16> = c.take(len * 2)?.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
                String::from_utf16_lossy(&w)
            } else {
                String::from_utf8_lossy(c.take(len)?).into()
            })
        })
        .collect()
}

/// ScriptObjects chunk: name batch, i32 count, FScriptObjectEntry[count] (32 bytes each).
fn parse_script_objects(buf: &[u8]) -> Res<HashMap<u64, String>> {
    let mut c = Cur(buf, 0);
    let names = load_name_batch(&mut c)?;
    let n = c.u32()? as usize;
    let mut entries = HashMap::new(); // global index -> (name, outer)
    for e in c.take(n.checked_mul(32).ok_or("bad script object count")?)?.chunks(32) {
        let u = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap());
        let (ni, num) = (u32::from_le_bytes(e[0..4].try_into()?) & 0x3FFF_FFFF, u32::from_le_bytes(e[4..8].try_into()?));
        let base = names.get(ni as usize).ok_or("bad script name index")?;
        let name = if num == 0 { base.clone() } else { format!("{base}_{}", num - 1) };
        entries.insert(u(8), (name, u(16)));
    }
    let mut out = HashMap::with_capacity(entries.len());
    for &idx in entries.keys() {
        let mut parts = vec![];
        let mut i = idx;
        for _ in 0..64 {
            let Some((name, outer)) = entries.get(&i) else { break };
            parts.push(name.as_str());
            i = *outer;
        }
        parts.reverse();
        out.insert(idx, join_path(&parts));
    }
    Ok(out)
}

/// "Palia/Content/Maps/Foo.umap" (container path) -> "/Game/Maps/Foo"
pub fn container_path_to_package(path: &str) -> Option<String> {
    let stem = path.rsplit_once('.').map_or(path, |(s, _)| s);
    let (root, rest) = stem.split_once("/Content/")?;
    let mount = match root {
        "Engine" => "Engine",
        r if !r.contains('/') => "Game",
        r => r.rsplit('/').next()?, // .../Plugins/.../<Plugin>/Content/X -> /<Plugin>/X
    };
    Some(format!("/{mount}/{rest}"))
}

/// UE FPackageId::FromName: CityHash64 of the lowercased package name, UTF-16LE bytes.
pub fn package_id(package_path: &str) -> u64 {
    let bytes: Vec<u8> = package_path.to_lowercase().encode_utf16().flat_map(u16::to_le_bytes).collect();
    city_hash64(&bytes)
}

/// Zen public export hash of a top-level export: CityHash64 of its lowercased name ("mi_foo"), UTF-16LE.
pub fn export_hash(name: &str) -> u64 {
    package_id(name)
}

// CityHash64 v1.1 (as in UE Core/Hash/CityHash.cpp).
const K0: u64 = 0xc3a5c85c97cb3127;
const K1: u64 = 0xb492b66fbe98f273;
const K2: u64 = 0x9ae16a3b2f90404f;

fn f64_(s: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(s[i..i + 8].try_into().unwrap())
}
fn f32_(s: &[u8], i: usize) -> u64 {
    u32::from_le_bytes(s[i..i + 4].try_into().unwrap()) as u64
}
fn smix(v: u64) -> u64 {
    v ^ (v >> 47)
}
fn len16(u: u64, v: u64, mul: u64) -> u64 {
    let a = smix((u ^ v).wrapping_mul(mul));
    smix((v ^ a).wrapping_mul(mul)).wrapping_mul(mul)
}
fn len0to16(s: &[u8]) -> u64 {
    let n = s.len();
    if n >= 8 {
        let mul = K2.wrapping_add(n as u64 * 2);
        let a = f64_(s, 0).wrapping_add(K2);
        let b = f64_(s, n - 8);
        let c = b.rotate_right(37).wrapping_mul(mul).wrapping_add(a);
        let d = a.rotate_right(25).wrapping_add(b).wrapping_mul(mul);
        return len16(c, d, mul);
    }
    if n >= 4 {
        let mul = K2.wrapping_add(n as u64 * 2);
        return len16((n as u64).wrapping_add(f32_(s, 0) << 3), f32_(s, n - 4), mul);
    }
    if n > 0 {
        let y = s[0] as u32 + ((s[n >> 1] as u32) << 8);
        let z = n as u32 + ((s[n - 1] as u32) << 2);
        return smix((y as u64).wrapping_mul(K2) ^ (z as u64).wrapping_mul(K0)).wrapping_mul(K2);
    }
    K2
}
fn len17to32(s: &[u8]) -> u64 {
    let n = s.len();
    let mul = K2.wrapping_add(n as u64 * 2);
    let a = f64_(s, 0).wrapping_mul(K1);
    let b = f64_(s, 8);
    let c = f64_(s, n - 8).wrapping_mul(mul);
    let d = f64_(s, n - 16).wrapping_mul(K2);
    len16(
        a.wrapping_add(b).rotate_right(43).wrapping_add(c.rotate_right(30)).wrapping_add(d),
        a.wrapping_add(b.wrapping_add(K2).rotate_right(18)).wrapping_add(c),
        mul,
    )
}
fn len33to64(s: &[u8]) -> u64 {
    let n = s.len();
    let mul = K2.wrapping_add(n as u64 * 2);
    let mut a = f64_(s, 0).wrapping_mul(K2);
    let mut b = f64_(s, 8);
    let c = f64_(s, n - 24);
    let d = f64_(s, n - 32);
    let e = f64_(s, 16).wrapping_mul(K2);
    let f = f64_(s, 24).wrapping_mul(9);
    let g = f64_(s, n - 8);
    let h = f64_(s, n - 16).wrapping_mul(mul);
    let u = a.wrapping_add(g).rotate_right(43).wrapping_add(b.rotate_right(30).wrapping_add(c).wrapping_mul(9));
    let v = (a.wrapping_add(g) ^ d).wrapping_add(f).wrapping_add(1);
    let w = u.wrapping_add(v).wrapping_mul(mul).swap_bytes().wrapping_add(h);
    let x = e.wrapping_add(f).rotate_right(42).wrapping_add(c);
    let y = v.wrapping_add(w).wrapping_mul(mul).swap_bytes().wrapping_add(g).wrapping_mul(mul);
    let z = e.wrapping_add(f).wrapping_add(c);
    a = x.wrapping_add(z).wrapping_mul(mul).wrapping_add(y).swap_bytes().wrapping_add(b);
    b = smix(z.wrapping_add(a).wrapping_mul(mul).wrapping_add(d).wrapping_add(h)).wrapping_mul(mul);
    b.wrapping_add(x)
}
fn weak32(s: &[u8], i: usize, mut a: u64, mut b: u64) -> (u64, u64) {
    let (w, x, y, z) = (f64_(s, i), f64_(s, i + 8), f64_(s, i + 16), f64_(s, i + 24));
    a = a.wrapping_add(w);
    b = b.wrapping_add(a).wrapping_add(z).rotate_right(21);
    let c = a;
    a = a.wrapping_add(x).wrapping_add(y);
    b = b.wrapping_add(a.rotate_right(44));
    (a.wrapping_add(z), b.wrapping_add(c))
}
fn city_hash64(s: &[u8]) -> u64 {
    let n = s.len();
    if n <= 16 {
        return len0to16(s);
    }
    if n <= 32 {
        return len17to32(s);
    }
    if n <= 64 {
        return len33to64(s);
    }
    let mut x = f64_(s, n - 40);
    let mut y = f64_(s, n - 16).wrapping_add(f64_(s, n - 56));
    let mut z = len16(f64_(s, n - 48).wrapping_add(n as u64), f64_(s, n - 24), K_MUL);
    let mut v = weak32(s, n - 64, n as u64, z);
    let mut w = weak32(s, n - 32, y.wrapping_add(K1), x);
    x = x.wrapping_mul(K1).wrapping_add(f64_(s, 0));
    let mut i = 0;
    let end = (n - 1) & !63;
    while i < end {
        x = x.wrapping_add(y).wrapping_add(v.0).wrapping_add(f64_(s, i + 8)).rotate_right(37).wrapping_mul(K1);
        y = y.wrapping_add(v.1).wrapping_add(f64_(s, i + 48)).rotate_right(42).wrapping_mul(K1);
        x ^= w.1;
        y = y.wrapping_add(v.0).wrapping_add(f64_(s, i + 40));
        z = z.wrapping_add(w.0).rotate_right(33).wrapping_mul(K1);
        v = weak32(s, i, v.1.wrapping_mul(K1), x.wrapping_add(w.0));
        w = weak32(s, i + 32, z.wrapping_add(w.1), y.wrapping_add(f64_(s, i + 16)));
        std::mem::swap(&mut z, &mut x);
        i += 64;
    }
    len16(
        len16(v.0, w.0, K_MUL).wrapping_add(smix(y).wrapping_mul(K1)).wrapping_add(z),
        len16(v.1, w.1, K_MUL).wrapping_add(x),
        K_MUL,
    )
}
const K_MUL: u64 = 0x9ddfea08eb382d69;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paths_and_ids() {
        assert_eq!(container_path_to_package("Palia/Content/Maps/Foo.umap").as_deref(), Some("/Game/Maps/Foo"));
        assert_eq!(container_path_to_package("Engine/Content/X.uasset").as_deref(), Some("/Engine/X"));
        assert_eq!(container_path_to_package("Palia/Plugins/A/B/Content/X.uasset").as_deref(), Some("/B/X"));
        assert_eq!(package_id("/Game/Maps/Foo"), package_id("/game/maps/foo"));
    }
}
