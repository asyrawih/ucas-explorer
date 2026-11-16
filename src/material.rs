//! StaticMesh -> OBJ + MTL + PNG textures, without .usmap.
// ponytail: no property schema, so material slots and texture parameters are recovered from object
// references (FPackageIndex i32, negative = import) found by byte scan, classified by loading the
// referenced package and checking its export classes. Ceiling: refs that merely look like an import
// index could be picked up; every hit is filtered by "referenced package is a Material/Texture2D".
use crate::global::{self, Resolver};
use crate::zen::ZenPackage;
use crate::{mesh, props, texture, Res};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Material,
    Instance,
    Texture,
    Other,
}

/// One material slot: material package path and its chosen textures (package paths).
#[derive(Default, Debug)]
struct Slot {
    material: String,
    diffuse: Option<String>,
    normal: Option<String>,
}

struct Ctx<'a> {
    res: &'a Resolver,
    load: &'a mut dyn FnMut(&str) -> Option<Vec<u8>>,
    pkgs: HashMap<String, Option<ZenPackage>>,
}

impl Ctx<'_> {
    /// "/Game/X/Y" -> container path "Palia/Content/X/Y.<ext>" via the resolver's package map.
    fn container_path(&self, package: &str, ext: &str) -> Option<String> {
        let p = self.res.packages.get(&global::package_id(package))?;
        Some(format!("{}.{ext}", p.rsplit_once('.').map_or(p.as_str(), |(s, _)| s)))
    }
    fn pkg(&mut self, package: &str) -> Option<&ZenPackage> {
        if !self.pkgs.contains_key(package) {
            let p = self.container_path(package, "uasset").and_then(|c| (self.load)(&c)).and_then(|d| crate::zen::parse(d).ok());
            self.pkgs.insert(package.to_string(), p);
        }
        self.pkgs.get(package)?.as_ref()
    }
    /// Main export index and its kind.
    fn kind(&mut self, package: &str) -> (Kind, usize) {
        let res = self.res;
        let Some(p) = self.pkg(package) else { return (Kind::Other, 0) };
        for i in 0..p.exports.len() {
            match res.class_name(p, i).as_str() {
                "Material" => return (Kind::Material, i),
                c if c.starts_with("MaterialInstance") => return (Kind::Instance, i),
                "Texture2D" => return (Kind::Texture, i),
                _ => {}
            }
        }
        (Kind::Other, 0)
    }
}

/// (offset, imported package path) of every i32 in `d` that is a FPackageIndex to a package import.
pub fn package_refs(pkg: &ZenPackage, d: &[u8]) -> Vec<(usize, String)> {
    (0..d.len().saturating_sub(3))
        .filter_map(|o| {
            let v = i32::from_le_bytes(d[o..o + 4].try_into().unwrap());
            let imp = *pkg.imports.get(usize::try_from(-(v as i64) - 1).ok()?)?;
            if imp >> 62 != 2 {
                return None;
            }
            let name = pkg.imported_package_names.get(((imp & ((1 << 62) - 1)) >> 32) as usize)?;
            Some((o, name.clone()))
        })
        .collect()
}

#[derive(PartialEq, Debug)]
enum Role {
    Diffuse,
    Normal,
}

/// Texture role from its name suffix (Palia: _C color, _N normal, _M/_ARM/_ARME masks) or, failing
/// that, from the material parameter name it is bound to.
fn role(param: &str, texture: &str) -> Option<Role> {
    let suffix = texture.rsplit(['_', '/']).next().unwrap_or("").to_lowercase();
    match suffix.as_str() {
        "n" | "nm" | "nrm" | "normal" | "normals" => return Some(Role::Normal),
        "c" | "d" | "bc" | "color" | "basecolor" | "albedo" | "diffuse" | "base" => return Some(Role::Diffuse),
        _ => {}
    }
    let p = param.to_lowercase();
    if p.contains("normal") {
        Some(Role::Normal)
    } else if ["basecolor", "albedo", "diffuse", "color"].iter().any(|k| p.contains(k)) && !p.contains("mask") {
        Some(Role::Diffuse)
    } else {
        None
    }
}

/// Textures of a material as (parameter name or "", texture package), walking up to the parent
/// material when an instance overrides no textures.
fn material_textures(ctx: &mut Ctx, material: &str, depth: u32) -> Vec<(String, String)> {
    let (kind, i) = ctx.kind(material);
    if depth > 6 || !matches!(kind, Kind::Material | Kind::Instance) {
        return vec![];
    }
    let (refs, names) = {
        let p = ctx.pkg(material).unwrap(); // kind() succeeded, so it's cached
        let d = p.export_data(i);
        let refs = package_refs(p, d);
        // TextureParameterValues entry: FName ParameterName, u8 Association, i32 Index(-1), i32 ParameterValue
        let names: Vec<Option<String>> = refs
            .iter()
            .map(|&(o, _)| {
                let u = |at: usize| d.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
                if o < 13 || u(o - 4)? != u32::MAX || u(o - 9)? != 0 {
                    return None;
                }
                p.names.get(u(o - 13)? as usize).cloned()
            })
            .collect();
        (refs, names)
    };
    let mut out: Vec<(String, String)> = vec![];
    let mut parent = None;
    for ((_, pkg), name) in refs.iter().zip(&names) {
        // instances: only parameter-bound refs (later refs are cached shader data); base materials:
        // every referenced texture (CachedExpressionData has no parameter names)
        if kind == Kind::Instance && name.is_none() {
            if parent.is_none() && matches!(ctx.kind(pkg).0, Kind::Material | Kind::Instance) {
                parent = Some(pkg.clone());
            }
            continue;
        }
        if !out.iter().any(|t| &t.1 == pkg) && ctx.kind(pkg).0 == Kind::Texture {
            out.push((name.clone().unwrap_or_default(), pkg.clone()));
        }
    }
    match parent {
        Some(p) if out.is_empty() => material_textures(ctx, &p, depth + 1),
        _ => out,
    }
}

/// Material slots of a StaticMesh export, in StaticMaterials order. bool = resolved from the
/// property data (false = fallback to material imports in import order).
/// Material slots of a mesh export. `end` = where the native data starts (refs after it are ignored),
/// `need` = max(MaterialIndex)+1 used by the sections.
fn slots(ctx: &mut Ctx, pkg: &ZenPackage, export: usize, end: usize, need: usize) -> (Vec<Slot>, bool) {
    let d = pkg.export_data(export);
    // ponytail: StaticMaterials / SkeletalMaterials is the first run of material refs in the property
    // block, i.e. before the native data; validated only by count >= max(MaterialIndex)+1.
    let mut mats: Vec<String> = package_refs(pkg, &d[..end.min(d.len())]).into_iter().map(|r| r.1).collect();
    mats.retain(|m| matches!(ctx.kind(m).0, Kind::Material | Kind::Instance));
    let resolved = mats.len() >= need;
    if !resolved {
        mats = pkg.imported_package_names.clone();
        mats.retain(|m| matches!(ctx.kind(m).0, Kind::Material | Kind::Instance));
    }
    let slots = mats.into_iter().map(|material| slot(ctx, material)).collect();
    (slots, resolved)
}

/// A material's base color stands in for a missing diffuse texture as this pseudo texture path
/// ("#color/RRGGBB", sRGB); texture_png renders it as a solid image, so every consumer handles it.
const COLOR: &str = "#color/";

fn get<'a>(fields: &'a [(String, props::Value)], name: &str) -> Option<&'a props::Value> {
    fields.iter().find(|f| f.0 == name).map(|f| &f.1)
}

/// Base color of a material from its vector parameters (tagged/legacy packages only: unversioned ones
/// need mappings), walking up the Parent chain. Names like BaseColor/Diffuse/Albedo/Color/Tint win;
/// else the first vector parameter that isn't emissive-like.
// ponytail: parameter-name heuristic; base Materials' expression defaults (CachedExpressionData) are not read
fn material_color(ctx: &mut Ctx, material: &str, depth: u32) -> Option<[f32; 3]> {
    let (kind, i) = ctx.kind(material);
    if depth > 6 || !matches!(kind, Kind::Material | Kind::Instance) {
        return None;
    }
    let res = ctx.res;
    let p = ctx.pkg(material)?;
    if p.flags & 0x2000 != 0 {
        return None;
    }
    let (fields, _) = props::read_properties(p, i, &res.class_name(p, i), &props::Mappings::default(), res).ok()?;
    let mut vecs: Vec<(String, [f32; 3])> = vec![];
    if let Some(props::Value::Array(a)) = get(&fields, "VectorParameterValues") {
        for v in a {
            let props::Value::Struct(_, f) = v else { continue };
            let name = match get(f, "ParameterInfo") {
                Some(props::Value::Struct(_, pi)) => match get(pi, "Name") {
                    Some(props::Value::Name(n)) => n.to_lowercase(),
                    _ => continue,
                },
                _ => continue,
            };
            let Some(props::Value::Struct(_, c)) = get(f, "ParameterValue") else { continue };
            let ch = |k: &str| match get(c, k) {
                Some(props::Value::Float(x)) => Some(*x as f32),
                _ => None,
            };
            if let (Some(r), Some(g), Some(b)) = (ch("R"), ch("G"), ch("B")) {
                vecs.push((name, [r, g, b]));
            }
        }
    }
    let skip = ["emissive", "emission", "glow", "fresnel", "rim", "edge", "light", "fog", "outline", "specular"];
    let usable = |n: &str| !skip.iter().any(|k| n.contains(k));
    let pick = ["basecolor", "base color", "diffuse", "albedo", "color", "tint"]
        .iter()
        .find_map(|k| vecs.iter().find(|(n, _)| n.contains(k) && usable(n)))
        .or_else(|| vecs.iter().find(|(n, _)| usable(n)));
    if let Some((_, c)) = pick {
        return Some(*c);
    }
    let parent = match get(&fields, "Parent") {
        Some(props::Value::Object(o)) if o != "None" => o.split(['.', ':']).next().unwrap_or(o).to_string(),
        _ => return None,
    };
    material_color(ctx, &parent, depth + 1)
}

/// Linear color -> "#color/RRGGBB" (sRGB).
fn color_path(c: [f32; 3]) -> String {
    let s = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let v = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (v * 255.0 + 0.5) as u8
    };
    format!("{COLOR}{:02x}{:02x}{:02x}", s(c[0]), s(c[1]), s(c[2]))
}

/// Diffuse/normal texture choice for one material.
fn slot(ctx: &mut Ctx, material: String) -> Slot {
    let mut tex = material_textures(ctx, &material, 0);
    // engine/template placeholders (e.g. layered materials' T_Template_C) go last
    tex.sort_by_key(|t| ["template", "default", "/engine/"].iter().any(|k| t.1.to_lowercase().contains(k)));
    // name suffix beats parameter name
    let pick = |r: Role| tex.iter().find(|t| role("", &t.1).as_ref() == Some(&r))
        .or_else(|| tex.iter().find(|t| role(&t.0, &t.1).as_ref() == Some(&r)))
        .map(|t| t.1.clone());
    let diffuse = pick(Role::Diffuse).or_else(|| material_color(ctx, &material, 0).map(color_path));
    Slot { diffuse, normal: pick(Role::Normal), material }
}

/// (material package, diffuse texture package) per material slot of a mesh export; `end` = where
/// its native data starts (usize::MAX = unknown), `need` = max(MaterialIndex)+1. `overrides[k]`
/// (a material package) replaces slot k's material.
pub fn slot_diffuse(
    res: &Resolver,
    pkg: &ZenPackage,
    export: usize,
    end: usize,
    need: usize,
    overrides: &[Option<String>],
    load: &mut dyn FnMut(&str) -> Option<Vec<u8>>,
) -> Vec<(String, Option<String>)> {
    let mut ctx = Ctx { res, load, pkgs: HashMap::new() };
    let (mut slots, _) = slots(&mut ctx, pkg, export, end, need);
    for (k, o) in overrides.iter().enumerate() {
        let Some(m) = o.clone().filter(|m| matches!(ctx.kind(m).0, Kind::Material | Kind::Instance)) else { continue };
        if k >= slots.len() {
            slots.resize_with(k + 1, Slot::default);
        }
        slots[k] = slot(&mut ctx, m);
    }
    slots.into_iter().map(|s| (s.material, s.diffuse)).collect()
}

/// Texture2D package -> PNG bytes.
pub fn texture_to_png(res: &Resolver, texture: &str, load: &mut dyn FnMut(&str) -> Option<Vec<u8>>) -> Option<Vec<u8>> {
    texture_png(&mut Ctx { res, load, pkgs: HashMap::new() }, texture)
}

/// Writes `<out_dir>/<mesh name>.obj` (LOD0, `usemtl` per section), `.mtl` and the referenced
/// textures as PNG. `load` reads another package's file by container path
/// ("Palia/Content/.../X.uasset" or ".ubulk"). Texture failures leave the material without a map.
/// Returns the written files.
pub fn export_obj_with_materials(
    res: &Resolver,
    pkg: &ZenPackage,
    export: usize,
    ubulk: Option<&[u8]>,
    load: &mut dyn FnMut(&str) -> Option<Vec<u8>>,
    out_dir: &Path,
) -> Res<Vec<PathBuf>> {
    use std::fmt::Write;
    let lods = mesh::decode_static_mesh(pkg.export_data(export), ubulk)?;
    let lod = &lods[0];
    let mut ctx = Ctx { res, load, pkgs: HashMap::new() };
    let d = pkg.export_data(export);
    let end = mesh::find_sections(d, lod).map_or(d.len(), |(o, _)| o);
    let need = lod.sections.iter().map(|s| s.0 as usize + 1).max().unwrap_or(1);
    let (slots, _) = slots(&mut ctx, pkg, export, end, need);
    let name = sanitize(&pkg.exports[export].name);
    std::fs::create_dir_all(out_dir)?;
    let mut written = vec![];

    let mut mtl = String::new();
    let mut pngs: HashMap<String, Option<String>> = HashMap::new(); // texture package -> png file name
    let mat_name = |k: usize| format!("{k}_{}", slots.get(k).map_or("default".into(), |s| sanitize(s.material.rsplit('/').next().unwrap_or(""))));
    for k in 0..slots.len().max(lod.sections.iter().map(|s| s.0 as usize + 1).max().unwrap_or(1)) {
        let _ = writeln!(mtl, "newmtl {}\nKd 1 1 1\nKs 0 0 0", mat_name(k));
        let Some(s) = slots.get(k).filter(|_| lod.sections.iter().any(|s| s.0 as usize == k)) else { continue };
        for (tex, key) in [(&s.diffuse, "map_Kd"), (&s.normal, "map_Bump")] {
            let Some(tex) = tex else { continue };
            if !pngs.contains_key(tex) {
                let f = write_texture(&mut ctx, tex, out_dir);
                if let Some(f) = &f {
                    written.push(out_dir.join(f));
                }
                pngs.insert(tex.clone(), f);
            }
            if let Some(Some(f)) = pngs.get(tex) {
                let _ = writeln!(mtl, "{key} {f}");
            }
        }
    }

    // Mesh::to_obj emits one "f" line per triangle in index order; insert usemtl at section starts
    let mut obj = format!("mtllib {name}.mtl\n");
    let mut tri = 0u32;
    let mut sec = lod.sections.iter().peekable();
    for line in lod.to_obj(&name).lines() {
        if line.starts_with("f ") {
            while let Some(s) = sec.next_if(|s| s.1 / 3 <= tri) {
                let _ = writeln!(obj, "usemtl {}", mat_name(s.0 as usize));
            }
            tri += 1;
        }
        obj.push_str(line);
        obj.push('\n');
    }
    for (file, data) in [(format!("{name}.obj"), obj), (format!("{name}.mtl"), mtl)] {
        std::fs::write(out_dir.join(&file), data)?;
        written.push(out_dir.join(file));
    }
    Ok(written)
}

/// One texture used by a material, for previews.
pub struct MatTexture {
    pub param: String,   // material parameter name ("" for base materials)
    pub texture: String, // texture package path
    pub role: &'static str, // "diffuse", "normal" or ""
    pub png: Option<Vec<u8>>,
}

/// Textures of material package `material` ("/Game/.../MI_Foo" or "M_Foo"), diffuse first, walking to the
/// parent when an instance overrides none. Empty if it isn't a material.
pub fn material_preview(res: &Resolver, material: &str, load: &mut dyn FnMut(&str) -> Option<Vec<u8>>) -> Vec<MatTexture> {
    let mut ctx = Ctx { res, load, pkgs: HashMap::new() };
    let mut tex = material_textures(&mut ctx, material, 0);
    // same ranking as mesh slots: placeholders last, diffuse (by suffix, then parameter) first
    tex.sort_by_key(|t| {
        let placeholder = ["template", "default", "/engine/"].iter().any(|k| t.1.to_lowercase().contains(k));
        let rank = match (role("", &t.1), role(&t.0, &t.1)) {
            (Some(Role::Diffuse), _) => 0,
            (_, Some(Role::Diffuse)) => 1,
            _ => 2,
        };
        (placeholder, rank)
    });
    let has_diffuse = tex.iter().any(|t| role(&t.0, &t.1) == Some(Role::Diffuse));
    let color = if has_diffuse { None } else { material_color(&mut ctx, material, 0) };
    let mut out: Vec<MatTexture> = color
        .map(|c| {
            let texture = color_path(c);
            MatTexture { param: "base color".into(), png: texture_png(&mut ctx, &texture), role: "diffuse", texture }
        })
        .into_iter()
        .collect();
    out.extend(tex.into_iter()
        .map(|(param, texture)| MatTexture {
            role: match role(&param, &texture) {
                Some(Role::Diffuse) => "diffuse",
                Some(Role::Normal) => "normal",
                None => "",
            },
            png: texture_png(&mut ctx, &texture),
            param,
            texture,
        }));
    out
}

/// Exports material package `material` into `out_dir`: every texture as PNG, `<name>.mtl` (diffuse +
/// normal) and `<name>.txt` listing parameter -> texture file. Returns the written files.
pub fn export_material(res: &Resolver, material: &str, load: &mut dyn FnMut(&str) -> Option<Vec<u8>>, out_dir: &Path) -> Res<Vec<PathBuf>> {
    use std::fmt::Write;
    let tex = material_preview(res, material, load);
    if tex.is_empty() {
        return Err("material: no textures found (not a material, or it uses none)".into());
    }
    let name = sanitize(material.rsplit('/').next().unwrap_or("material"));
    std::fs::create_dir_all(out_dir)?;
    let (mut written, mut list) = (vec![], format!("# {material}\n# parameter -> texture file (role)\n"));
    let mut mtl = format!("newmtl {name}\nKd 1 1 1\nKs 0 0 0\n");
    let (mut has_kd, mut has_bump) = (false, false);
    for t in &tex {
        let file = format!("{}.png", sanitize(t.texture.rsplit('/').next().unwrap_or("texture")));
        let ok = match &t.png {
            Some(png) => std::fs::write(out_dir.join(&file), png).is_ok(),
            None => false,
        };
        if ok {
            written.push(out_dir.join(&file));
        }
        let shown = if ok { file.as_str() } else { "(decode failed)" };
        let _ = writeln!(list, "{} -> {shown} ({})", if t.param.is_empty() { "-" } else { &t.param }, if t.role.is_empty() { "other" } else { t.role });
        // first diffuse / normal wins (list is diffuse-first)
        if ok && t.role == "diffuse" && !has_kd {
            let _ = writeln!(mtl, "map_Kd {file}");
            has_kd = true;
        } else if ok && t.role == "normal" && !has_bump {
            let _ = writeln!(mtl, "map_Bump {file}");
            has_bump = true;
        }
    }
    for (file, data) in [(format!("{name}.mtl"), mtl), (format!("{name}.txt"), list)] {
        std::fs::write(out_dir.join(&file), data)?;
        written.push(out_dir.join(file));
    }
    Ok(written)
}

/// Diffuse texture of each material slot as PNG bytes (None = no diffuse found / decode failed).
/// For meshes whose native layout material.rs doesn't know (e.g. SkeletalMesh): refs are taken from the
/// whole export. `need` = max(MaterialIndex)+1.
pub fn slot_diffuse_pngs(
    res: &Resolver,
    pkg: &ZenPackage,
    export: usize,
    need: usize,
    load: &mut dyn FnMut(&str) -> Option<Vec<u8>>,
) -> Vec<Option<Vec<u8>>> {
    let mut ctx = Ctx { res, load, pkgs: HashMap::new() };
    let (slots, _) = slots(&mut ctx, pkg, export, usize::MAX, need);
    slots.iter().map(|s| s.diffuse.as_ref().and_then(|t| texture_png(&mut ctx, t))).collect()
}

/// Decodes a Texture2D package to PNG bytes.
fn texture_png(ctx: &mut Ctx, texture: &str) -> Option<Vec<u8>> {
    if let Some(hex) = texture.strip_prefix(COLOR) {
        let c = u32::from_str_radix(hex, 16).ok()?.to_be_bytes();
        return texture::to_png(&texture::Image { width: 4, height: 4, rgba: [c[1], c[2], c[3], 255].repeat(16) }).ok();
    }
    let (_, i) = ctx.kind(texture);
    let ubulk = ctx.container_path(texture, "ubulk").and_then(|c| (ctx.load)(&c));
    let img = texture::decode_texture(ctx.pkg(texture)?, i, ubulk.as_deref()).ok()?;
    texture::to_png(&img).ok()
}

/// Decodes a Texture2D package to `<out_dir>/<name>.png`; returns the file name.
fn write_texture(ctx: &mut Ctx, texture: &str, out_dir: &Path) -> Option<String> {
    let png = texture_png(ctx, texture)?;
    let file = match texture.strip_prefix(COLOR) {
        Some(hex) => format!("color_{hex}.png"),
        None => format!("{}.png", sanitize(texture.rsplit('/').next()?)),
    };
    std::fs::write(out_dir.join(&file), png).ok()?;
    Some(file)
}

pub fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}

#[cfg(test)]
mod tests {
    use super::{role, Role};
    use std::path::Path;
    static UTOC: std::sync::LazyLock<String> = std::sync::LazyLock::new(crate::test_utoc);

    #[test]
    fn texture_roles() {
        assert_eq!(role("PackA", "/Game/T_Bush_02_RRGV"), None);
        assert_eq!(role("", "/Game/T_Bush_Mossy_02_C"), Some(Role::Diffuse));
        assert_eq!(role("BasecolorA", "/Game/T_Foo_CC"), Some(Role::Diffuse));
        assert_eq!(role("ColorMask", "/Game/T_Foo_CM"), None);
        assert_eq!(role("", "/Game/T_Rock_N"), Some(Role::Normal));
    }

    fn open() -> (crate::Toc, crate::global::Resolver, std::fs::File) {
        let toc = crate::open_archive(Path::new(UTOC.as_str())).unwrap();
        let res = crate::global::Resolver::load(&Path::new(UTOC.as_str()).with_file_name("global.utoc"), &toc).unwrap();
        let ucas = std::fs::File::open(crate::data_path(Path::new(UTOC.as_str()))).unwrap();
        (toc, res, ucas)
    }

    /// MAT="MI_Foo,M_Bar" cargo test --release material_textures_real -- --ignored --nocapture
    #[test]
    #[ignore]
    fn material_textures_real() {
        let (toc, res, mut ucas) = open();
        for want in std::env::var("MAT").unwrap().split(',') {
            let Some((path, _)) = toc.files.iter().find(|(p, _)| p.ends_with(&format!("/{want}.uasset"))) else { continue };
            let pkg = crate::global::container_path_to_package(path).unwrap();
            let mut load = |p: &str| -> Option<Vec<u8>> {
                let (_, e) = toc.files.iter().find(|(q, _)| q == p)?;
                crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).ok()
            };
            let t = super::material_preview(&res, &pkg, &mut load);
            println!("{pkg}: {} textures", t.len());
            for t in &t {
                println!("  {:<28} {:<8} {} png={}", t.param, t.role, t.texture, t.png.as_ref().map_or(0, |p| p.len()));
            }
            if let Ok(out) = std::env::var("OUT") {
                println!("  wrote {:?}", super::export_material(&res, &pkg, &mut load, Path::new(&out)).map(|f| f.len()));
            }
        }
    }

    /// EXPORT="SM_Coastal_Bush_01,SM_Foo" OUT=<dir> cargo test --release export_meshes -- --ignored --nocapture
    #[test]
    #[ignore]
    fn export_meshes() {
        let (toc, res, mut ucas) = open();
        let out = std::env::var("OUT").unwrap();
        for f in std::env::var("EXPORT").unwrap().split(',') {
            let (p, e) = toc.files.iter().find(|(p, _)| p.ends_with(&format!("/{f}.uasset"))).unwrap();
            let pkg = crate::zen::parse(crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).unwrap()).unwrap();
            let mut load = |path: &str| {
                let (_, e) = toc.files.iter().find(|(q, _)| q == path)?;
                crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).ok()
            };
            let ubulk = load(&format!("{}.ubulk", p.trim_end_matches(".uasset")));
            let i = (0..pkg.exports.len()).find(|&i| res.class_name(&pkg, i) == "StaticMesh").unwrap();
            let files = super::export_obj_with_materials(&res, &pkg, i, ubulk.as_deref(), &mut load, Path::new(&out)).unwrap();
            println!("{f}: {files:?}");
        }
    }

    /// Stats over every Nth StaticMesh (STEP, default 23): sections, slots, diffuse. Writes nothing.
    #[test]
    #[ignore]
    fn batch_stats() {
        let (toc, res, mut ucas) = open();
        let step: usize = std::env::var("STEP").map(|s| s.parse().unwrap()).unwrap_or(23);
        let meshes: Vec<_> = toc.files.iter().filter(|(p, _)| p.contains("/SM_") && p.ends_with(".uasset")).step_by(step).cloned().collect();
        let (mut n, mut fail, mut secs, mut slots_ok, mut diffuse, mut all_diffuse, mut multi) = (0, 0, 0, 0, 0, 0, 0);
        let mut params: HashMap<String, usize> = HashMap::new();
        use std::collections::HashMap;
        for (p, e) in &meshes {
            let Ok(pkg) = crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).and_then(crate::zen::parse) else { fail += 1; continue };
            let Some(i) = (0..pkg.exports.len()).find(|&i| res.class_name(&pkg, i) == "StaticMesh") else { continue };
            let mut load = |path: &str| {
                let (_, e) = toc.files.iter().find(|(q, _)| q == path)?;
                crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).ok()
            };
            let ubulk = load(&format!("{}.ubulk", p.trim_end_matches(".uasset")));
            let Ok(lods) = crate::mesh::decode_static_mesh(pkg.export_data(i), ubulk.as_deref()) else { fail += 1; continue };
            n += 1;
            let found = crate::mesh::find_sections(pkg.export_data(i), &lods[0]).is_some();
            secs += found as usize;
            multi += (lods[0].sections.len() > 1) as usize;
            let mut ctx = super::Ctx { res: &res, load: &mut load, pkgs: HashMap::new() };
            let d = pkg.export_data(i);
            let end = crate::mesh::find_sections(d, &lods[0]).map_or(d.len(), |(o, _)| o);
            let need = lods[0].sections.iter().map(|s| s.0 as usize + 1).max().unwrap_or(1);
            let (slots, ok) = super::slots(&mut ctx, &pkg, i, end, need);
            slots_ok += ok as usize;
            diffuse += slots.iter().any(|s| s.diffuse.is_some()) as usize;
            all_diffuse += (!slots.is_empty() && slots.iter().all(|s| s.diffuse.is_some())) as usize;
            for s in &slots {
                for (param, tex) in super::material_textures(&mut ctx, &s.material, 0) {
                    *params.entry(format!("{param} / {}", tex.rsplit('_').next().unwrap_or(""))).or_default() += 1;
                }
                if s.diffuse.is_none() {
                    println!("no diffuse: {p} slot {}", s.material);
                }
            }
            if !ok || !found {
                println!("sections={found} slots_ok={ok}: {p}");
            }
        }
        let mut pv: Vec<_> = params.into_iter().collect();
        pv.sort_by_key(|x| std::cmp::Reverse(x.1));
        println!("param/suffix: {:?}", &pv[..pv.len().min(60)]);
        println!("meshes {n} (decode fail {fail}) sections {secs} multi-section {multi} slots {slots_ok} any-diffuse {diffuse} all-diffuse {all_diffuse}");
    }
}
