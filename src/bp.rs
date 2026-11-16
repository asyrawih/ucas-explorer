//! Blueprint actor assembly: SimpleConstructionScript component tree -> mesh parts placed with their
//! world transforms, merged into one mesh for preview / OBJ export.
use crate::global::{self, Resolver};
use crate::props::{self, Drift, Mappings, Value};
use crate::zen::ZenPackage;
use crate::{material, mesh, skeletal, Res};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Schema drift of this build vs the 2024 mappings, measured on BP component templates (see the
/// gen_classes test): where RelativeLocation/RelativeScale3D sit per class, e.g. the SceneComponent
/// part starts at build slot 160 for StaticMeshComponent (mappings: 145), 119 for SphereComponent
/// (114), 227 for SkeletalMeshComponent (215); StaticMesh is still slot 6, OverrideMaterials
/// (MeshComponent slot 0) build slot 45. SceneComponent / ActorComponent themselves are unchanged.
// ponytail: hand-measured; which struct of a chain grew is a guess where only the sum is known (the
// grown struct is then marked unknown). Classes deriving from a listed struct inherit its drift,
// others are assumed unchanged; a wrong guess shows up as unread (unknown) transforms, since every
// value read after a changed range must parse sanely and end at a known trailer.
const DRIFT: &[Drift] = &[
    Drift { name: "StaticMeshComponent", slots: 45, known: 7 },
    Drift { name: "PrimitiveComponent", slots: 111, known: 0 },
    Drift { name: "InstancedStaticMeshComponent", slots: 20, known: 0 },
    Drift { name: "SkeletalMeshComponent", slots: 72, known: 0 },
    Drift { name: "SkinnedMeshComponent", slots: 40, known: 0 },
    Drift { name: "NiagaraComponent", slots: 22, known: 4 },
    Drift { name: "BoxComponent", slots: 1, known: 0 },
    Drift { name: "LightComponent", slots: 20, known: 0 },
    Drift { name: "ChildActorComponent", slots: 5, known: 1 },
    Drift { name: "TextRenderComponent", slots: 12, known: 0 },
];

/// The slot table above was measured for one specific mappings file (Palia, 2024-09-25: 10066 structs,
/// 1985 enums) against the newer Palia build. For any other usmap (another game, or a matching dump
/// of this game) it must not apply — those mappings are assumed to fit.
// ponytail: identified by struct/enum counts; use a content hash if two usmaps ever collide.
fn drift_for(maps: &props::Mappings) -> &'static [Drift] {
    if maps.structs.len() == 10066 && maps.enums.len() == 1985 { DRIFT } else { &[] }
}

/// What follows the properties of a component template (UObject + native data): SceneComponent-
/// derived templates end with 8 zero bytes; StaticMeshComponent-derived ones with these 20 bytes
/// (ISM: then the instance data).
const TAILS: &[(&[u8], bool)] = &[(&[0; 8], true), (&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], false)];

/// UE FTransform: rotation quaternion (x, y, z, w), translation, scale. UE space (cm, Z up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xf {
    pub rot: [f64; 4],
    pub trans: [f64; 3],
    pub scale: [f64; 3],
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

impl Xf {
    pub const IDENTITY: Xf = Xf { rot: [0.0, 0.0, 0.0, 1.0], trans: [0.0; 3], scale: [1.0; 3] };

    /// UE FRotator::Quaternion (degrees; yaw about Z, pitch about Y, roll about X).
    pub fn rotator(pitch: f64, yaw: f64, roll: f64) -> [f64; 4] {
        let h = std::f64::consts::PI / 360.0;
        let ((sp, cp), (sy, cy), (sr, cr)) = ((pitch * h).sin_cos(), (yaw * h).sin_cos(), (roll * h).sin_cos());
        [cr * sp * sy - sr * cp * cy, -cr * sp * cy - sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]
    }

    /// UE FQuat::Rotator (pitch, yaw, roll in degrees), inverse of `rotator` away from pitch +-90.
    pub fn to_rotator(q: [f64; 4]) -> [f64; 3] {
        let [x, y, z, w] = q;
        let pitch = (2.0 * (z * x - w * y)).clamp(-1.0, 1.0).asin().to_degrees();
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z)).to_degrees();
        let roll = (-2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)).to_degrees();
        [pitch, yaw, roll]
    }

    pub fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
        let u = [q[0], q[1], q[2]];
        let t = cross(u, v).map(|x| 2.0 * x);
        let c = cross(u, t);
        std::array::from_fn(|k| v[k] + q[3] * t[k] + c[k])
    }

    /// `self * parent` as UE FTransform::Multiply: child relative -> child world.
    pub fn then(&self, parent: &Xf) -> Xf {
        let (a, b) = (parent.rot, self.rot); // rotation = parent.rot * self.rot
        let rot = [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ];
        let t = Xf::rotate(parent.rot, std::array::from_fn(|k| parent.scale[k] * self.trans[k]));
        Xf { rot, trans: std::array::from_fn(|k| t[k] + parent.trans[k]), scale: std::array::from_fn(|k| self.scale[k] * parent.scale[k]) }
    }

    pub fn point(&self, v: [f64; 3]) -> [f64; 3] {
        let r = Xf::rotate(self.rot, std::array::from_fn(|k| v[k] * self.scale[k]));
        std::array::from_fn(|k| r[k] + self.trans[k])
    }

    /// Normal direction (inverse scale, then rotation), normalized.
    pub fn normal(&self, n: [f64; 3]) -> [f64; 3] {
        let s = std::array::from_fn(|k| if self.scale[k] == 0.0 { 0.0 } else { n[k] / self.scale[k] });
        norm(Xf::rotate(self.rot, s))
    }
}

fn norm(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 0.0 { v.map(|x| x / l) } else { v }
}

/// One scene component of the actor.
#[derive(Clone)]
pub struct Part {
    pub name: String,  // component variable name (child actors: "<component>/<name>")
    pub class: String, // component class
    /// StaticMesh / SkeletalMesh package ("/Game/.../SM_Foo"), if any
    pub mesh: Option<String>,
    /// transform relative to the parent component, and whether it was fully read
    pub rel: Xf,
    pub rel_known: bool,
    /// index of the parent component in `Assembly::parts` (None: attached to the actor)
    pub parent: Option<usize>,
    pub world: Xf,
    /// false: some transform on the way from the root wasn't readable (world is a guess)
    pub transform_known: bool,
    /// (Hierarchical)InstancedStaticMesh: per-instance matrices relative to the component (UE FMatrix,
    /// row vectors); None = not instanced
    pub instances: Option<Vec<[[f64; 4]; 4]>>,
    /// OverrideMaterials (material package per slot)
    pub materials: Vec<Option<String>>,
    /// socket or bone of the parent's mesh it is attached to (applied by apply_sockets, then cleared)
    pub socket: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone)]
pub struct Assembly {
    pub name: String,
    /// parent Blueprint package, if the class derives from another BP
    pub parent: Option<String>,
    pub parts: Vec<Part>,
    pub warnings: Vec<String>,
}

pub type Load<'a> = &'a mut dyn FnMut(&str) -> Option<Vec<u8>>;

/// "/Game/X/Y" -> parsed package, via the resolver's package map and `load` (container paths).
fn load_pkg(res: &Resolver, package: &str, load: Load) -> Option<(ZenPackage, String)> {
    let path = res.packages.get(&global::package_id(package))?;
    Some((crate::zen::parse(load(path)?).ok()?, path.clone()))
}

/// Object path -> its package ("/Game/X/Y.Z:W" -> "/Game/X/Y").
fn package_of(path: &str) -> &str {
    path.split(['.', ':']).next().unwrap_or(path)
}

fn get<'a>(props: &'a [(String, Value)], name: &str) -> Option<&'a Value> {
    props.iter().find(|p| p.0 == name).map(|p| &p.1)
}

fn obj(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::Object(s)) if s != "None" => Some(s.clone()),
        _ => None,
    }
}

fn floats(v: Option<&Value>, names: [&str; 3]) -> Option<[f64; 3]> {
    let Some(Value::Struct(_, f)) = v else { return None };
    let g = |n: &str| match get(f, n) {
        Some(Value::Float(x)) => Some(*x),
        _ => None,
    };
    Some([g(names[0])?, g(names[1])?, g(names[2])?])
}

/// Object path -> export index, for every export of `pkg`.
fn export_index(res: &Resolver, pkg: &ZenPackage) -> HashMap<String, usize> {
    (0..pkg.exports.len()).filter_map(|i| Some((res.object_path(i as u64, pkg)?, i))).collect()
}

/// Legacy UE4 package: tagged properties, so an absent property simply has its class default.
fn tagged(pkg: &ZenPackage) -> bool {
    pkg.flags & 0x2000 == 0
}

fn is_mesh_class(c: &str) -> bool {
    c.contains("StaticMeshComponent") || c.contains("SkeletalMeshComponent") || c.contains("SkinnedMeshComponent")
}

struct Node {
    name: String,
    template: Option<usize>,
    children: Vec<usize>,
    parent_name: Option<String>, // root nodes: inherited component to attach to
    socket: Option<String>,
}

/// Assembles the actor of BP package `pkg` (the package holding a BlueprintGeneratedClass).
pub fn assemble(res: &Resolver, maps: &Mappings, pkg: &ZenPackage, load: Load) -> Res<Assembly> {
    let mut asm = assemble_depth(res, maps, pkg, load, 0)?;
    apply_sockets(res, maps, &mut asm, load);
    // world transforms down the hierarchy (parents always precede their children)
    for i in 0..asm.parts.len() {
        let (pw, pk) = match asm.parts[i].parent {
            Some(p) if p < i => (asm.parts[p].world, asm.parts[p].transform_known),
            _ => (Xf::IDENTITY, true),
        };
        let part = &mut asm.parts[i];
        part.world = part.rel.then(&pw);
        part.transform_known = part.rel_known && pk;
        if !pk {
            part.warnings.push("a parent's transform is unknown".into());
        }
    }
    Ok(asm)
}

const MAX_DEPTH: u32 = 4;

/// Asset classes shown through their Skeleton's (or PhysicsAsset's) SkeletalMesh.
pub const SKELETON_USERS: &[&str] = &[
    "AnimBlueprintGeneratedClass", "AnimSequence", "AnimMontage", "AnimComposite", "AnimStreamable", "BlendSpace", "BlendSpace1D",
    "AimOffsetBlendSpace", "AimOffsetBlendSpace1D", "PoseAsset", "Skeleton", "PhysicsAsset",
];

/// Classes the 3D view picks by default (it works on any export: see assemble_any).
pub fn is_3d_class(class: &str) -> bool {
    matches!(class, "BlueprintGeneratedClass" | "World" | "Level") || SKELETON_USERS.contains(&class)
}

/// The export a package's 3D view starts from: World for maps (not their level script Blueprint),
/// else the first 3D-able class.
pub fn main_3d_export(res: &Resolver, pkg: &ZenPackage) -> Option<usize> {
    let class = |i: usize| res.class_name(pkg, i);
    (0..pkg.exports.len()).find(|&i| class(i) == "World").or_else(|| (0..pkg.exports.len()).find(|&i| is_3d_class(&class(i))))
}

/// Main export of a package (named like it; Blueprints "<name>_C"), else the first.
pub fn main_export(pkg: &ZenPackage) -> usize {
    let short = pkg.name.rsplit('/').next().unwrap_or("");
    (0..pkg.exports.len()).find(|&i| pkg.exports[i].name == short || pkg.exports[i].name.strip_suffix("_C") == Some(short)).unwrap_or(0)
}

/// Cheap guess whether the 3D view of a data asset has something to show: it imports a package named
/// like a mesh (SK_*, SM_*). The 3D tab works on any export regardless.
pub fn references_meshes(pkg: &ZenPackage) -> bool {
    pkg.imported_package_names.iter().any(|n| {
        let s = n.rsplit('/').next().unwrap_or("").to_ascii_lowercase();
        s.starts_with("sk_") || s.starts_with("sm_")
    })
}

/// A part showing mesh package `mesh` untransformed.
fn mesh_part(name: &str, class: &str, mesh: String) -> Part {
    Part {
        name: name.into(),
        class: class.into(),
        mesh: Some(mesh),
        rel: Xf::IDENTITY,
        rel_known: true,
        parent: None,
        world: Xf::IDENTITY,
        transform_known: true,
        instances: None,
        materials: vec![],
        socket: None,
        warnings: vec![],
    }
}

/// 3D view of any export: Blueprints and maps are assembled from their components; animation assets,
/// AnimBlueprints, Skeletons and PhysicsAssets show a SkeletalMesh using their Skeleton; anything else
/// shows the Static/SkeletalMeshes its properties reference.
pub fn assemble_any(res: &Resolver, maps: &Mappings, pkg: &ZenPackage, export: usize, load: Load) -> Res<Assembly> {
    let class = res.class_name(pkg, export);
    let short = pkg.name.rsplit('/').next().unwrap_or(&pkg.name).to_string();
    if class == "BlueprintGeneratedClass" {
        return assemble(res, maps, pkg, load);
    }
    if class == "World" || class == "Level" {
        return assemble_level(res, maps, pkg, load);
    }
    let mut asm = Assembly { name: short, parent: None, parts: vec![], warnings: vec![] };
    if SKELETON_USERS.contains(&class.as_str()) {
        let target = match class.as_str() {
            "Skeleton" | "PhysicsAsset" => Some(pkg.name.clone()),
            _ => {
                let (p, _) = props::read_properties_partial(pkg, export, &class, maps, res);
                obj(get(&p, "TargetSkeleton")).or_else(|| obj(get(&p, "Skeleton"))).map(|s| package_of(&s).to_string()).or_else(|| crate::anim::skeleton_of(res, pkg, load))
            }
        };
        let Some(target) = target else {
            asm.warnings.push("Skeleton not found".into());
            return Ok(asm);
        };
        match crate::anim::find_skeletal_mesh(res, &target, None, load) {
            Some((p, _, _)) => asm.parts.push(mesh_part("SkeletalMesh", "SkeletalMesh", p.name)),
            // no mesh uses it (e.g. an orphaned Skeleton): show the bones themselves
            None if class != "PhysicsAsset" => {
                asm.warnings.push(format!("no SkeletalMesh uses {target}: showing its bones"));
                asm.parts.push(mesh_part("Bones", "Skeleton", format!("{BONES}{target}")));
            }
            None => asm.warnings.push(format!("no SkeletalMesh uses {target}")),
        }
        return Ok(asm);
    }
    // referenced meshes: object/soft references in the properties, plus imported packages
    fn walk(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(s) | Value::SoftObject(s) if s != "None" => out.push(package_of(s).to_string()),
            Value::Struct(_, f) => f.iter().for_each(|x| walk(&x.1, out)),
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Map(m) => m.iter().for_each(|(k, x)| {
                walk(k, out);
                walk(x, out);
            }),
            _ => {}
        }
    }
    // the export first, then the package's other exports (data assets keep meshes in subobjects, e.g.
    // ItemSkin -> SkeletalMeshSkinEffect_0.Mesh)
    let mut refs = vec![];
    for i in std::iter::once(export).chain((0..pkg.exports.len()).filter(|&i| i != export)) {
        let (p, _) = props::read_properties_partial(pkg, i, &res.class_name(pkg, i), maps, res);
        p.iter().for_each(|x| walk(&x.1, &mut refs));
        refs.extend(material::package_refs(pkg, pkg.export_data(i)).into_iter().map(|r| r.1));
    }
    let mut seen = std::collections::HashSet::new();
    for r in refs.into_iter().filter(|r| !r.starts_with("/Script/") && *r != pkg.name) {
        if asm.parts.len() >= 64 || !seen.insert(r.to_lowercase()) {
            continue;
        }
        if let Some((mp, _)) = load_pkg(res, &r, load) {
            if let Some(i) = (0..mp.exports.len()).find(|&i| matches!(res.class_name(&mp, i).as_str(), "StaticMesh" | "SkeletalMesh")) {
                asm.parts.push(mesh_part(&mp.exports[i].name, &res.class_name(&mp, i), r));
            }
        }
    }
    if asm.parts.is_empty() {
        asm.warnings.push(format!("{class}: no 3D content (no meshes referenced)"));
    }
    layout_row(res, &mut asm, load);
    Ok(asm)
}

/// Data assets carry no placement: SkeletalMeshes stay together at the origin (they share a character
/// skeleton: head, beard, armor...), StaticMeshes are lined up next to them along +Y, spaced by bounds.
fn layout_row(res: &Resolver, asm: &mut Assembly, load: Load) {
    let bounds = |p: &Part, load: Load| -> Option<([f64; 3], [f64; 3])> {
        let m = load_mesh(res, p.mesh.as_deref()?, 2_000, load).ok()?.mesh;
        let (lo, hi) = m.positions.iter().fold(([f64::MAX; 3], [f64::MIN; 3]), |(lo, hi), v| {
            (std::array::from_fn(|k| lo[k].min(v[k] as f64)), std::array::from_fn(|k| hi[k].max(v[k] as f64)))
        });
        (lo[0] <= hi[0]).then_some((lo, hi))
    };
    let statics: Vec<usize> = (0..asm.parts.len()).filter(|&i| asm.parts[i].class == "StaticMesh").collect();
    if statics.len() + usize::from(statics.len() < asm.parts.len()) < 2 {
        return; // a single mesh (or one group): nothing to arrange
    }
    // start right of the skeletal group
    let start = (0..asm.parts.len()).filter(|i| !statics.contains(i)).filter_map(|i| bounds(&asm.parts[i], load)).map(|b| b.1[1]).reduce(f64::max);
    let b: Vec<Option<([f64; 3], [f64; 3])>> = statics.iter().map(|&i| bounds(&asm.parts[i], load)).collect();
    for (&i, y) in statics.iter().zip(row_offsets(&b, start)) {
        if let Some(y) = y {
            asm.parts[i].rel.trans[1] = y;
            asm.parts[i].world = asm.parts[i].rel;
        }
    }
}

/// Y offset per mesh (by its bounds) so they sit side by side along +Y, a 15% gap apart, starting right
/// of `start` (the right edge of what is already at the origin) or at 0. None for unknown bounds.
fn row_offsets(bounds: &[Option<([f64; 3], [f64; 3])>], start: Option<f64>) -> Vec<Option<f64>> {
    let mut edge = start;
    bounds
        .iter()
        .map(|b| {
            let (lo, hi) = (*b)?;
            let gap = 0.15 * (hi[1] - lo[1]).max(hi[0] - lo[0]).max(10.0);
            let at = match edge {
                Some(e) => e + gap - lo[1],
                None => -lo[1],
            };
            edge = Some(at + hi[1]);
            Some(at)
        })
        .collect()
}

/// Full level: the map itself plus everything streamed into it, merged in world space: streaming
/// sublevels (LevelStreaming*.WorldAsset, placed by LevelTransform, recursively) and World Partition
/// content found by path (cooked cells "<map>/_Generated_/*", uncooked "/<mount>/__ExternalActors__/<map>/*").
pub fn assemble_level(res: &Resolver, maps: &Mappings, pkg: &ZenPackage, load: Load) -> Res<Assembly> {
    let mut asm = level_one(res, maps, pkg, load)?;
    let mut seen = std::collections::HashSet::from([pkg.name.to_lowercase()]);
    let mut skipped = vec![];
    let mut queue: Vec<(String, Xf)> = level_children(res, maps, pkg, &Xf::IDENTITY, &mut skipped);
    let mut subs = vec![];
    while let Some((path, xf)) = queue.pop() {
        if !seen.insert(path.to_lowercase()) {
            continue;
        }
        let Some((sub, _)) = load_pkg(res, &path, load) else {
            asm.warnings.push(format!("sublevel {path}: package not found"));
            continue;
        };
        queue.extend(level_children(res, maps, &sub, &xf, &mut skipped));
        match level_one(res, maps, &sub, load) {
            Ok(s) => {
                subs.push(path.rsplit('/').next().unwrap_or(&path).to_string());
                let base = asm.parts.len();
                asm.parts.extend(s.parts.into_iter().map(|mut p| {
                    p.parent = p.parent.map(|q| q + base);
                    p.world = p.world.then(&xf);
                    p
                }));
                asm.warnings.extend(s.warnings.into_iter().map(|w| format!("{path}: {w}")));
            }
            Err(e) => asm.warnings.push(format!("sublevel {path}: {e}")),
        }
    }
    if !subs.is_empty() {
        asm.warnings.push(format!("{} sublevel/World Partition package(s) merged: {}", subs.len(), subs.join(", ")));
    }
    if !skipped.is_empty() {
        asm.warnings.push(format!("{} sublevel(s) the game streams in at runtime not shown (events/variants): {}", skipped.len(), skipped.join(", ")));
    }
    if !asm.parts.iter().any(|p| p.mesh.is_some()) {
        asm.warnings.push("no mesh components in this map".into());
    }
    Ok(asm)
}

/// Packages streamed into level `pkg` (whose own placement is `xf`), each with its world placement.
/// Dynamic sublevels not initially loaded (loaded by game code: seasonal/event variants that would
/// overlap each other) go to `skipped` instead.
fn level_children(res: &Resolver, maps: &Mappings, pkg: &ZenPackage, xf: &Xf, skipped: &mut Vec<String>) -> Vec<(String, Xf)> {
    let mut out = vec![];
    for i in 0..pkg.exports.len() {
        let class = res.class_name(pkg, i);
        if !class.contains("LevelStreaming") {
            continue;
        }
        let (p, _) = props::read_properties_partial(pkg, i, &class, maps, res);
        let world = match get(&p, "WorldAsset") {
            Some(Value::SoftObject(s) | Value::Object(s)) if s != "None" && !s.is_empty() => package_of(s).to_string(),
            _ => continue,
        };
        if class.contains("Dynamic") && !matches!(get(&p, "bInitiallyLoaded"), Some(Value::Bool(true))) {
            skipped.push(world.rsplit('/').next().unwrap_or(&world).to_string());
            continue;
        }
        out.push((world, level_xf(get(&p, "LevelTransform")).unwrap_or(Xf::IDENTITY).then(xf)));
    }
    // World Partition: no references to follow, the packages are found by their path
    let name = pkg.name.to_lowercase();
    let generated = format!("{name}/_generated_/");
    let external = name.get(1..).and_then(|s| s.split_once('/')).map(|(mount, rest)| format!("/{mount}/__externalactors__/{rest}/"));
    let mut wp: Vec<String> = res
        .packages
        .values()
        .filter_map(|c| global::container_path_to_package(c))
        .filter(|p| {
            let l = p.to_lowercase();
            l.starts_with(&generated) || external.as_deref().is_some_and(|e| l.starts_with(e))
        })
        .collect();
    wp.sort();
    out.extend(wp.into_iter().map(|p| (p, *xf)));
    out
}

/// FTransform property value (Rotation quat, Translation, Scale3D) -> Xf; None if unreadable.
fn level_xf(v: Option<&Value>) -> Option<Xf> {
    let Some(Value::Struct(_, f)) = v else { return None };
    let w = match get(f, "Rotation") {
        Some(Value::Struct(_, q)) => match get(q, "W") {
            Some(Value::Float(w)) => *w,
            _ => return None,
        },
        _ => return None,
    };
    let r = floats(get(f, "Rotation"), ["X", "Y", "Z"])?;
    Some(Xf {
        rot: [r[0], r[1], r[2], w],
        trans: floats(get(f, "Translation"), ["X", "Y", "Z"])?,
        scale: floats(get(f, "Scale3D"), ["X", "Y", "Z"]).unwrap_or([1.0; 3]),
    })
}

/// One map package: every scene component of every actor, placed by its AttachParent chain.
/// Values a level component doesn't override come from its archetype (the Blueprint's component template).
fn level_one(res: &Resolver, maps: &Mappings, pkg: &ZenPackage, load: Load) -> Res<Assembly> {
    let short = pkg.name.rsplit('/').next().unwrap_or(&pkg.name).to_string();
    let mut asm = Assembly { name: short, parent: None, parts: vec![], warnings: vec![] };
    let idx = export_index(res, pkg);
    let mut cache: HashMap<String, Option<std::rc::Rc<ZenPackage>>> = HashMap::new();
    let mut parts: Vec<(usize, Part, Option<usize>)> = vec![]; // (export, part, AttachParent export)
    for i in 0..pkg.exports.len() {
        let class = res.class_name(pkg, i);
        if !class.ends_with("Component") {
            continue;
        }
        let name = pkg.exports[i].name.clone();
        let base = archetype(res, maps, pkg, i, &name, &mut cache, load, 0);
        let Some(part) = component(res, maps, pkg, i, &name, base.as_ref(), load) else { continue };
        let (p, _) = props::read_properties_partial(pkg, i, &class, maps, res);
        let parent = obj(get(&p, "AttachParent")).and_then(|x| idx.get(&x).copied());
        parts.push((i, part, parent));
    }
    // mesh-less leaves are dropped
    let has_child: std::collections::HashSet<usize> = parts.iter().filter_map(|p| p.2).collect();
    parts.retain(|p| p.1.mesh.is_some() || has_child.contains(&p.0));
    let pos: HashMap<usize, usize> = parts.iter().enumerate().map(|(k, p)| (p.0, k)).collect();
    for (k, (_, mut part, parent)) in parts.into_iter().enumerate() {
        part.parent = parent.and_then(|e| pos.get(&e).copied()).filter(|&j| j != k);
        asm.parts.push(part);
    }
    let order = topo(&asm.parts, 0);
    let moved: Vec<Part> = order.iter().map(|&j| asm.parts[j].clone()).collect();
    let remap: HashMap<usize, usize> = order.iter().enumerate().map(|(new, &old)| (old, new)).collect();
    asm.parts = moved.into_iter().map(|mut p| {
        p.parent = p.parent.map(|q| remap[&q]);
        p
    }).collect();
    apply_sockets(res, maps, &mut asm, load);
    for i in 0..asm.parts.len() {
        let (pw, pk) = match asm.parts[i].parent {
            Some(p) if p < i => (asm.parts[p].world, asm.parts[p].transform_known),
            _ => (Xf::IDENTITY, true),
        };
        let part = &mut asm.parts[i];
        part.world = part.rel.then(&pw);
        part.transform_known = part.rel_known && pk;
    }
    Ok(asm)
}

/// Folds each part's attach socket into its relative transform: the socket's transform on the parent's
/// mesh (StaticMeshSocket; SkeletalMeshSocket of the mesh or its Skeleton, relative to its bone), or the
/// bind-pose transform of a bone of that name.
fn apply_sockets(res: &Resolver, maps: &Mappings, asm: &mut Assembly, load: Load) {
    let mut cache: HashMap<(String, String), Option<Xf>> = HashMap::new();
    for i in 0..asm.parts.len() {
        let Some(sock) = asm.parts[i].socket.take() else { continue };
        let Some(mesh) = asm.parts[i].parent.and_then(|p| asm.parts.get(p)).and_then(|p| p.mesh.clone()) else {
            asm.parts[i].warnings.push(format!("attached to socket {sock}, but the parent has no mesh: offset ignored"));
            continue;
        };
        let xf = *cache.entry((mesh.clone(), sock.clone())).or_insert_with(|| socket_xf(res, maps, &mesh, &sock, load));
        match xf {
            Some(x) => asm.parts[i].rel = asm.parts[i].rel.then(&x),
            None => asm.parts[i].warnings.push(format!("socket {sock} not found on {}: offset ignored", mesh.rsplit('/').next().unwrap_or(&mesh))),
        }
    }
}

/// Transform of socket (or bone) `name` in the component space of mesh package `mesh`.
fn socket_xf(res: &Resolver, maps: &Mappings, mesh: &str, name: &str, load: Load) -> Option<Xf> {
    let (pkg, path) = load_pkg(res, mesh, load)?;
    let mi = (0..pkg.exports.len()).find(|&i| matches!(res.class_name(&pkg, i).as_str(), "StaticMesh" | "SkeletalMesh"))?;
    let skeletal = res.class_name(&pkg, mi) == "SkeletalMesh";
    // (relative transform, bone) of a socket object named `name` in package `p`
    let find = |p: &ZenPackage| -> Option<(Xf, Option<String>)> {
        (0..p.exports.len()).find_map(|i| {
            let class = res.class_name(p, i);
            if !class.ends_with("MeshSocket") {
                return None;
            }
            let (v, _) = props::read_properties_partial(p, i, &class, maps, res);
            if !matches!(get(&v, "SocketName"), Some(Value::Name(n)) if n.eq_ignore_ascii_case(name)) {
                return None;
            }
            let loc = floats(get(&v, "RelativeLocation"), ["X", "Y", "Z"]).unwrap_or([0.0; 3]);
            let rot = floats(get(&v, "RelativeRotation"), ["Pitch", "Yaw", "Roll"]).unwrap_or([0.0; 3]);
            let scale = floats(get(&v, "RelativeScale"), ["X", "Y", "Z"]).unwrap_or([1.0; 3]);
            let bone = match get(&v, "BoneName") {
                Some(Value::Name(b)) if b != "None" => Some(b.clone()),
                _ => None,
            };
            Some((Xf { rot: Xf::rotator(rot[0], rot[1], rot[2]), trans: loc, scale }, bone))
        })
    };
    let mut hit = find(&pkg);
    if !skeletal {
        return hit.map(|h| h.0);
    }
    if hit.is_none() {
        // sockets defined on the Skeleton
        let (v, _) = props::read_properties_partial(&pkg, mi, "SkeletalMesh", maps, res);
        let skel = obj(get(&v, "Skeleton")).map(|s| package_of(&s).to_string()).or_else(|| crate::anim::skeleton_of(res, &pkg, load));
        hit = skel.and_then(|s| load_pkg(res, &s, load)).and_then(|(sp, _)| find(&sp));
    }
    let bones = skeletal::find_reference_skeleton(pkg.export_data(mi), &pkg.names)?;
    let bone_xf = |b: &str| -> Option<Xf> {
        let mut i = bones.iter().position(|x| x.name.eq_ignore_ascii_case(b))?;
        let mut xf = Xf::IDENTITY;
        for _ in 0..bones.len() {
            let bn = &bones[i];
            let local = Xf { rot: bn.rotation.map(|v| v as f64), trans: bn.translation.map(|v| v as f64), scale: bn.scale.map(|v| v as f64) };
            xf = xf.then(&local);
            if bn.parent < 0 {
                break;
            }
            i = bn.parent as usize;
        }
        Some(xf)
    };
    let _ = path;
    match hit {
        Some((rel, bone)) => Some(match bone.and_then(|b| bone_xf(&b)) {
            Some(b) => rel.then(&b),
            None => rel,
        }),
        None => bone_xf(name), // attached straight to a bone
    }
}

/// The part a component's archetype (template export in a Blueprint package) describes, recursively.
#[allow(clippy::too_many_arguments)]
fn archetype(res: &Resolver, maps: &Mappings, pkg: &ZenPackage, i: usize, name: &str, cache: &mut HashMap<String, Option<std::rc::Rc<ZenPackage>>>, load: Load, depth: u32) -> Option<Part> {
    let t = pkg.exports[i].template_index;
    if depth > MAX_DEPTH || t >> 62 != 2 {
        return None;
    }
    let path = res.object_path(t, pkg)?;
    let pk = package_of(&path).to_string();
    let bp = cache.entry(pk.clone()).or_insert_with(|| load_pkg(res, &pk, load).map(|(p, _)| std::rc::Rc::new(p))).clone()?;
    let j = *export_index(res, &bp).get(&path)?;
    let base = archetype(res, maps, &bp, j, name, cache, load, depth + 1);
    component(res, maps, &bp, j, name, base.as_ref(), load)
}

/// Appends `sub`'s parts under part `under` (None: the actor), keeping their hierarchy.
fn adopt(parts: &mut Vec<Part>, sub: Vec<Part>, under: Option<usize>, prefix: &str) {
    let base = parts.len();
    for mut p in sub {
        p.parent = p.parent.map(|j| j + base).or(under);
        p.name = format!("{prefix}{}", p.name);
        parts.push(p);
    }
}

fn assemble_depth(res: &Resolver, maps: &Mappings, pkg: &ZenPackage, load: Load, depth: u32) -> Res<Assembly> {
    let short = pkg.name.rsplit('/').next().unwrap_or(&pkg.name).to_string();
    let bgc = (0..pkg.exports.len()).find(|&i| res.class_name(pkg, i) == "BlueprintGeneratedClass").ok_or("no BlueprintGeneratedClass export")?;
    let mut asm = Assembly { name: short, parent: None, parts: vec![], warnings: vec![] };
    let idx = export_index(res, pkg);
    let read = |i: usize| props::read_properties_partial(pkg, i, &res.class_name(pkg, i), maps, res);
    let (cprops, err) = read(bgc);
    if let Some(e) = err {
        asm.warnings.push(format!("class properties: {e}"));
    }
    let list = |v: Option<&Value>| -> Vec<Value> {
        match v {
            Some(Value::Array(a)) => a.clone(),
            _ => vec![],
        }
    };

    // parent BP class: its components come first (ours may attach to them), with this BP's
    // overrides of their templates (InheritableComponentHandler) applied
    let parent = res.object_path(pkg.exports[bgc].super_index, pkg).unwrap_or_default();
    if !parent.is_empty() && !parent.starts_with("/Script/") {
        let ppkg = package_of(&parent).to_string();
        asm.parent = Some(ppkg.clone());
        if depth >= MAX_DEPTH {
            asm.warnings.push(format!("parent BP {ppkg} not followed (depth limit)"));
        } else {
            match load_pkg(res, &ppkg, load).map(|(p, _)| assemble_depth(res, maps, &p, load, depth + 1)) {
                Some(Ok(pa)) => {
                    asm.warnings.extend(pa.warnings.into_iter().map(|w| format!("parent {}: {w}", pa.name)));
                    adopt(&mut asm.parts, pa.parts, None, "");
                }
                Some(Err(e)) => asm.warnings.push(format!("parent BP {ppkg}: {e}")),
                None => asm.warnings.push(format!("parent BP {ppkg}: package not found")),
            }
        }
    }
    if let Some(ich) = obj(get(&cprops, "InheritableComponentHandler")).and_then(|p| idx.get(&p).copied()) {
        let (ip, err) = read(ich);
        if let Some(e) = err {
            asm.warnings.push(format!("InheritableComponentHandler: {e}"));
        }
        for rec in list(get(&ip, "Records")) {
            let Value::Struct(_, f) = rec else { continue };
            let name = match get(f.as_slice(), "ComponentKey") {
                Some(Value::Struct(_, k)) => match get(k, "SCSVariableName") {
                    Some(Value::Name(n)) => n.clone(),
                    _ => continue,
                },
                _ => continue,
            };
            let Some(t) = obj(get(&f, "ComponentTemplate")).and_then(|p| idx.get(&p).copied()) else { continue };
            match asm.parts.iter().position(|p| p.name == name) {
                Some(k) => {
                    let base = asm.parts[k].clone();
                    if let Some(mut o) = component(res, maps, pkg, t, &name, Some(&base), load) {
                        (o.parent, o.warnings) = (base.parent, [base.warnings, o.warnings].concat());
                        if o.class == "ChildActorComponent" {
                            o.warnings.push("child actor override: class change not followed".into());
                        }
                        asm.parts[k] = o;
                    }
                }
                // ponytail: overrides of native or skipped (unmapped, mesh-less) components are dropped silently
                None if maps.structs.contains_key(&res.class_name(pkg, t)) && is_mesh_class(&res.class_name(pkg, t)) => {
                    asm.warnings.push(format!("override of inherited mesh component {name}: not in the parent BP (native component?), ignored"))
                }
                None => {}
            }
        }
    }

    // native components (C++ default subobjects) live in the package as subobjects of the class default
    // object when the BP sets any of their properties; tagged packages let us read them like SCS templates
    if tagged(pkg) {
        let cdo = (0..pkg.exports.len()).find(|&i| pkg.exports[i].class_index == bgc as u64 && pkg.exports[i].name.starts_with("Default__"));
        let subs: Vec<usize> = (0..pkg.exports.len()).filter(|&i| cdo.is_some_and(|c| pkg.exports[i].outer_index == c as u64)).collect();
        let base = asm.parts.len();
        let mut attach = vec![];
        for &i in &subs {
            let name = pkg.exports[i].name.clone();
            if asm.parts.iter().any(|p| p.name == name) {
                continue; // an inherited BP component of that name was already placed
            }
            let Some(part) = component(res, maps, pkg, i, &name, None, load) else { continue };
            let props = read(i).0;
            let parent = obj(get(&props, "AttachParent"));
            // ponytail: no schema, so "scene component" = has a mesh, a transform or an attach parent
            let scene = part.mesh.is_some() || parent.is_some() || ["RelativeLocation", "RelativeRotation", "RelativeScale3D"].iter().any(|n| get(&props, n).is_some());
            if scene {
                attach.push(parent.and_then(|p| idx.get(&p).copied()));
                asm.parts.push(part);
            }
        }
        // attach to the native parent component, if it is one of the parts just added
        for (k, parent) in attach.into_iter().enumerate() {
            let pname = parent.map(|e| pkg.exports[e].name.as_str());
            asm.parts[base + k].parent = pname.and_then(|n| asm.parts[base..].iter().position(|p| p.name == n)).map(|j| base + j).filter(|&j| j != base + k);
        }
        // parents must precede children for the world-transform pass
        let order = topo(&asm.parts[base..], base);
        let moved: Vec<Part> = order.iter().map(|&j| asm.parts[base + j].clone()).collect();
        let remap: HashMap<usize, usize> = order.iter().enumerate().map(|(new, &old)| (base + old, base + new)).collect();
        for (k, mut p) in moved.into_iter().enumerate() {
            p.parent = p.parent.map(|q| remap.get(&q).copied().unwrap_or(q));
            asm.parts[base + k] = p;
        }
    }

    let scs = obj(get(&cprops, "SimpleConstructionScript")).and_then(|p| idx.get(&p).copied());
    let Some(scs) = scs else {
        if asm.parts.is_empty() {
            asm.warnings.push("no SimpleConstructionScript (components only from a native class?)".into());
        }
        return Ok(asm);
    };
    let (sprops, err) = read(scs);
    if let Some(e) = err {
        asm.warnings.push(format!("SimpleConstructionScript: {e}"));
    }
    let paths = |v: Option<&Value>| -> Vec<String> { list(v).iter().filter_map(|x| obj(Some(x))).collect() };
    let all = paths(get(&sprops, "AllNodes"));
    let roots = paths(get(&sprops, "RootNodes"));
    // SCS nodes
    let mut nodes: Vec<Node> = vec![];
    let mut by_path: HashMap<String, usize> = HashMap::new();
    let mut child_paths: Vec<Vec<String>> = vec![];
    for path in all.iter().chain(roots.iter()) {
        if by_path.contains_key(path) {
            continue;
        }
        let Some(&ni) = idx.get(path) else {
            asm.warnings.push(format!("SCS node {path} not in package"));
            continue;
        };
        let (np, err) = read(ni);
        if let Some(e) = err {
            asm.warnings.push(format!("{}: {e}", pkg.exports[ni].name));
        }
        let name = |n: &str| match get(&np, n) {
            Some(Value::Name(s)) if s != "None" => Some(s.clone()),
            _ => None,
        };
        by_path.insert(path.clone(), nodes.len());
        child_paths.push(paths(get(&np, "ChildNodes")));
        nodes.push(Node {
            name: name("InternalVariableName").unwrap_or_else(|| pkg.exports[ni].name.clone()),
            template: obj(get(&np, "ComponentTemplate")).and_then(|p| idx.get(&p).copied()),
            children: vec![],
            parent_name: name("ParentComponentOrVariableName"),
            socket: name("AttachToName"),
        });
    }
    let mut has_parent = vec![false; nodes.len()];
    for (n, cps) in child_paths.iter().enumerate() {
        for c in cps {
            if let Some(&ci) = by_path.get(c) {
                if ci != n && !has_parent[ci] {
                    has_parent[ci] = true;
                    nodes[n].children.push(ci);
                }
            }
        }
    }
    // walk from the roots (nodes nobody lists as child); (node, parent part, depth)
    let (mut stack, mut unmapped): (Vec<(usize, Option<usize>, u32)>, usize) = (vec![], 0);
    for n in (0..nodes.len()).rev().filter(|&n| !has_parent[n]) {
        let parent = match &nodes[n].parent_name {
            Some(p) => {
                let k = asm.parts.iter().position(|q| &q.name == p);
                if k.is_none() {
                    asm.warnings.push(format!("{} attaches to {p} (native or skipped inherited component): placed at the actor origin", nodes[n].name));
                }
                k
            }
            None => None,
        };
        stack.push((n, parent, 0));
    }
    while let Some((n, parent, d)) = stack.pop() {
        if d > 64 {
            asm.warnings.push("component tree too deep".into());
            break;
        }
        let node = &nodes[n];
        let Some(t) = node.template else {
            asm.warnings.push(format!("{}: no component template", node.name));
            continue;
        };
        let Some(mut part) = component(res, maps, pkg, t, &node.name, None, load) else { continue }; // not a scene component
        if !maps.structs.contains_key(&res.class_name(pkg, t)) && part.mesh.is_none() && node.children.is_empty() && part.class != "ChildActorComponent" {
            unmapped += 1; // ponytail: unknown class, no mesh, no children: most likely not a scene component
            continue;
        }
        if node.socket.is_some() {
            part.socket = node.socket.clone();
        }
        part.parent = parent;
        let me = asm.parts.len();
        for &c in node.children.iter().rev() {
            stack.push((c, Some(me), d + 1));
        }
        let child = if part.class == "ChildActorComponent" { child_actor(res, maps, pkg, t, &mut part, &mut asm.warnings, load, depth) } else { vec![] };
        let prefix = format!("{}/", part.name);
        asm.parts.push(part);
        adopt(&mut asm.parts, child, Some(me), &prefix);
    }
    if unmapped > 0 && !tagged(pkg) {
        asm.warnings.push(format!("{unmapped} leaf component(s) of classes without mappings and without a mesh skipped"));
    }
    Ok(asm)
}

/// Order of `parts` (= asm.parts[base..], parents are absolute indices) with parents first; parts in a
/// parent cycle go last in their original order (the world-transform pass then treats them as roots).
fn topo(parts: &[Part], base: usize) -> Vec<usize> {
    let mut placed = vec![false; parts.len()];
    let mut order = Vec::with_capacity(parts.len());
    while order.len() < parts.len() {
        let before = order.len();
        for i in 0..parts.len() {
            let ready = match parts[i].parent {
                Some(q) if (base..base + parts.len()).contains(&q) => placed[q - base],
                _ => true,
            };
            if !placed[i] && ready {
                placed[i] = true;
                order.push(i);
            }
        }
        if order.len() == before {
            order.extend((0..parts.len()).filter(|&i| !placed[i]));
            break;
        }
    }
    order
}

/// Parts of the child actor class of ChildActorComponent template `t` (relative to the component).
#[allow(clippy::too_many_arguments)]
fn child_actor(res: &Resolver, maps: &Mappings, pkg: &ZenPackage, t: usize, part: &mut Part, warnings: &mut Vec<String>, load: Load, depth: u32) -> Vec<Part> {
    let p = props::read_properties_drift(pkg, t, &part.class, maps, res, drift_for(maps), TAILS);
    let Some(class) = obj(get(&p.props, "ChildActorClass")) else {
        part.warnings.push(format!("child actor class unknown{}", p.error.map(|e| format!(" ({e})")).unwrap_or_default()));
        return vec![];
    };
    let cpkg = package_of(&class).to_string();
    if cpkg.starts_with("/Script/") {
        part.warnings.push(format!("child actor is native class {class}: not shown"));
        return vec![];
    }
    if depth >= MAX_DEPTH {
        part.warnings.push(format!("child actor {cpkg} not followed (depth limit)"));
        return vec![];
    }
    match load_pkg(res, &cpkg, load).map(|(c, _)| assemble_depth(res, maps, &c, load, depth + 1)) {
        Some(Ok(ca)) => {
            part.warnings.push(format!("child actor {} ({} parts)", ca.name, ca.parts.len()));
            warnings.extend(ca.warnings.into_iter().map(|w| format!("child actor {}: {w}", ca.name)));
            ca.parts
        }
        Some(Err(e)) => {
            part.warnings.push(format!("child actor {cpkg}: {e}"));
            vec![]
        }
        None => {
            part.warnings.push(format!("child actor {cpkg}: package not found"));
            vec![]
        }
    }
}

/// One component template: relative transform, mesh, materials, instances. Values absent from the
/// template come from `base` (an inherited component this template overrides) or the class default.
/// None if its class is known and isn't a SceneComponent.
fn component(res: &Resolver, maps: &Mappings, pkg: &ZenPackage, t: usize, name: &str, base: Option<&Part>, load: Load) -> Option<Part> {
    let mut class = res.class_name(pkg, t);
    let mapped = maps.structs.contains_key(&class);
    if mapped && props::build_slot(maps, &class, drift_for(maps), "RelativeLocation").is_none() {
        return None;
    }
    if pkg.exports[t].class_index >> 62 == 2 {
        // a Blueprint component class: name it after its package
        let p = res.object_path(pkg.exports[t].class_index, pkg).unwrap_or_default();
        class = format!("{} (BP class)", package_of(&p).rsplit('/').next().unwrap_or(&p));
    }
    let mut part = match base {
        Some(b) => Part { class: class.clone(), warnings: vec![], ..b.clone() },
        None => Part {
            name: name.into(),
            class: class.clone(),
            mesh: None,
            rel: Xf::IDENTITY,
            rel_known: true,
            parent: None,
            world: Xf::IDENTITY,
            transform_known: true,
            instances: None,
            materials: vec![],
            socket: None,
            warnings: vec![],
        },
    };
    let p = props::read_properties_drift(pkg, t, &res.class_name(pkg, t), maps, res, drift_for(maps), TAILS);
    if let Some(Value::Name(sock)) = get(&p.props, "AttachSocketName") {
        if sock != "None" {
            part.socket = Some(sock.clone()); // level / native components
        }
    }
    let slot = |prop: &str| props::build_slot(maps, &res.class_name(pkg, t), drift_for(maps), prop);
    // a value is: not in the header -> base/class default; zero-masked -> zero; else read or unknown
    let mut missing = vec![];
    let tagged = tagged(pkg);
    let mut value = |prop: &str, default: [f64; 3], names: [&str; 3]| -> [f64; 3] {
        if tagged {
            return floats(get(&p.props, prop), names).unwrap_or(default);
        }
        let s = slot(prop);
        match s.and_then(|s| p.header.iter().find(|h| h.0 == s)) {
            Some(&(_, true)) => [0.0; 3],
            None if s.is_some() => default,
            _ => floats(get(&p.props, prop), names).unwrap_or_else(|| {
                missing.push(prop.to_string());
                default
            }),
        }
    };
    let b = part.rel;
    let loc = value("RelativeLocation", b.trans, ["X", "Y", "Z"]);
    let rot = value("RelativeRotation", Xf::to_rotator(b.rot), ["Pitch", "Yaw", "Roll"]);
    let scale = value("RelativeScale3D", b.scale, ["X", "Y", "Z"]);
    part.rel = Xf { rot: Xf::rotator(rot[0], rot[1], rot[2]), trans: loc, scale };
    if !missing.is_empty() {
        part.rel_known = false;
        part.warnings.push(format!("transform unknown ({} not read: {})", missing.join(", "), p.error.clone().unwrap_or_default()));
    } else if let Some(e) = &p.error {
        part.warnings.push(format!("properties: {e}"));
    }
    if let Some(Value::Array(a)) = get(&p.props, "OverrideMaterials") {
        part.materials = a.iter().map(|m| obj(Some(m)).map(|s| package_of(&s).to_string())).collect();
    }
    if mapped && !is_mesh_class(&class) {
        return Some(part);
    }
    let mesh_prop = ["StaticMesh", "SkeletalMeshAsset", "SkinnedAsset", "SkeletalMesh"];
    let read_mesh = mesh_prop.iter().find_map(|n| obj(get(&p.props, n))).map(|s| package_of(&s).to_string());
    let set = mesh_prop.iter().any(|n| slot(n).is_some_and(|s| p.header.iter().any(|h| h.0 == s && !h.1)));
    let set = set || mesh_prop.iter().all(|n| slot(n).is_none()); // mesh property in a changed range
    part.mesh = read_mesh.or(part.mesh);
    if !tagged && (set || !mapped) && mesh_prop.iter().all(|n| get(&p.props, n).is_none()) {
        // ponytail: mesh property not readable -> first import ref in the template bytes to a package
        // whose main export is a Static/SkeletalMesh; ceiling: picks the wrong one if several are referenced
        let d = pkg.export_data(t);
        let mut found = false;
        for (_, cand) in material::package_refs(pkg, &d[..p.end.unwrap_or(d.len()).min(d.len())]) {
            if let Some((mp, _)) = load_pkg(res, &cand, load) {
                if (0..mp.exports.len()).any(|i| matches!(res.class_name(&mp, i).as_str(), "StaticMesh" | "SkeletalMesh")) {
                    part.warnings.push(format!("mesh found by reference scan: {cand}"));
                    part.mesh = Some(cand);
                    found = true;
                    break;
                }
            }
        }
        if !found && mapped {
            part.warnings.push("mesh property set but unreadable".into());
        }
    }
    // components that don't render: helper volumes (terrain carvers, nav/collision proxies, scanner shapes)
    // ponytail: "Collision" in the class name stands in for C++ class defaults we can't read (e.g. DRG's
    // PathfinderCollisionComponent is hidden by default)
    let flag = |n: &str| match get(&p.props, n) {
        Some(Value::Bool(b)) => Some(*b),
        _ => None,
    };
    if part.mesh.is_some() && (flag("bHiddenInGame") == Some(true) || flag("bVisible") == Some(false) || class.contains("Collision")) {
        part.warnings.push(format!("hidden in game: mesh {} not shown", part.mesh.take().unwrap_or_default().rsplit('/').next().unwrap_or("")));
    }
    if class.contains("InstancedStaticMeshComponent") {
        let d = pkg.export_data(t);
        match instances(d, p.end.unwrap_or(0), tagged) {
            Some(v) => part.instances = Some(v),
            None => {
                part.instances = Some(vec![]);
                part.warnings.push("instance data not found: no instances shown".into());
            }
        }
    }
    Some(part)
}

/// PerInstanceSMData after the properties: TBulkArray of FInstancedStaticMeshInstanceData
/// (i32 element size 128 = FMatrix of doubles, i32 count, matrices; UE4: 64 = FMatrix of floats).
// ponytail: signature scan from the end of the properties (or from 0 if unknown); every matrix must
// be finite with last column (0,0,0,1). Ceiling: an earlier coincidental match would be taken.
fn instances(d: &[u8], from: usize, ue4: bool) -> Option<Vec<[[f64; 4]; 4]>> {
    let u = |o: usize| d.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let es = if ue4 { 64 } else { 128 };
    let f = |c: &[u8], i: usize| if ue4 { f32::from_le_bytes(c[i * 4..][..4].try_into().unwrap()) as f64 } else { f64::from_le_bytes(c[i * 8..][..8].try_into().unwrap()) };
    for o in from..d.len().saturating_sub(8) {
        if u(o)? != es as u32 {
            continue;
        }
        let n = u(o + 4)? as usize;
        let Some(body) = n.checked_mul(es).and_then(|len| d.get(o + 8..o + 8 + len)) else { continue };
        let m: Vec<[[f64; 4]; 4]> = body.chunks_exact(es).map(|c| std::array::from_fn(|r| std::array::from_fn(|k| f(c, r * 4 + k)))).collect();
        let ok = |m: &[[f64; 4]; 4]| m.iter().flatten().all(|v| v.is_finite() && v.abs() < 1e7) && m[0][3] == 0.0 && m[1][3] == 0.0 && m[2][3] == 0.0 && m[3][3] == 1.0;
        if m.iter().all(ok) && (n > 0 || u(o + 8) == Some(4)) {
            return Some(m);
        }
    }
    None
}

// ---------------------------------------------------------------- merged mesh

/// A material slot of the merged mesh: mesh package + its slot index + material (after overrides).
pub struct GSlot {
    pub mesh: String,
    pub slot: usize,
    pub material: String,
    pub diffuse: Option<String>, // texture package
}

/// Merged geometry: all parts' meshes in actor space; sections index `slots`. `groups` = (part index,
/// first triangle) where each part's triangles start.
pub struct Built {
    pub mesh: mesh::Mesh,
    pub slots: Vec<GSlot>,
    pub groups: Vec<(usize, usize)>,
    pub warnings: Vec<String>,
}

struct Loaded {
    mesh: mesh::Mesh,
    slots: Vec<(String, Option<String>)>, // (material, diffuse) per slot, before overrides
    pkg: ZenPackage,
    export: usize,
    end: usize,
}

/// Pseudo mesh package "#bones:<Skeleton package>": the skeleton's bones as geometry (see bone_mesh).
const BONES: &str = "#bones:";

/// Reference skeleton as geometry: a thin double pyramid from each bone to its parent (UE space).
fn bone_mesh(bones: &[skeletal::Bone]) -> mesh::Mesh {
    let mut cs: Vec<Xf> = Vec::with_capacity(bones.len());
    for b in bones {
        let local = Xf { rot: b.rotation.map(|v| v as f64), trans: b.translation.map(|v| v as f64), scale: b.scale.map(|v| v as f64) };
        let parent = (b.parent >= 0).then(|| cs.get(b.parent as usize).copied()).flatten();
        cs.push(parent.map_or(local, |p| local.then(&p)));
    }
    let mut m = mesh::Mesh { positions: vec![], normals: vec![], uvs: vec![], indices: vec![], sections: vec![] };
    for (i, b) in bones.iter().enumerate() {
        if b.parent < 0 {
            continue;
        }
        let (a, c) = (cs[b.parent as usize].trans, cs[i].trans);
        let d = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len < 1e-4 {
            continue;
        }
        let dn = d.map(|x| x / len);
        let helper = if dn[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        let u = norm(cross(dn, helper));
        let v = cross(dn, u);
        let w = (0.1 * len).clamp(0.2, 4.0);
        let mid = std::array::from_fn::<f64, 3, _>(|k| a[k] + 0.2 * d[k]);
        let ring: Vec<[f64; 3]> = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)].iter().map(|(x, y)| std::array::from_fn(|k| mid[k] + w * (x * u[k] + y * v[k]))).collect();
        let base = m.positions.len() as u32;
        for p in [a, c].iter().chain(&ring) {
            m.positions.push(p.map(|x| x as f32));
            let n = norm(std::array::from_fn(|k| p[k] - mid[k]));
            m.normals.push(n.map(|x| x as f32));
            m.uvs.push([0.0, 0.0]);
        }
        for k in 0..4u32 {
            let (r0, r1) = (base + 2 + k, base + 2 + (k + 1) % 4);
            m.indices.extend([base, r1, r0, base + 1, r0, r1]);
        }
    }
    let n = m.indices.len() as u32 / 3;
    m.sections.push((0, 0, n));
    m
}

fn load_mesh(res: &Resolver, package: &str, max_tris: usize, load: Load) -> Res<Loaded> {
    if let Some(skel) = package.strip_prefix(BONES) {
        let (pkg, _) = load_pkg(res, skel, load).ok_or("skeleton package not found")?;
        let export = (0..pkg.exports.len()).find(|&i| res.class_name(&pkg, i) == "Skeleton").ok_or("no Skeleton export")?;
        let bones = skeletal::find_reference_skeleton(pkg.export_data(export), &pkg.names).ok_or("reference skeleton not found")?;
        return Ok(Loaded { mesh: bone_mesh(&bones), slots: vec![], pkg, export, end: usize::MAX });
    }
    let (pkg, path) = load_pkg(res, package, load).ok_or("package not found")?;
    let ubulk = load(&format!("{}.ubulk", path.rsplit_once('.').map_or(path.as_str(), |p| p.0)));
    let export = (0..pkg.exports.len()).find(|&i| matches!(res.class_name(&pkg, i).as_str(), "StaticMesh" | "SkeletalMesh")).ok_or("no StaticMesh/SkeletalMesh export")?;
    let (mesh, end) = if res.class_name(&pkg, export) == "SkeletalMesh" {
        let m = skeletal::decode_skeletal_mesh(&pkg, export, ubulk.as_deref())?;
        (mesh::Mesh { positions: m.positions, normals: m.normals, uvs: m.uvs, indices: m.indices, sections: m.sections }, usize::MAX)
    } else {
        let mut lods = mesh::decode_static_mesh(pkg.export_data(export), ubulk.as_deref())?;
        let k = lods.iter().position(|m| m.tris() <= max_tris).unwrap_or(lods.len() - 1);
        let m = lods.swap_remove(k);
        let end = mesh::find_sections(pkg.export_data(export), &m).map_or(usize::MAX, |(o, _)| o);
        (m, end)
    };
    Ok(Loaded { mesh, slots: vec![], pkg, export, end })
}

/// Builds the merged mesh (most detailed LOD with at most `max_tris` triangles per mesh) and resolves
/// each slot's material/diffuse texture.
pub fn build(res: &Resolver, asm: &Assembly, max_tris: usize, load: Load) -> Built {
    let mut cache: HashMap<String, Result<Loaded, String>> = HashMap::new();
    let mut out = Built { mesh: mesh::Mesh { positions: vec![], normals: vec![], uvs: vec![], indices: vec![], sections: vec![] }, slots: vec![], groups: vec![], warnings: vec![] };
    let mut slot_ids: HashMap<(String, usize, String), usize> = HashMap::new();
    for (pi, part) in asm.parts.iter().enumerate() {
        let Some(mp) = &part.mesh else { continue };
        if !cache.contains_key(mp) {
            let l = load_mesh(res, mp, max_tris, load).map(|mut l| {
                let need = l.mesh.sections.iter().map(|s| s.0 as usize + 1).max().unwrap_or(1);
                l.slots = material::slot_diffuse(res, &l.pkg, l.export, l.end, need, &[], load);
                l
            });
            cache.insert(mp.clone(), l.map_err(|e| e.to_string()));
        }
        let l = match &cache[mp] {
            Ok(l) => l,
            Err(e) => {
                out.warnings.push(format!("{}: {mp}: {e}", part.name));
                continue;
            }
        };
        // global slot per local slot (overrides resolved per distinct material)
        let local: Vec<usize> = (0..l.slots.len().max(l.mesh.sections.iter().map(|s| s.0 as usize + 1).max().unwrap_or(1)))
            .map(|k| {
                let over = part.materials.get(k).cloned().flatten();
                let base = l.slots.get(k).map(|s| s.0.clone()).unwrap_or_default();
                let material = over.clone().unwrap_or(base);
                *slot_ids.entry((mp.clone(), k, material.clone())).or_insert_with(|| {
                    let diffuse = match &over {
                        Some(o) if Some(o) != l.slots.get(k).map(|s| &s.0) => {
                            let mut ov = vec![None; k + 1];
                            ov[k] = Some(o.clone());
                            let need = l.mesh.sections.iter().map(|s| s.0 as usize + 1).max().unwrap_or(1);
                            material::slot_diffuse(res, &l.pkg, l.export, l.end, need, &ov, load).get(k).and_then(|s| s.1.clone())
                        }
                        _ => l.slots.get(k).and_then(|s| s.1.clone()),
                    };
                    out.slots.push(GSlot { mesh: mp.clone(), slot: k, material, diffuse });
                    out.slots.len() - 1
                })
            })
            .collect();
        let copies: Vec<Option<[[f64; 4]; 4]>> = match &part.instances {
            Some(v) => v.iter().map(|m| Some(*m)).collect(),
            None => vec![None],
        };
        out.groups.push((pi, out.mesh.indices.len() / 3));
        for inst in copies {
            append(&mut out.mesh, &l.mesh, &part.world, inst.as_ref(), &local);
        }
    }
    out
}

/// Appends `m` transformed by instance matrix (row vectors) then `xf`; sections remapped by `local`.
fn append(dst: &mut mesh::Mesh, m: &mesh::Mesh, xf: &Xf, inst: Option<&[[f64; 4]; 4]>, local: &[usize]) {
    let base = dst.positions.len() as u32;
    let im = |v: [f64; 3], w: f64| -> [f64; 3] {
        match inst {
            Some(m) => std::array::from_fn(|k| v[0] * m[0][k] + v[1] * m[1][k] + v[2] * m[2][k] + w * m[3][k]),
            None => v,
        }
    };
    let f = |v: [f32; 3]| v.map(|x| x as f64);
    dst.positions.extend(m.positions.iter().map(|&p| xf.point(im(f(p), 1.0)).map(|x| x as f32)));
    // ponytail: instance normals transformed as directions (exact only without non-uniform instance scale)
    dst.normals.extend(m.normals.iter().map(|&n| xf.normal(norm(im(f(n), 0.0))).map(|x| x as f32)));
    dst.uvs.extend(m.uvs.iter().copied());
    dst.uvs.resize(dst.positions.len(), [0.0, 0.0]);
    dst.normals.resize(dst.positions.len(), [0.0, 0.0, 1.0]);
    // mirrored (negative determinant) transforms flip the winding
    let det3 = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    let flip = (xf.scale.iter().product::<f64>() < 0.0) != inst.is_some_and(|m| det3(m[0][..3].try_into().unwrap(), m[1][..3].try_into().unwrap(), m[2][..3].try_into().unwrap()) < 0.0);
    let first = dst.indices.len() as u32;
    for t in m.indices.chunks_exact(3) {
        let t = if flip { [t[0], t[2], t[1]] } else { [t[0], t[1], t[2]] };
        dst.indices.extend(t.map(|i| i + base));
    }
    for &(mat, f0, n) in &m.sections {
        dst.sections.push((local.get(mat as usize).copied().unwrap_or(0) as u32, first + f0, n));
    }
}

// ---------------------------------------------------------------- export

/// Writes `<out_dir>/<BP name>.obj` (one `o` per part, `usemtl` per section), `.mtl` and the diffuse
/// textures as PNG (each once). Returns the written files.
pub fn export_obj(res: &Resolver, asm: &Assembly, load: Load, out_dir: &Path) -> Res<Vec<PathBuf>> {
    use std::fmt::Write;
    let b = build(res, asm, usize::MAX, load);
    if b.mesh.indices.is_empty() {
        return Err(format!("{}: no mesh parts to export", asm.name).into());
    }
    std::fs::create_dir_all(out_dir)?;
    let name = material::sanitize(&asm.name);
    let mut written = vec![];
    let short = |p: &str| material::sanitize(p.rsplit('/').next().unwrap_or(p));
    let mat_name = |k: usize| b.slots.get(k).map_or(format!("slot{k}"), |s| format!("{}_{}_{}", short(&s.mesh), s.slot, short(&s.material)));
    let mut mtl = String::new();
    let mut pngs: HashMap<String, Option<String>> = HashMap::new();
    for (k, s) in b.slots.iter().enumerate() {
        let _ = writeln!(mtl, "newmtl {}\nKd 1 1 1\nKs 0 0 0", mat_name(k));
        let Some(t) = &s.diffuse else { continue };
        if !pngs.contains_key(t) {
            let f = material::texture_to_png(res, t, load).and_then(|png| {
                let f = format!("{}.png", short(t));
                std::fs::write(out_dir.join(&f), png).ok()?;
                written.push(out_dir.join(&f));
                Some(f)
            });
            pngs.insert(t.clone(), f);
        }
        if let Some(Some(f)) = pngs.get(t) {
            let _ = writeln!(mtl, "map_Kd {f}");
        }
    }
    // Y-up meters like Mesh::to_obj (swapping Y/Z also turns UE's winding into OBJ's)
    let m = &b.mesh;
    let mut obj = format!("# {} assembled by ucas-explorer\nmtllib {name}.mtl\n", asm.name);
    for p in &m.positions {
        let _ = writeln!(obj, "v {} {} {}", p[0] / 100.0, p[2] / 100.0, p[1] / 100.0);
    }
    for t in &m.uvs {
        let _ = writeln!(obj, "vt {} {}", t[0], 1.0 - t[1]);
    }
    for n in &m.normals {
        let _ = writeln!(obj, "vn {} {} {}", n[0], n[2], n[1]);
    }
    let mut slot = vec![0u32; m.tris()];
    for &(s, first, n) in &m.sections {
        for x in slot.iter_mut().skip(first as usize / 3).take(n as usize) {
            *x = s;
        }
    }
    let (mut g, mut cur) = (b.groups.iter().peekable(), u32::MAX);
    for (k, t) in m.indices.chunks_exact(3).enumerate() {
        while let Some(&&(pi, _)) = g.peek().filter(|x| x.1 <= k) {
            g.next();
            let _ = writeln!(obj, "o {}", material::sanitize(&asm.parts[pi].name));
            cur = u32::MAX;
        }
        if slot[k] != cur {
            cur = slot[k];
            let _ = writeln!(obj, "usemtl {}", mat_name(cur as usize));
        }
        let [a, b2, c] = [t[0] + 1, t[1] + 1, t[2] + 1];
        let _ = writeln!(obj, "f {a}/{a}/{a} {b2}/{b2}/{b2} {c}/{c}/{c}");
    }
    for (file, data) in [(format!("{name}.obj"), obj), (format!("{name}.mtl"), mtl)] {
        std::fs::write(out_dir.join(&file), data)?;
        written.push(out_dir.join(file));
    }
    Ok(written)
}

/// One-line-per-part summary (CLI / logs).
pub fn summary(asm: &Assembly) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for p in &asm.parts {
        let t = p.world.trans;
        let _ = writeln!(
            s,
            "  {:<40} {:<36} {} @({:.1},{:.1},{:.1}) s({:.2},{:.2},{:.2}){}{}",
            p.name,
            p.class,
            p.mesh.as_deref().map_or("-", |m| m.rsplit('/').next().unwrap_or(m)),
            t[0], t[1], t[2], p.world.scale[0], p.world.scale[1], p.world.scale[2],
            if p.transform_known { "" } else { " TRANSFORM UNKNOWN" },
            p.instances.as_ref().map(|v| format!(" x{} instances", v.len())).unwrap_or_default()
        );
        for w in &p.warnings {
            let _ = writeln!(s, "      ! {w}");
        }
    }
    for w in &asm.warnings {
        let _ = writeln!(s, "  ! {w}");
    }
    s
}

#[cfg(test)]
mod layout_tests {
    #[test]
    fn row_offsets_place_meshes_side_by_side() {
        // two 100-wide boxes centered on the origin, after a group ending at y=50: no overlaps, 15 gap
        let b = Some(([-50.0, -50.0, 0.0], [50.0, 50.0, 10.0]));
        let o = super::row_offsets(&[b, None, b], Some(50.0));
        assert_eq!(o, vec![Some(50.0 + 15.0 + 50.0), None, Some(165.0 + 15.0 + 50.0)]);
        // nothing at the origin: the first mesh starts at y=0 (its min edge)
        assert_eq!(super::row_offsets(&[b], None), vec![Some(50.0)]);
    }
}

#[cfg(test)]
mod tests {
    use super::Xf;
    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() < 1e-9)
    }

    #[test]
    fn level_children_finds_world_partition_and_reads_level_transform() {
        use crate::props::Value;
        let mut res = crate::global::Resolver::default();
        for (k, p) in ["G/Content/Maps/Lobby.umap", "G/Content/Maps/Lobby/_Generated_/Cell1.umap", "G/Content/__ExternalActors__/Maps/Lobby/A/B.uasset", "G/Content/Maps/LobbyOther.umap", "G/Content/__ExternalActors__/Maps/LobbyOther/C.uasset"].iter().enumerate() {
            res.packages.insert(k as u64, p.to_string());
        }
        let pkg = crate::zen::ZenPackage { name: "/Game/Maps/Lobby".into(), flags: 0, names: vec![], imported_packages: vec![], imported_package_names: vec![], imported_public_export_hashes: vec![], imports: vec![], exports: vec![], data: vec![], import_paths: vec![] };
        let got: Vec<String> = super::level_children(&res, &Default::default(), &pkg, &Xf::IDENTITY, &mut vec![]).into_iter().map(|c| c.0).collect();
        assert_eq!(got, ["/Game/Maps/Lobby/_Generated_/Cell1", "/Game/__ExternalActors__/Maps/Lobby/A/B"]);

        let v = |f: &[(&str, f64)]| Value::Struct("".into(), f.iter().map(|(n, x)| (n.to_string(), Value::Float(*x))).collect());
        let t = Value::Struct("Transform".into(), vec![
            ("Rotation".into(), v(&[("X", 0.0), ("Y", 0.0), ("Z", 0.0), ("W", 1.0)])),
            ("Translation".into(), v(&[("X", 100.0), ("Y", 0.0), ("Z", 5.0)])),
        ]);
        let xf = super::level_xf(Some(&t)).unwrap();
        assert!(close(xf.point([1.0, 2.0, 3.0]), [101.0, 2.0, 8.0]));
        assert!(super::level_xf(None).is_none());
    }

    #[test]
    fn transforms_compose_like_ue() {
        // as FRotationMatrix: yaw turns X toward Y, pitch turns X up toward Z, roll turns Y toward -Z
        assert!(close(Xf::rotate(Xf::rotator(0.0, 90.0, 0.0), [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        assert!(close(Xf::rotate(Xf::rotator(90.0, 0.0, 0.0), [1.0, 0.0, 0.0]), [0.0, 0.0, 1.0]));
        assert!(close(Xf::rotate(Xf::rotator(0.0, 0.0, 90.0), [0.0, 1.0, 0.0]), [0.0, 0.0, -1.0]));
        let parent = Xf { rot: Xf::rotator(0.0, 90.0, 0.0), trans: [100.0, 0.0, 0.0], scale: [2.0; 3] };
        let child = Xf { rot: Xf::rotator(0.0, 90.0, 0.0), trans: [10.0, 0.0, 5.0], scale: [1.0, 1.0, 3.0] };
        let w = child.then(&parent);
        assert!(close(w.trans, [100.0, 20.0, 10.0]));
        assert!(close(w.scale, [2.0, 2.0, 6.0]));
        // world point = parent(child(p)); yaw 90+90 = 180
        assert!(close(w.point([1.0, 0.0, 0.0]), parent.point(child.point([1.0, 0.0, 0.0]))));
        assert!(close(Xf::rotate(w.rot, [1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0]));
        assert_eq!(Xf::IDENTITY.then(&Xf::IDENTITY), Xf::IDENTITY);
        assert!(close(Xf::to_rotator(Xf::rotator(30.0, -120.0, 45.0)), [30.0, -120.0, 45.0]));
    }
}

#[cfg(test)]
mod real {
    use std::collections::HashMap;
    use std::path::Path;
    static UTOC: std::sync::LazyLock<String> = std::sync::LazyLock::new(crate::test_utoc);

    /// Per component class in BP templates: count, guessed build scene base (most frequent slot that is
    /// followed by RelativeScale3D two slots later) vs the drift-corrected one.
    /// cargo test --release gen_classes -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gen_classes() {
        let toc = crate::open_archive(Path::new(UTOC.as_str())).unwrap();
        let res = crate::global::Resolver::load(&Path::new(UTOC.as_str()).with_file_name("global.utoc"), &toc).unwrap();
        let maps = crate::props::Mappings::load(&crate::props::Mappings::find(Path::new(UTOC.as_str()), Some("Palia")).unwrap()).unwrap();
        let mut ucas = std::fs::File::open(crate::data_path(Path::new(UTOC.as_str()))).unwrap();
        let mut stats: HashMap<String, (usize, HashMap<usize, usize>)> = HashMap::new();
        for (_, e) in toc.files.iter().filter(|(p, _)| p.contains("/BP_") && p.ends_with(".uasset")).step_by(3) {
            let Ok(pkg) = crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).and_then(crate::zen::parse) else { continue };
            for i in 0..pkg.exports.len() {
                if !pkg.exports[i].name.ends_with("_GEN_VARIABLE") {
                    continue;
                }
                let p = crate::props::read_properties_drift(&pkg, i, "Object", &maps, &res, &[], &[]);
                let s = stats.entry(res.class_name(&pkg, i)).or_default();
                s.0 += 1;
                let set: Vec<usize> = p.header.iter().filter(|h| !h.1).map(|h| h.0).collect();
                for &k in &set {
                    if k >= 5 && set.contains(&(k + 2)) {
                        *s.1.entry(k - 5).or_default() += 1;
                    }
                }
            }
        }
        let mut v: Vec<_> = stats.into_iter().collect();
        v.sort_by_key(|x| std::cmp::Reverse(x.1 .0));
        for (c, (n, b)) in v.iter().take(60) {
            let mut b: Vec<_> = b.iter().collect();
            b.sort_by_key(|x| std::cmp::Reverse(*x.1));
            let ours = crate::props::build_slot(&maps, c, super::DRIFT, "RelativeLocation").map(|s| s - 5);
            let mapped = crate::props::build_slot(&maps, c, &[], "RelativeLocation").map(|s| s - 5);
            println!("{c:<45} {n:>6}  guess {:?}  drift {ours:?}  mappings {mapped:?}", &b[..b.len().min(3)]);
        }
    }

    /// Assembly stats over every STEP-th BP (default 25), nothing written:
    /// STEP=25 cargo test --release bp_batch -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bp_batch() {
        let toc = crate::open_archive(Path::new(UTOC.as_str())).unwrap();
        let res = crate::global::Resolver::load(&Path::new(UTOC.as_str()).with_file_name("global.utoc"), &toc).unwrap();
        let maps = crate::props::Mappings::load(&crate::props::Mappings::find(Path::new(UTOC.as_str()), Some("Palia")).unwrap()).unwrap();
        let mut ucas = std::fs::File::open(crate::data_path(Path::new(UTOC.as_str()))).unwrap();
        let step: usize = std::env::var("STEP").map(|s| s.parse().unwrap()).unwrap_or(25);
        let bps: Vec<_> = toc.files.iter().filter(|(p, _)| p.contains("/BP_") && p.ends_with(".uasset")).step_by(step).cloned().collect();
        let (mut n, mut fail, mut with_mesh, mut mesh_parts, mut known, mut ism, mut inherited, mut child) = (0, 0, 0, 0, 0, 0, 0, 0);
        let mut warns: HashMap<String, usize> = HashMap::new();
        let mut unknown_cls: HashMap<String, usize> = HashMap::new();
        for (p, e) in &bps {
            let Ok(pkg) = crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).and_then(crate::zen::parse) else { continue };
            let mut load = |q: &str| -> Option<Vec<u8>> {
                let (_, e) = toc.files.iter().find(|(f, _)| f == q)?;
                crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).ok()
            };
            let asm = match super::assemble(&res, &maps, &pkg, &mut load) {
                Ok(a) => a,
                Err(err) => {
                    fail += 1;
                    *warns.entry(format!("FAIL {err}")).or_default() += 1;
                    continue;
                }
            };
            n += 1;
            let meshes: Vec<_> = asm.parts.iter().filter(|p| p.mesh.is_some()).collect();
            with_mesh += !meshes.is_empty() as usize;
            if std::env::var("LIST").is_ok() && meshes.len() >= 4 {
                println!("LIST {} {p}", meshes.len());
            }
            mesh_parts += meshes.len();
            known += meshes.iter().filter(|p| p.transform_known).count();
            ism += meshes.iter().filter(|p| p.instances.is_some()).count();
            child += asm.parts.iter().filter(|p| p.class == "ChildActorComponent").count();
            inherited += asm.parent.is_some() as usize;
            for part in &asm.parts {
                if !part.transform_known && part.warnings.iter().any(|w| w.starts_with("transform unknown")) {
                    *unknown_cls.entry(part.class.clone()).or_default() += 1;
                }
            }
            let kinds = asm.warnings.iter().chain(asm.parts.iter().flat_map(|p| p.warnings.iter()));
            for w in kinds {
                let k: String = w.chars().map(|c| if c.is_ascii_digit() { '#' } else { c }).collect();
                let k = k.trim_start_matches("parent ").trim_start_matches("child actor ");
                let k = if k.starts_with("BP_") { k.split_once(": ").map_or(k, |x| x.1) } else { k };
                let k = k.split(" at 0x").next().unwrap_or(k);
                let k: String = k.split(';').take(2).collect::<Vec<_>>().join(";").split(", the").next().unwrap_or(k).to_string();
                let k = if k.starts_with("properties: .") || k.starts_with("properties: slot") { k.split(": ").take(3).collect::<Vec<_>>().join(": ") } else { k };
                *warns.entry(k).or_default() += 1;
            }
            if std::env::var("SHOW").is_ok_and(|s| p.contains(&s) || asm.parts.iter().any(|q| q.warnings.iter().any(|w| w.contains(&s)))) {
                println!("{p}\n{}", super::summary(&asm));
            }
        }
        println!("{n} BPs assembled (every {step}th of {}), {fail} failed", toc.files.iter().filter(|(p, _)| p.contains("/BP_") && p.ends_with(".uasset")).count());
        println!("with >=1 mesh part: {with_mesh} ({:.1}%)", 100.0 * with_mesh as f64 / n.max(1) as f64);
        println!("mesh parts: {mesh_parts}, transform known {known} ({:.1}%)", 100.0 * known as f64 / mesh_parts.max(1) as f64);
        println!("ISM parts {ism}, child actor components {child}, BPs with a BP parent {inherited}");
        let mut u: Vec<_> = unknown_cls.into_iter().collect();
        u.sort_by_key(|x| std::cmp::Reverse(x.1));
        println!("transform unknown by class: {:?}", &u[..u.len().min(15)]);
        let mut w: Vec<_> = warns.into_iter().collect();
        w.sort_by_key(|x| std::cmp::Reverse(x.1));
        for (k, c) in w.iter().take(25) {
            println!("  {c:>5}x {k}");
        }
    }
}
