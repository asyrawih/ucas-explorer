//! egui browser: folder tree | file list + search | info, hex preview, extract.
use crate::global::Resolver;
use crate::zen::{self, ZenPackage};
use crate::{anim, bp, data_path, material, mesh, preview, props, skeletal, extract_one, index_path, open_archive, read_chunk, texture, Res, Toc};
use eframe::egui;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

const PREVIEW_BYTES: u64 = 64 * 1024;

struct Dir {
    name: String,
    parent: usize,
    dirs: Vec<usize>,
    files: Vec<usize>, // indices into toc.files
}

struct Loaded {
    utoc: PathBuf,
    toc: Arc<Toc>,
    dirs: Vec<Dir>,
    lower: Vec<String>, // lowercase paths for search
    ucas: Option<File>,
    resolver: Arc<Resolver>, // shared with background texture loaders
    maps: Result<props::Mappings, String>, // .usmap auto-detected at open
}

/// Parsed .uasset/.umap (loaded on a background thread).
struct Asset {
    pkg: ZenPackage,
    classes: Vec<String>,
    ubulk: Option<Vec<u8>>,
}

#[derive(PartialEq, Clone, Copy, Default)]
enum Tab {
    #[default]
    Hex,
    Texture,
    Mesh,
    Anim,
    Props,
    Material,
    Blueprint,
}

type AssetMsg = Result<(ZenPackage, Option<Vec<u8>>), String>;

struct Job {
    total: usize,
    ok: Arc<AtomicUsize>,
    failed: Arc<AtomicUsize>,
    done: Arc<AtomicBool>,
}

#[derive(Default)]
struct App {
    loaded: Option<Loaded>,
    path_input: String,
    out_dir: String,
    search: String,
    results: Vec<usize>,
    cur_dir: usize,
    selected: Option<usize>,
    preview: Vec<u8>,
    status: String,
    job: Option<Job>,
    asset_rx: Option<Receiver<AssetMsg>>,
    asset: Option<Result<Asset, String>>,
    export: Option<usize>,
    tab: Tab,
    tex: Option<Result<(texture::Image, Option<egui::TextureHandle>), String>>,
    mesh: Option<Result<Vec<mesh::Mesh>, String>>,
    mesh_rot: crate::viewport::Camera,
    mesh_tex: Option<Vec<preview::Tex>>, // per material slot; None = not loaded yet
    mesh_tex_rx: Option<Receiver<preview::SlotImages>>,
    /// material tab: (parameter, texture name, role, uploaded texture), diffuse first
    mat_tex: Option<Vec<(String, String, &'static str, Option<egui::TextureHandle>, preview::Tex)>>,
    mat_rx: Option<Receiver<preview::MaterialImages>>,
    anim: Option<Result<anim::Anim, String>>,
    props: Option<(props::Report, Vec<String>)>, // report + its JSON lines
    /// Blueprint tab: assembly, then merged mesh + warnings from the background builder
    bp: Option<Result<bp::Assembly, String>>,
    bp_mesh: Option<(mesh::Mesh, Vec<String>)>,
    bp_rx: Option<Receiver<preview::BpBuilt>>,
    /// focus mode (Ctrl+Space): the preview panel fills the window, tree and file list hidden
    focus: bool,
}

fn hex_view(ui: &mut egui::Ui, id: &str, data: &[u8]) {
    let row_h = ui.text_style_height(&egui::TextStyle::Monospace);
    egui::ScrollArea::vertical().id_salt(id).auto_shrink(false).show_rows(ui, row_h, data.len().div_ceil(16), |ui, range| {
        for r in range {
            let chunk = &data[r * 16..(r * 16 + 16).min(data.len())];
            let hex: String = chunk.iter().map(|b| format!("{b:02x} ")).collect();
            let asc: String = chunk.iter().map(|&b| if b.is_ascii_graphic() { b as char } else { '.' }).collect();
            ui.monospace(format!("{:08x}  {hex:<48} {asc}", r * 16));
        }
    });
}

const MAX_PREVIEW_TRIS: usize = 150_000;

fn is_material(class: &str) -> bool {
    class == "Material" || class.starts_with("MaterialInstance")
}

/// Exports material package `package` (textures + .mtl + parameter list) into the asset's folder.
fn save_material(l: &Loaded, package: &str, path: &str, out_dir: &str) -> Res<PathBuf> {
    let mut ucas = File::open(data_path(&l.utoc))?;
    let toc = &l.toc;
    let mut load = |p: &str| -> Option<Vec<u8>> {
        let (_, e) = toc.files.iter().find(|(q, _)| q == p)?;
        read_chunk(toc, &mut ucas, *e, u64::MAX).ok()
    };
    let dir = out_path(out_dir, path, "")?.parent().ok_or("no parent")?.to_path_buf();
    let files = material::export_material(&l.resolver, package, &mut load, &dir)?;
    files.into_iter().find(|f| f.extension().is_some_and(|e| e == "mtl")).ok_or_else(|| "no mtl written".into())
}

fn is_mesh(class: &str) -> bool {
    class == "StaticMesh" || class == "SkeletalMesh"
}

/// Decodes a Static/SkeletalMesh export into LODs for preview (skeletal: one LOD, no bones).
fn decode_for_preview(a: &Asset, i: usize) -> Res<Vec<mesh::Mesh>> {
    if a.classes[i] != "SkeletalMesh" {
        return mesh::decode_static_mesh(a.pkg.export_data(i), a.ubulk.as_deref());
    }
    let m = skeletal::decode_skeletal_mesh(&a.pkg, i, a.ubulk.as_deref())?;
    Ok(vec![mesh::Mesh { positions: m.positions, normals: m.normals, uvs: m.uvs, indices: m.indices, sections: m.sections }])
}

/// StaticMesh -> OBJ+MTL+PNG in the asset's folder; SkeletalMesh -> textured .glb. Returns the main file.
fn save_mesh(l: &Loaded, a: &Asset, i: usize, path: &str, out_dir: &str) -> Res<PathBuf> {
    let mut ucas = File::open(data_path(&l.utoc))?;
    let toc = &l.toc;
    let mut load = |p: &str| -> Option<Vec<u8>> {
        let (_, e) = toc.files.iter().find(|(q, _)| q == p)?;
        read_chunk(toc, &mut ucas, *e, u64::MAX).ok()
    };
    if a.classes[i] == "SkeletalMesh" {
        let m = skeletal::decode_skeletal_mesh(&a.pkg, i, a.ubulk.as_deref())?;
        let need = m.sections.iter().map(|s| s.0 as usize + 1).max().unwrap_or(1);
        let tex = material::slot_diffuse_pngs(&l.resolver, &a.pkg, i, need, &mut load);
        let dst = out_path(out_dir, path, ".glb")?;
        write_file(&dst, &skeletal::to_glb(&m, &a.pkg.exports[i].name, &tex)?)?;
        return Ok(dst);
    }
    let dir = out_path(out_dir, path, "")?.parent().ok_or("no parent")?.to_path_buf();
    let files = material::export_obj_with_materials(&l.resolver, &a.pkg, i, a.ubulk.as_deref(), &mut load, &dir)?;
    files.into_iter().find(|f| f.extension().is_some_and(|e| e == "obj")).ok_or_else(|| "no obj written".into())
}

/// Blueprint assembly -> OBJ+MTL+PNG in the asset's folder. Returns the .obj.
fn save_bp(l: &Loaded, asm: &bp::Assembly, path: &str, out_dir: &str) -> Res<PathBuf> {
    let mut ucas = File::open(data_path(&l.utoc))?;
    let toc = &l.toc;
    let mut load = |p: &str| -> Option<Vec<u8>> {
        let (_, e) = toc.files.iter().find(|(q, _)| q == p)?;
        read_chunk(toc, &mut ucas, *e, u64::MAX).ok()
    };
    let dir = out_path(out_dir, path, "")?.parent().ok_or("no parent")?.to_path_buf();
    let files = bp::export_obj(&l.resolver, asm, &mut load, &dir)?;
    files.into_iter().find(|f| f.extension().is_some_and(|e| e == "obj")).ok_or_else(|| "no obj written".into())
}

/// AnimSequence -> .glb (skeleton + auto-picked skinned mesh + animation). Returns (file, mesh used).
fn save_anim(l: &Loaded, a: &Asset, i: usize, path: &str, out_dir: &str) -> Res<(PathBuf, Option<String>)> {
    let mut ucas = File::open(data_path(&l.utoc))?;
    let toc = &l.toc;
    let mut load = |p: &str| -> Option<Vec<u8>> {
        let (_, e) = toc.files.iter().find(|(q, _)| q == p)?;
        read_chunk(toc, &mut ucas, *e, u64::MAX).ok()
    };
    let (glb, mesh) = anim::anim_to_glb(&l.resolver, &a.pkg, i, None, &mut load)?;
    let dst = out_path(out_dir, path, ".glb")?;
    write_file(&dst, &glb)?;
    Ok((dst, mesh))
}

/// Uploads a decoded image for display; None if it is too large for the GPU (or malformed).
fn upload_image(ctx: &egui::Context, name: &str, img: &texture::Image) -> Option<egui::TextureHandle> {
    let max = ctx.input(|i| i.max_texture_side);
    let n = img.width as usize * img.height as usize * 4;
    (img.width as usize <= max && img.height as usize <= max && img.rgba.len() == n).then(|| {
        let ci = egui::ColorImage::from_rgba_unmultiplied([img.width as usize, img.height as usize], &img.rgba);
        ctx.load_texture(name, ci, Default::default())
    })
}

/// Draws an uploaded image scaled to fit the remaining space, or a note if it could not be uploaded.
fn fit_image(ui: &mut egui::Ui, img: &texture::Image, handle: &Option<egui::TextureHandle>) {
    match handle {
        Some(t) => {
            let avail = ui.available_size() - egui::vec2(4.0, 4.0); // slack, else the resizable panel creeps wider
            let s = (avail.x / img.width as f32).min(avail.y / img.height as f32);
            ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(img.width as f32, img.height as f32) * s));
        }
        None => {
            ui.weak("too large (or malformed) to display; export still works");
        }
    }
}

fn is_texture(name: &str, class: &str) -> bool {
    class.contains("Texture") || (class == "?" && name.starts_with("T_"))
}

/// out_dir/<asset path without ext><suffix>, refusing archive paths that escape out_dir.
fn out_path(out_dir: &str, asset: &str, suffix: &str) -> Res<PathBuf> {
    if asset.split('/').any(|s| s == ".." || s.is_empty()) {
        return Err("unsafe path".into());
    }
    let stem = asset.rsplit_once('.').map_or(asset, |(a, _)| a);
    Ok(Path::new(out_dir).join(format!("{stem}{suffix}")))
}

fn write_file(path: &Path, data: &[u8]) -> Res<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(path, data)?;
    Ok(())
}

/// .uexp / .ubulk / .uptnl next to a .uasset/.umap: part of that package, hidden from the tree and
/// search, extracted along with it.
fn is_companion(path: &str) -> bool {
    path.rsplit_once('.').is_some_and(|(_, e)| matches!(e, "uexp" | "ubulk" | "uptnl"))
}

/// `files` plus the companions of every package among them (toc.files is sorted by path).
fn with_companions(toc: &Toc, files: Vec<usize>) -> Vec<usize> {
    let mut out = files.clone();
    for f in files {
        let path = &toc.files[f].0;
        let Some((stem, "uasset" | "umap")) = path.rsplit_once('.') else { continue };
        for ext in ["uexp", "ubulk", "uptnl"] {
            let c = format!("{stem}.{ext}");
            if let Ok(k) = toc.files.binary_search_by(|(p, _)| p.as_str().cmp(&c)) {
                out.push(k);
            }
        }
    }
    out
}

fn build_tree(toc: &Toc) -> Vec<Dir> {
    let mut dirs = vec![Dir { name: "/".into(), parent: 0, dirs: vec![], files: vec![] }];
    let mut lookup: HashMap<(usize, &str), usize> = HashMap::new();
    for (i, (path, _)) in toc.files.iter().enumerate().filter(|(_, (p, _))| !is_companion(p)) {
        let mut cur = 0;
        let mut parts = path.split('/').peekable();
        while let Some(part) = parts.next() {
            if parts.peek().is_none() {
                dirs[cur].files.push(i);
            } else {
                cur = *lookup.entry((cur, part)).or_insert_with(|| {
                    dirs.push(Dir { name: part.into(), parent: cur, dirs: vec![], files: vec![] });
                    let id = dirs.len() - 1;
                    dirs[cur].dirs.push(id);
                    id
                });
            }
        }
    }
    for d in 0..dirs.len() {
        let mut v = std::mem::take(&mut dirs[d].dirs);
        v.sort_by(|&a, &b| dirs[a].name.cmp(&dirs[b].name));
        dirs[d].dirs = v;
    }
    dirs
}

fn human(n: u64) -> String {
    let mut f = n as f64;
    for u in ["B", "KB", "MB", "GB"] {
        if f < 1024.0 || u == "GB" {
            return if u == "B" { format!("{n} B") } else { format!("{f:.1} {u}") };
        }
        f /= 1024.0;
    }
    unreachable!()
}

impl App {
    fn open(&mut self, path: PathBuf) {
        let path = index_path(&path);
        match (|| -> Res<Loaded> {
            let toc = open_archive(&path)?;
            if toc.files.is_empty() {
                return Err("container has no directory index (e.g. global.utoc)".into());
            }
            let dirs = build_tree(&toc);
            let lower = toc.files.iter().map(|(p, _)| p.to_lowercase()).collect();
            let ucas = File::open(data_path(&path)).ok();
            let maps = match props::Mappings::find(&path, props::game_name(&toc.files).as_deref()) {
                Some(p) => props::Mappings::load(&p).map_err(|e| format!("{}: {e}", p.display())),
                None if toc.pak.is_some() => Ok(props::Mappings::default()), // legacy .pak: tagged properties
                None => Err("drop a .usmap on the window; it is remembered for this game".to_string()),
            };
            Ok(Loaded { utoc: path.clone(), toc: Arc::new(toc), dirs, lower, ucas, resolver: Arc::default(), maps })
        })() {
            Ok(mut l) => {
                let res_note = match Resolver::load(&path.with_file_name("global.utoc"), &l.toc) {
                    _ if l.toc.unity.is_some() => format!("  ·  Unity {}", l.toc.unity.as_ref().unwrap().unity),
                    Ok(r) => {
                        l.resolver = Arc::new(r);
                        String::new()
                    }
                    Err(e) => format!("  ·  resolver: {e}"),
                };
                let out_dir = match std::mem::take(&mut self.out_dir) {
                    s if s.is_empty() => path.with_file_name("Exports").display().to_string(),
                    s => s,
                };
                *self = App {
                    status: format!("{} files{res_note}", l.toc.files.len()),
                    path_input: path.display().to_string(),
                    out_dir,
                    loaded: Some(l),
                    ..Default::default()
                };
            }
            Err(e) => self.status = format!("open failed: {e}"),
        }
    }

    /// Loads a dropped .usmap for the open archive and saves it to the per-game library.
    fn drop_usmap(&mut self, p: &Path) {
        let Some(l) = &mut self.loaded else {
            self.status = "open an archive first, then drop its .usmap".into();
            return;
        };
        let game = props::game_name(&l.toc.files);
        let lib = match &game {
            Some(g) => props::Mappings::remember(p, g),
            None => Ok(p.to_path_buf()),
        };
        self.status = match lib.and_then(|lp| props::Mappings::load(&lp).map(|m| (lp, m))) {
            Ok((lp, m)) => {
                l.maps = Ok(m);
                self.props = None;
                match game {
                    Some(g) => format!("mappings loaded and remembered for {g}: {}", lp.display()),
                    None => format!("mappings loaded: {}", lp.display()),
                }
            }
            Err(e) => format!("usmap failed: {e}"),
        };
    }

    fn select_export(&mut self, i: usize) {
        self.export = Some(i);
        (self.tex, self.mesh, self.mesh_rot, self.anim, self.props) = (None, None, Default::default(), None, None);
        let has_maps = self.loaded.as_ref().is_some_and(|l| l.maps.is_ok());
        (self.mesh_tex, self.mesh_tex_rx, self.mat_tex, self.mat_rx) = (None, None, None, None);
        (self.bp, self.bp_mesh, self.bp_rx) = (None, None, None);
        if let Some(Ok(a)) = &self.asset {
            if bp::is_3d_class(&a.classes[i]) || (i == bp::main_export(&a.pkg) && bp::references_meshes(&a.pkg) && !is_material(&a.classes[i])) {
                self.tab = Tab::Blueprint;
            } else if is_texture(&a.pkg.exports[i].name, &a.classes[i]) {
                self.tab = Tab::Texture;
            } else if is_material(&a.classes[i]) {
                self.tab = Tab::Material;
            } else if is_mesh(&a.classes[i]) {
                self.tab = Tab::Mesh;
            } else if a.classes[i] == "AnimSequence" {
                self.tab = Tab::Anim;
            } else {
                self.tab = if has_maps { Tab::Props } else { Tab::Hex };
            }
        }
    }

    fn select(&mut self, f: usize) {
        let Some(l) = &mut self.loaded else { return };
        self.selected = Some(f);
        self.preview.clear();
        (self.asset, self.asset_rx, self.export, self.tex, self.mesh, self.anim, self.props) = (None, None, None, None, None, None, None);
        (self.mesh_tex, self.mesh_tex_rx, self.bp, self.bp_mesh, self.bp_rx) = (None, None, None, None, None);
        let path = &l.toc.files[f].0;
        if path.ends_with(".uasset") || path.ends_with(".umap") {
            let (tx, rx) = std::sync::mpsc::channel();
            let toc = l.toc.clone();
            let ucas_path = data_path(&l.utoc);
            let entry = l.toc.files[f].1;
            let ubulk_path = format!("{}.ubulk", path.rsplit_once('.').unwrap().0);
            let ubulk_entry = l.toc.files.iter().find(|(p, _)| *p == ubulk_path).map(|e| e.1);
            std::thread::spawn(move || {
                let r = (|| -> Res<(ZenPackage, Option<Vec<u8>>)> {
                    let mut ucas = File::open(ucas_path)?;
                    let pkg = zen::parse(read_chunk(&toc, &mut ucas, entry, u64::MAX)?)?;
                    let ubulk = ubulk_entry.and_then(|e| read_chunk(&toc, &mut ucas, e, u64::MAX).ok());
                    Ok((pkg, ubulk))
                })();
                let _ = tx.send(r.map_err(|e| e.to_string()));
            });
            self.asset_rx = Some(rx);
        }
        if l.toc.unity.as_ref().is_some_and(|u| matches!(u.class(l.toc.files[f].1), 28 | 43 | 83)) {
            return; // previewed decoded in the info panel, not as hex
        }
        let Some(ucas) = &mut l.ucas else {
            self.status = "missing data file (.ucas next to .utoc)".into();
            return;
        };
        match read_chunk(&l.toc, ucas, l.toc.files[f].1, PREVIEW_BYTES) {
            Ok(d) => self.preview = d,
            Err(e) => self.status = format!("read failed: {e}"),
        }
    }

    fn extract(&mut self, files: Vec<usize>) {
        let Some(l) = &self.loaded else { return };
        if self.job.as_ref().is_some_and(|j| !j.done.load(Relaxed)) {
            return;
        }
        let files = with_companions(&l.toc, files);
        let job = Job {
            total: files.len(),
            ok: Default::default(),
            failed: Default::default(),
            done: Default::default(),
        };
        let (toc, ok, failed, done) = (l.toc.clone(), job.ok.clone(), job.failed.clone(), job.done.clone());
        let ucas_path = data_path(&l.utoc);
        let out = PathBuf::from(&self.out_dir);
        std::thread::spawn(move || {
            if let Ok(mut ucas) = File::open(ucas_path) {
                for f in files {
                    let (p, e) = &toc.files[f];
                    match extract_one(&toc, &mut ucas, &out, p, *e) {
                        Ok(()) => ok.fetch_add(1, Relaxed),
                        Err(_) => failed.fetch_add(1, Relaxed),
                    };
                }
            }
            done.store(true, Relaxed);
        });
        self.job = Some(job);
    }

    fn files_under(&self, d: usize) -> Vec<usize> {
        let Some(l) = &self.loaded else { return vec![] };
        let (mut out, mut stack) = (vec![], vec![d]);
        while let Some(d) = stack.pop() {
            out.extend(&l.dirs[d].files);
            stack.extend(&l.dirs[d].dirs);
        }
        out
    }

    fn dir_path(&self, mut d: usize) -> String {
        let Some(l) = &self.loaded else { return String::new() };
        let mut parts = vec![];
        while d != 0 {
            parts.push(l.dirs[d].name.as_str());
            d = l.dirs[d].parent;
        }
        parts.reverse();
        parts.join("/")
    }
}

fn tree(ui: &mut egui::Ui, dirs: &[Dir], d: usize, cur: &mut usize) {
    let dir = &dirs[d];
    let label = |ui: &mut egui::Ui, cur: &mut usize| {
        if ui.selectable_label(*cur == d, format!("📁 {}", dir.name)).clicked() {
            *cur = d;
        }
    };
    if dir.dirs.is_empty() {
        ui.horizontal(|ui| {
            ui.add_space(18.0);
            label(ui, cur);
        });
        return;
    }
    let id = ui.make_persistent_id(("dir", d));
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, d == 0)
        .show_header(ui, |ui| label(ui, cur))
        .body(|ui| {
            for &c in &dir.dirs {
                tree(ui, dirs, c, cur);
            }
        });
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if let Some(p) = ui.ctx().input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf())) {
            if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("usmap")) {
                self.drop_usmap(&p);
            } else {
                self.open(p);
            }
        }
        if let Some(rx) = &self.asset_rx {
            match rx.try_recv() {
                Ok(r) => {
                    self.asset_rx = None;
                    let res = &self.loaded.as_ref().unwrap().resolver;
                    self.asset = Some(r.map(|(pkg, ubulk)| {
                        let classes = (0..pkg.exports.len()).map(|i| res.class_name(&pkg, i)).collect();
                        Asset { pkg, classes, ubulk }
                    }));
                    // auto-select the main export: first texture/mesh, else the only one
                    let main = match &self.asset {
                        Some(Ok(a)) => (0..a.pkg.exports.len())
                            .find(|&i| is_texture(&a.pkg.exports[i].name, &a.classes[i]) || is_mesh(&a.classes[i]))
                            .or_else(|| bp::main_3d_export(res, &a.pkg))
                            .or((!a.pkg.exports.is_empty()).then(|| bp::main_export(&a.pkg))),
                        _ => None,
                    };
                    if let Some(i) = main {
                        self.select_export(i);
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => ui.ctx().request_repaint(),
                Err(_) => {
                    self.asset_rx = None;
                    self.asset = Some(Err("loader thread died".into()));
                }
            }
        }

        egui::Panel::top("top").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("archive:");
                let r = ui.add(egui::TextEdit::singleline(&mut self.path_input).desired_width(420.0).hint_text(".utoc / .ucas / .pak / .assets (or drop it here)"));
                if ui.button("Open").clicked() || (r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                    self.open(PathBuf::from(self.path_input.trim()));
                }
                ui.separator();
                ui.toggle_value(&mut self.focus, "⛶ Focus view").on_hover_text("preview fills the window (Ctrl+Space)");
                ui.separator();
                ui.label("🔍");
                if ui.add(egui::TextEdit::singleline(&mut self.search).desired_width(260.0).hint_text("search all files")).changed() {
                    let q = self.search.to_lowercase();
                    self.results = match &self.loaded {
                        Some(l) if !q.is_empty() => (0..l.lower.len()).filter(|&i| l.lower[i].contains(&q) && !is_companion(&l.lower[i])).collect(),
                        _ => vec![],
                    };
                }
            });
        });

        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                if let Some(l) = self.loaded.as_ref().filter(|l| l.toc.unity.is_none()) {
                    ui.separator();
                    match &l.maps {
                        Ok(m) if m.file.as_os_str().is_empty() => ui.weak("mappings: not needed (tagged properties)"),
                        Ok(m) => ui.label(format!("mappings: {}", m.file.file_name().unwrap_or_default().to_string_lossy())),
                        Err(e) => ui.weak(format!("mappings: none ({e})")),
                    };
                }
                if let Some(j) = &self.job {
                    let n = j.ok.load(Relaxed) + j.failed.load(Relaxed);
                    ui.separator();
                    if j.done.load(Relaxed) {
                        ui.label(format!("extracted {} / {}, failed {}", j.ok.load(Relaxed), j.total, j.failed.load(Relaxed)));
                    } else {
                        ui.add(egui::ProgressBar::new(n as f32 / j.total.max(1) as f32).desired_width(200.0).text(format!("{n}/{}", j.total)));
                        ui.ctx().request_repaint();
                    }
                }
            });
        });

        if self.loaded.is_none() {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.centered_and_justified(|ui| ui.heading("Drop a .utoc, .ucas, .pak or Unity .assets file here or enter its path above"));
            });
            return;
        }

        if ui.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, egui::Key::Space)) {
            self.focus = !self.focus;
        }
        let focus = self.focus;
        let mut cur = self.cur_dir;
        if !focus {
            egui::Panel::left("tree").resizable(true).default_size(280.0).show(ui, |ui| {
                egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
                    tree(ui, &self.loaded.as_ref().unwrap().dirs, 0, &mut cur);
                });
            });
        }
        if cur != self.cur_dir {
            self.cur_dir = cur;
            self.search.clear();
            self.results.clear();
        }

        let mut action: Option<Vec<usize>> = None;
        let mut export_clicked = None;
        let info = |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                ui.label("Export to:");
                ui.text_edit_singleline(&mut self.out_dir);
            });
            let folder = self.dir_path(self.cur_dir);
            if ui.button(format!("Extract folder /{folder}")).clicked() {
                action = Some(self.files_under(self.cur_dir));
            }
            ui.separator();
            let l = self.loaded.as_ref().unwrap();
            let Some(f) = self.selected else {
                ui.label("Select a file to preview");
                return;
            };
            let (path, entry) = &l.toc.files[f];
            let size = l.toc.size(*entry);
            ui.label(egui::RichText::new(path).strong());
            ui.label(format!("{}  ·  entry #{entry}", size.map_or("size n/a".into(), human)));
            if ui.button("Extract file").clicked() {
                action = Some(vec![f]);
            }
            ui.separator();
            if let Some(u) = &l.toc.unity {
                let e = *entry;
                match u.class(e) {
                    28 => {
                        let r = self.tex.get_or_insert_with(|| {
                            let img = u.texture(e).map_err(|e| e.to_string())?;
                            let handle = upload_image(ui.ctx(), path, &img);
                            Ok((img, handle))
                        });
                        match r {
                            Err(e) => {
                                ui.colored_label(ui.visuals().error_fg_color, e.as_str());
                            }
                            Ok((img, handle)) => {
                                ui.label(format!("{} x {}  ·  Texture2D  ·  Extract file saves the PNG", img.width, img.height));
                                fit_image(ui, img, handle);
                            }
                        }
                        return;
                    }
                    43 => {
                        match self.mesh.get_or_insert_with(|| u.mesh(e).map(|m| vec![m]).map_err(|e| e.to_string())) {
                            Err(e) => {
                                ui.colored_label(ui.visuals().error_fg_color, e.as_str());
                            }
                            Ok(m) => {
                                // materials -> textures (first call per file builds the mesh->renderer index)
                                let tex = self.mesh_tex.get_or_insert_with(|| {
                                    u.mesh_textures(e)
                                        .into_iter()
                                        .map(|t| t.map(|i| std::sync::Arc::new(egui::ColorImage::from_rgba_unmultiplied([i.width as usize, i.height as usize], &i.rgba))))
                                        .collect()
                                });
                                ui.horizontal(|ui| {
                                    let n = tex.iter().filter(|t| t.is_some()).count();
                                    ui.label(format!("{} verts, {} tris  ·  {n}/{} slots textured", m[0].positions.len(), m[0].tris(), m[0].sections.len()));
                                    if ui.button("Save OBJ + textures").clicked() {
                                        self.status = match out_path(&self.out_dir, path, ".obj").and_then(|p| u.save_obj_textured(e, &p).map(|_| p)) {
                                            Ok(p) => format!("saved {}", p.display()),
                                            Err(e) => format!("save failed: {e}"),
                                        };
                                    }
                                });
                                preview::mesh_view(ui, &m[0], tex, &mut self.mesh_rot);
                            }
                        }
                        return;
                    }
                    83 => {
                        match u.audio_info(e) {
                            Ok(s) => ui.label(format!("AudioClip  ·  {s}")),
                            Err(e) => ui.colored_label(ui.visuals().error_fg_color, e.to_string()),
                        };
                        ui.weak(if path.ends_with(".wav") { "Extract file decodes it to WAV" } else { "Extract file saves the raw FSB5 (play/convert with vgmstream)" });
                        return;
                    }
                    _ => {}
                }
            }
            if self.asset_rx.is_some() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("parsing package…");
                });
                ui.separator();
            }
            if let Some(Err(e)) = &self.asset {
                ui.colored_label(ui.visuals().error_fg_color, format!("package parse failed: {e}"));
                ui.separator();
            }
            if let Some(Ok(a)) = &self.asset {
                ui.label(format!("{}  ·  {} exports{}", a.pkg.name, a.pkg.exports.len(), if a.ubulk.is_some() { "  ·  +ubulk" } else { "" }));
                egui::ScrollArea::vertical().id_salt("exports").max_height(160.0).auto_shrink([false, true]).show(ui, |ui| {
                    for (i, e) in a.pkg.exports.iter().enumerate() {
                        if ui.selectable_label(self.export == Some(i), format!("{}  ({})", e.name, a.classes[i])).clicked() {
                            export_clicked = Some(i);
                        }
                    }
                });
                ui.separator();
                if let Some(i) = self.export {
                    let (name, class) = (&a.pkg.exports[i].name, &a.classes[i]);
                    let is_tex = is_texture(name, class);
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.tab, Tab::Hex, "Hex");
                        if is_tex {
                            ui.selectable_value(&mut self.tab, Tab::Texture, "Texture");
                        } else if self.tab == Tab::Texture {
                            self.tab = Tab::Hex;
                        }
                        if is_mesh(class) {
                            ui.selectable_value(&mut self.tab, Tab::Mesh, "Mesh");
                        } else if self.tab == Tab::Mesh {
                            self.tab = Tab::Hex;
                        }
                        if is_material(class) {
                            ui.selectable_value(&mut self.tab, Tab::Material, "Material");
                        } else if self.tab == Tab::Material {
                            self.tab = Tab::Hex;
                        }
                        if class == "AnimSequence" {
                            ui.selectable_value(&mut self.tab, Tab::Anim, "Anim");
                        } else if self.tab == Tab::Anim {
                            self.tab = Tab::Hex;
                        }
                        // any export: assemblies, Skeleton -> SkeletalMesh, or meshes it references (bp::assemble_any)
                        if !is_tex && !is_mesh(class) {
                            ui.selectable_value(&mut self.tab, Tab::Blueprint, "3D");
                        } else if self.tab == Tab::Blueprint {
                            self.tab = Tab::Hex;
                        }
                        ui.selectable_value(&mut self.tab, Tab::Props, "Properties");
                    });
                    match self.tab {
                        Tab::Hex => hex_view(ui, "exphex", a.pkg.export_data(i)),
                        Tab::Blueprint => {
                            let l = self.loaded.as_ref().unwrap();
                            let r = self.bp.get_or_insert_with(|| {
                                let maps = l.maps.as_ref().map_err(|e| format!("Blueprint assembly needs mappings: {e}"))?;
                                let mut ucas = File::open(data_path(&l.utoc)).map_err(|e| e.to_string())?;
                                let mut load = |p: &str| -> Option<Vec<u8>> {
                                    let (_, e) = l.toc.files.iter().find(|(q, _)| q == p)?;
                                    read_chunk(&l.toc, &mut ucas, *e, u64::MAX).ok()
                                };
                                bp::assemble_any(&l.resolver, maps, &a.pkg, i, &mut load).map_err(|e| e.to_string())
                            });
                            match r {
                                Err(e) => {
                                    ui.colored_label(ui.visuals().error_fg_color, e.as_str());
                                }
                                Ok(asm) => {
                                    // meshes + textures load on a background thread
                                    if self.bp_mesh.is_none() && self.bp_rx.is_none() {
                                        self.bp_rx = Some(preview::spawn_bp(ui.ctx().clone(), l.toc.clone(), l.utoc.clone(), l.resolver.clone(), asm.clone()));
                                    }
                                    if let Some((m, imgs, w)) = self.bp_rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
                                        self.mesh_tex = Some(preview::prepare(imgs));
                                        self.bp_mesh = Some((m, w));
                                        self.bp_rx = None;
                                    }
                                    let meshes = asm.parts.iter().filter(|p| p.mesh.is_some()).count();
                                    let unknown = asm.parts.iter().filter(|p| p.mesh.is_some() && !p.transform_known).count();
                                    ui.horizontal(|ui| {
                                        ui.label(format!("{} components  ·  {meshes} with a mesh  ·  {unknown} of those with unknown transform", asm.parts.len()));
                                        if ui.button("Export assembly").clicked() {
                                            self.status = match save_bp(l, asm, path, &self.out_dir) {
                                                Ok(p) => format!("saved {}", p.display()),
                                                Err(e) => format!("export assembly failed: {e}"),
                                            };
                                        }
                                    });
                                    let warn = ui.visuals().warn_fg_color;
                                    egui::CollapsingHeader::new(format!("parts ({})", asm.parts.len())).default_open(asm.parts.len() <= 50).show(ui, |ui| {
                                        egui::ScrollArea::vertical().id_salt("bpparts").max_height(180.0).auto_shrink([false, true]).show(ui, |ui| {
                                            for p in &asm.parts {
                                                let mesh = p.mesh.as_deref().map_or("-", |m| m.rsplit('/').next().unwrap_or(m));
                                                let inst = p.instances.as_ref().map(|v| format!("  ·  {} instances", v.len())).unwrap_or_default();
                                                let line = format!("{}  ({})  {mesh}{inst}", p.name, p.class);
                                                if p.transform_known {
                                                    ui.label(line);
                                                } else {
                                                    ui.colored_label(warn, format!("{line}  ·  transform unknown"));
                                                }
                                                for w in &p.warnings {
                                                    ui.weak(format!("      {w}"));
                                                }
                                            }
                                            for w in &asm.warnings {
                                                ui.colored_label(warn, w.as_str());
                                            }
                                        });
                                    });
                                    match &self.bp_mesh {
                                        None => {
                                            ui.horizontal(|ui| {
                                                ui.spinner();
                                                ui.weak("loading meshes and textures…");
                                            });
                                        }
                                        Some((m, w)) => {
                                            for w in w {
                                                ui.colored_label(warn, w.as_str());
                                            }
                                            if m.indices.is_empty() {
                                                ui.weak("no mesh parts to preview");
                                            } else {
                                                ui.weak(format!("{} tris", m.tris()));
                                                preview::mesh_view_framed(ui, m, self.mesh_tex.as_deref().unwrap_or(&[]), &mut self.mesh_rot, preview::part_framing(asm));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Tab::Material => {
                            let l = self.loaded.as_ref().unwrap();
                            if self.mat_tex.is_none() && self.mat_rx.is_none() {
                                self.mat_rx = Some(preview::spawn_material_images(
                                    ui.ctx().clone(), l.toc.clone(), l.utoc.clone(), l.resolver.clone(), a.pkg.name.clone(),
                                ));
                            }
                            if let Some(imgs) = self.mat_rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
                                let ctx = ui.ctx().clone();
                                let max = ctx.input(|i| i.max_texture_side);
                                self.mat_tex = Some(
                                    imgs.into_iter()
                                        .enumerate()
                                        .map(|(k, (param, tex, role, img))| {
                                            let img = img.map(std::sync::Arc::new);
                                            let h = img.as_ref().filter(|i| i.size[0] <= max && i.size[1] <= max)
                                                .map(|i| ctx.load_texture(format!("mat{k}"), (**i).clone(), egui::TextureOptions::LINEAR_REPEAT));
                                            (param, tex, role, h, img)
                                        })
                                        .collect(),
                                );
                                self.mat_rx = None;
                            }
                            ui.horizontal(|ui| {
                                if ui.button("Export material").clicked() {
                                    self.status = match save_material(l, &a.pkg.name, path, &self.out_dir) {
                                        Ok(p) => format!("saved {}", p.display()),
                                        Err(e) => format!("export material failed: {e}"),
                                    };
                                }
                                if self.mat_rx.is_some() {
                                    ui.spinner();
                                    ui.weak("loading textures…");
                                }
                            });
                            match &self.mat_tex {
                                Some(t) if t.is_empty() => {
                                    ui.weak("no textures found for this material");
                                }
                                Some(t) => {
                                    // thumbnails: parameter, texture, role
                                    egui::ScrollArea::vertical().id_salt("matthumbs").max_height(170.0).show(ui, |ui| {
                                        ui.horizontal_wrapped(|ui| {
                                            for (param, tex, role, h, _) in t {
                                                ui.vertical(|ui| {
                                                    ui.set_width(110.0);
                                                    match h {
                                                        Some(h) => {
                                                            ui.add(egui::Image::new(h).fit_to_exact_size(egui::vec2(96.0, 96.0)));
                                                        }
                                                        None => {
                                                            ui.weak("(no image)");
                                                        }
                                                    }
                                                    ui.small(if param.is_empty() { tex.as_str() } else { param.as_str() }).on_hover_text(tex.as_str());
                                                    if !role.is_empty() {
                                                        ui.weak(*role);
                                                    }
                                                });
                                            }
                                        });
                                    });
                                    // sphere with the diffuse (first entry if it is one)
                                    let diffuse = t.iter().find(|x| x.2 == "diffuse").or(t.first()).and_then(|x| x.4.clone());
                                    preview::mesh_view(ui, &preview::sphere(), &[diffuse], &mut self.mesh_rot);
                                }
                                None => {}
                            }
                        }
                        Tab::Props => {
                            let l = self.loaded.as_ref().unwrap();
                            let maps = match &l.maps {
                                Ok(m) => m,
                                Err(e) => {
                                    ui.colored_label(ui.visuals().error_fg_color, format!("no mappings: {e}"));
                                    return;
                                }
                            };
                            let (rep, lines) = self.props.get_or_insert_with(|| {
                                let r = props::report(&a.pkg, i, maps, &l.resolver);
                                let mut json = match &r.props {
                                    Ok((p, _)) => props::to_json(p),
                                    Err(_) => String::new(),
                                };
                                if let Some(Ok(rows)) = &r.rows {
                                    json.push_str(&format!("\n\nrows ({}):\n{}", rows.len(), props::to_json(rows)));
                                }
                                let lines = json.lines().map(String::from).collect();
                                (r, lines)
                            });
                            ui.horizontal(|ui| {
                                let parsed = match &rep.props {
                                    Ok((p, n)) => format!("{} properties  ·  {n} of {} bytes", p.len(), rep.size),
                                    Err(_) => format!("{} bytes", rep.size),
                                };
                                ui.label(format!("{class} ({})  ·  {parsed}", rep.kind));
                                if ui.button("Save JSON").clicked() {
                                    self.status = match out_path(&self.out_dir, path, ".json")
                                        .and_then(|p| write_file(&p, props::package_json(&a.pkg, maps, &l.resolver).as_bytes()).map(|_| p))
                                    {
                                        Ok(p) => format!("saved {}", p.display()),
                                        Err(e) => format!("save JSON failed: {e}"),
                                    };
                                }
                            });
                            let err = ui.visuals().error_fg_color;
                            for e in [rep.props.as_ref().err(), rep.rows.as_ref().and_then(|r| r.as_ref().err())].into_iter().flatten() {
                                ui.colored_label(err, format!("parse failed: {e}"));
                            }
                            if let Some(w) = &rep.warning {
                                ui.colored_label(ui.visuals().warn_fg_color, w.as_str());
                            }
                            let row_h = ui.text_style_height(&egui::TextStyle::Monospace);
                            egui::ScrollArea::both().id_salt("props").auto_shrink(false).show_rows(ui, row_h, lines.len(), |ui, range| {
                                for line in &lines[range] {
                                    ui.add(egui::Label::new(egui::RichText::new(line).monospace()).extend());
                                }
                            });
                        }
                        Tab::Mesh => {
                            let r = self.mesh.get_or_insert_with(|| decode_for_preview(a, i).map_err(|e| e.to_string()));
                            let skel = class == "SkeletalMesh";
                            match r {
                                Err(e) => {
                                    ui.colored_label(ui.visuals().error_fg_color, e.as_str());
                                }
                                Ok(lods) => {
                                    let preview = lods.iter().find(|m| m.tris() <= MAX_PREVIEW_TRIS).unwrap_or(&lods[lods.len() - 1]);
                                    ui.horizontal(|ui| {
                                        ui.label(format!(
                                            "{} LODs  ·  LOD0 {} verts, {} tris  ·  preview {} tris",
                                            lods.len(),
                                            lods[0].positions.len(),
                                            lods[0].tris(),
                                            preview.tris()
                                        ));
                                        if ui.button(if skel { "Save GLB (rigged)" } else { "Save OBJ + textures" }).clicked() {
                                            self.status = match save_mesh(self.loaded.as_ref().unwrap(), a, i, path, &self.out_dir) {
                                                Ok(p) => format!("saved {}", p.display()),
                                                Err(e) => format!("save failed: {e}"),
                                            };
                                        }
                                    });
                                    // textures resolve on a background thread; preview is untextured until then
                                    if self.mesh_tex.is_none() && self.mesh_tex_rx.is_none() {
                                        let l = self.loaded.as_ref().unwrap();
                                        let need = preview.sections.iter().map(|s| s.0 as usize + 1).max().unwrap_or(1);
                                        let entry = l.toc.files[self.selected.unwrap_or(0)].1;
                                        self.mesh_tex_rx = Some(preview::spawn_slot_images(
                                            ui.ctx().clone(), l.toc.clone(), l.utoc.clone(), l.resolver.clone(), entry, i, need,
                                        ));
                                    }
                                    if let Some(imgs) = self.mesh_tex_rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
                                        self.mesh_tex = Some(preview::prepare(imgs));
                                        self.mesh_tex_rx = None;
                                    }
                                    ui.horizontal(|ui| {
                                        if self.mesh_tex_rx.is_some() {
                                            ui.spinner();
                                            ui.weak("loading textures…");
                                        }
                                    });
                                    preview::mesh_view(ui, preview, self.mesh_tex.as_deref().unwrap_or(&[]), &mut self.mesh_rot);
                                }
                            }
                        }
                        Tab::Anim => match self.anim.get_or_insert_with(|| anim::decode_anim_sequence(&a.pkg, i).map_err(|e| e.to_string())) {
                            Err(e) => {
                                ui.colored_label(ui.visuals().error_fg_color, e.as_str());
                            }
                            Ok(an) => {
                                ui.label(format!(
                                    "{} frames  ·  {:.2} s @ {} fps  ·  {} tracks  ·  {}",
                                    an.frames,
                                    an.length(),
                                    an.rate,
                                    an.tracks.len(),
                                    an.codec
                                ));
                                if ui.button("Save GLB (skeleton + mesh + animation)").clicked() {
                                    self.status = match save_anim(self.loaded.as_ref().unwrap(), a, i, path, &self.out_dir) {
                                        Ok((p, mesh)) => format!("saved {} (mesh: {})", p.display(), mesh.as_deref().unwrap_or("none, skeleton only")),
                                        Err(e) => format!("save failed: {e}"),
                                    };
                                }
                            }
                        },
                        Tab::Texture => {
                            let r = self.tex.get_or_insert_with(|| {
                                let img = texture::decode_texture(&a.pkg, i, a.ubulk.as_deref())
                                    .map_err(|e| e.to_string())?;
                                let handle = upload_image(ui.ctx(), name, &img);
                                Ok((img, handle))
                            });
                            match r {
                                Err(e) => {
                                    ui.colored_label(ui.visuals().error_fg_color, format!("texture: {e}"));
                                }
                                Ok((img, handle)) => {
                                    ui.horizontal(|ui| {
                                        ui.label(format!("{} x {}  ·  {class}", img.width, img.height));
                                        if ui.button("Save PNG").clicked() {
                                            self.status = match texture::to_png(img)
                                                .and_then(|png| out_path(&self.out_dir, path, ".png").and_then(|p| write_file(&p, &png).map(|_| p)))
                                            {
                                                Ok(p) => format!("saved {}", p.display()),
                                                Err(e) => format!("save PNG failed: {e}"),
                                            };
                                        }
                                    });
                                    fit_image(ui, img, handle);
                                }
                            }
                        }
                    }
                    return;
                }
            }
            if size.is_none_or(|s| s > PREVIEW_BYTES) {
                ui.weak(format!("hex preview: first {}", human(PREVIEW_BYTES)));
            }
            hex_view(ui, "hex", &self.preview);
        };
        if focus {
            egui::CentralPanel::default().show(ui, info);
        } else {
            egui::Panel::right("info").resizable(true).default_size(460.0).show(ui, info);
        }
        if let Some(i) = export_clicked {
            self.select_export(i);
        }
        if let Some(files) = action {
            self.extract(files);
        }

        let mut clicked = None;
        if !focus {
            egui::CentralPanel::default().show(ui, |ui| {
                let l = self.loaded.as_ref().unwrap();
                let searching = !self.search.is_empty();
                let list: &[usize] = if searching { &self.results } else { &l.dirs[self.cur_dir].files };
                ui.label(if searching {
                    format!("{} results", list.len())
                } else {
                    format!("/{}  ·  {} folders, {} files", self.dir_path(self.cur_dir), l.dirs[self.cur_dir].dirs.len(), list.len())
                });
                ui.separator();
                let row_h = ui.spacing().interact_size.y;
                egui::ScrollArea::vertical().id_salt("files").auto_shrink(false).show_rows(ui, row_h, list.len(), |ui, range| {
                    for &f in &list[range] {
                        let (path, entry) = &l.toc.files[f];
                        let name = if searching { path.as_str() } else { path.rsplit('/').next().unwrap_or(path) };
                        let size = l.toc.size(*entry);
                        ui.horizontal(|ui| {
                            if ui.selectable_label(self.selected == Some(f), name).clicked() {
                                clicked = Some(f);
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.weak(size.map(human).unwrap_or_default()));
                        });
                    }
                });
            });
        }
        if let Some(f) = clicked {
            self.select(f);
        }
    }
}

pub fn run(path: Option<String>) -> Res<()> {
    let mut app = App::default();
    if let Some(p) = path {
        app.open(PathBuf::from(p));
    }
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 850.0]).with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "UCAS Explorer",
        opts,
        Box::new(|cc| {
            if let Some(rs) = cc.wgpu_render_state.clone() {
                crate::viewport::init(rs);
            }
            Ok(Box::new(app))
        }),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn tree_groups_paths() {
        let toc = crate::Toc {
            block_size: 1,
            ids: vec![],
            pak: None,
            unity: None,
            chunks: vec![],
            blocks: vec![],
            methods: vec![],
            files: vec![("A/B/x.uasset".into(), 0), ("A/B/x.uexp".into(), 1), ("A/B/z.uasset".into(), 2), ("A/y.uasset".into(), 3)],
        };
        let d = super::build_tree(&toc);
        assert_eq!(d.len(), 3); // root, A, B
        assert_eq!(d[1].files, vec![3]);
        assert_eq!(d[2].files, vec![0, 2]); // x.uexp hidden...
        assert_eq!(super::with_companions(&toc, vec![0]), vec![0, 1]); // ...but extracted with x.uasset
        assert_eq!(super::human(1536), "1.5 KB");
    }
}
