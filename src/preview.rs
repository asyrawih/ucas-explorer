//! Textured, flat-shaded CPU preview of a mesh in egui (drag to orbit).
// ponytail: painter's algorithm over all triangles each frame (no z-buffer); fine for preview-sized
// LODs. Upgrade path: egui_wgpu paint callback with a real depth buffer.
use crate::{data_path, global::Resolver, material, read_chunk, zen, Res, Toc};
use eframe::egui;
use std::path::PathBuf;
use std::sync::{mpsc, Arc};

pub type SlotImages = Vec<Option<egui::ColorImage>>;

/// Resolves the diffuse texture of each material slot of mesh export `export` (package = toc entry
/// `entry`) on a background thread, decoded and ready to upload. `need` = max(MaterialIndex)+1.
/// Repaints `ctx` when done. Errors yield an empty list (preview just stays untextured).
pub fn spawn_slot_images(
    ctx: egui::Context,
    toc: Arc<Toc>,
    archive: PathBuf,
    res: Arc<Resolver>,
    entry: u32,
    export: usize,
    need: usize,
) -> mpsc::Receiver<SlotImages> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let imgs = (|| -> Res<SlotImages> {
            let mut ucas = std::fs::File::open(data_path(&archive))?;
            let pkg = zen::parse(read_chunk(&toc, &mut ucas, entry, u64::MAX)?)?;
            let mut load = |p: &str| -> Option<Vec<u8>> {
                let (_, e) = toc.files.iter().find(|(q, _)| q == p)?;
                read_chunk(&toc, &mut ucas, *e, u64::MAX).ok()
            };
            let pngs = material::slot_diffuse_pngs(&res, &pkg, export, need, &mut load);
            Ok(pngs.iter().map(|p| p.as_deref().and_then(decode_png)).collect())
        })()
        .unwrap_or_default();
        let _ = tx.send(imgs);
        ctx.request_repaint();
    });
    rx
}

/// Merged Blueprint mesh, image per global material slot, build warnings.
pub type BpBuilt = (crate::mesh::Mesh, SlotImages, Vec<String>);

/// Builds the merged mesh of a Blueprint assembly and decodes each slot's diffuse texture on a
/// background thread. Repaints `ctx` when done.
pub fn spawn_bp(ctx: egui::Context, toc: Arc<Toc>, archive: PathBuf, res: Arc<Resolver>, asm: crate::bp::Assembly) -> mpsc::Receiver<BpBuilt> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let r = std::fs::File::open(data_path(&archive)).map(|mut ucas| {
            let mut load = |p: &str| -> Option<Vec<u8>> {
                let (_, e) = toc.files.iter().find(|(q, _)| q == p)?;
                read_chunk(&toc, &mut ucas, *e, u64::MAX).ok()
            };
            // ponytail: per-mesh triangle cap keeps the preview responsive (maps: thousands of meshes); export uses LOD0
            let cap = if asm.parts.len() > 200 { 2_000 } else { 20_000 };
            let b = crate::bp::build(&res, &asm, cap, &mut load);
            let mut cache: std::collections::HashMap<String, Option<egui::ColorImage>> = Default::default();
            let imgs = b.slots.iter().map(|s| {
                let t = s.diffuse.as_ref()?;
                cache.entry(t.clone()).or_insert_with(|| material::texture_to_png(&res, t, &mut load).as_deref().and_then(decode_png)).clone()
            });
            (imgs.collect(), b)
        });
        let msg = match r {
            Ok((imgs, b)) => (b.mesh, imgs, b.warnings),
            Err(e) => (crate::mesh::Mesh { positions: vec![], normals: vec![], uvs: vec![], indices: vec![], sections: vec![] }, vec![], vec![e.to_string()]),
        };
        let _ = tx.send(msg);
        ctx.request_repaint();
    });
    rx
}

/// (parameter, texture name, role, image) per texture of a material, diffuse first.
pub type MaterialImages = Vec<(String, String, &'static str, Option<egui::ColorImage>)>;

/// Resolves and decodes the textures of material package `package` ("/Game/.../MI_Foo") on a
/// background thread. Repaints `ctx` when done.
pub fn spawn_material_images(ctx: egui::Context, toc: Arc<Toc>, archive: PathBuf, res: Arc<Resolver>, package: String) -> mpsc::Receiver<MaterialImages> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let imgs = std::fs::File::open(data_path(&archive))
            .map(|mut ucas| {
                let mut load = |p: &str| -> Option<Vec<u8>> {
                    let (_, e) = toc.files.iter().find(|(q, _)| q == p)?;
                    read_chunk(&toc, &mut ucas, *e, u64::MAX).ok()
                };
                material::material_preview(&res, &package, &mut load)
                    .into_iter()
                    .map(|t| {
                        let name = t.texture.rsplit('/').next().unwrap_or(&t.texture).to_string();
                        (t.param, name, t.role, t.png.as_deref().and_then(decode_png))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let _ = tx.send(imgs);
        ctx.request_repaint();
    });
    rx
}

/// UV sphere (UE space, radius 100, one section using material slot 0) to preview materials on.
pub fn sphere() -> crate::mesh::Mesh {
    let (seg, rings) = (48u32, 24u32);
    let (mut positions, mut uvs, mut normals, mut indices) = (vec![], vec![], vec![], vec![]);
    for r in 0..=rings {
        let lat = std::f32::consts::PI * r as f32 / rings as f32;
        for s in 0..=seg {
            let lon = std::f32::consts::TAU * s as f32 / seg as f32;
            let n = [lat.sin() * lon.cos(), lat.sin() * lon.sin(), lat.cos()];
            positions.push(n.map(|v| v * 100.0));
            normals.push(n);
            uvs.push([2.0 * s as f32 / seg as f32, r as f32 / rings as f32]); // tile 2x around
        }
    }
    for r in 0..rings {
        for s in 0..seg {
            let (a, b) = (r * (seg + 1) + s, (r + 1) * (seg + 1) + s);
            indices.extend([a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    let n = indices.len() as u32 / 3;
    crate::mesh::Mesh { positions, normals, uvs, indices, sections: vec![(0, 0, n)] }
}

/// RGBA8 PNG (as written by texture::to_png) -> egui image.
fn decode_png(png: &[u8]) -> Option<egui::ColorImage> {
    let mut r = png::Decoder::new(std::io::Cursor::new(png)).read_info().ok()?;
    let mut buf = vec![0; r.output_buffer_size()?];
    let info = r.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width as usize, info.height as usize);
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return None;
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], buf.get(..w * h * 4)?))
}

/// A slot texture for the preview rasterizer (CPU side; sampled per pixel).
pub type Tex = Option<Arc<egui::ColorImage>>;

/// Decoded slot images -> preview textures.
pub fn prepare(imgs: SlotImages) -> Vec<Tex> {
    imgs.into_iter().map(|i| i.map(Arc::new)).collect()
}

/// Software rasterizer with a depth buffer: correct occlusion for any mesh, both faces drawn (no
/// culling), textures sampled per pixel (nearest, repeat). Orthographic, so affine UVs are exact.
// ponytail: CPU raster at panel resolution, re-rendered only when view/mesh/size change; a GPU paint
// callback is the upgrade if huge meshes make rotating sluggish.
fn raster(m: &crate::mesh::Mesh, tex: &[Tex], rot: (f32, f32), w: usize, h: usize, bg: egui::Color32, frame: Option<([f32; 3], f32)>) -> egui::ColorImage {
    let (c, r) = frame.unwrap_or_else(|| framing(&m.positions));
    let scale = w.min(h) as f32 * 0.42 / r;
    let (near, far) = (-1.6 * r, 1.6 * r); // depth clip around the framed region
    let (sy, cy, sp, cp) = (rot.0.sin(), rot.0.cos(), rot.1.sin(), rot.1.cos());
    // UE is Z-up: yaw around Z, then pitch around the view's X axis; depth = Y' (larger = farther)
    let xf: Vec<[f32; 3]> = m
        .positions
        .iter()
        .map(|p| {
            let (x, y, z) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
            let (x1, y1) = (x * cy - y * sy, x * sy + y * cy);
            // (screen x, screen y, depth)
            [x1 * scale + w as f32 / 2.0, -(y1 * sp + z * cp) * scale + h as f32 / 2.0, y1 * cp - z * sp]
        })
        .collect();
    let mut slot = vec![0u32; m.indices.len() / 3];
    for &(mat, first, n) in &m.sections {
        for s in slot.iter_mut().skip(first as usize / 3).take(n as usize) {
            *s = mat;
        }
    }
    let mut zbuf = vec![f32::INFINITY; w * h];
    let mut px = vec![bg; w * h];
    for (k, t) in m.indices.chunks_exact(3).enumerate() {
        let [a, b, cc] = [xf[t[0] as usize], xf[t[1] as usize], xf[t[2] as usize]];
        let area = (b[0] - a[0]) * (cc[1] - a[1]) - (b[1] - a[1]) * (cc[0] - a[0]);
        if area.abs() < 1e-12 {
            continue;
        }
        // flat light from the camera: |normal . view| (view axis = depth, index 2)
        let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [cc[0] - a[0], cc[1] - a[1], cc[2] - a[2]]);
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        let shade = 0.35 + 0.65 * (n[2] / len).abs();
        let img = tex.get(slot[k] as usize).and_then(|t| t.as_deref());
        let uv = |i: u32| m.uvs.get(i as usize).copied().unwrap_or([0.0, 0.0]);
        let (ua, ub, uc) = (uv(t[0]), uv(t[1]), uv(t[2]));
        let x0 = a[0].min(b[0]).min(cc[0]).floor().max(0.0) as usize;
        let x1 = (a[0].max(b[0]).max(cc[0]).ceil() as isize).min(w as isize - 1);
        let y0 = a[1].min(b[1]).min(cc[1]).floor().max(0.0) as usize;
        let y1 = (a[1].max(b[1]).max(cc[1]).ceil() as isize).min(h as isize - 1);
        if x1 < 0 || y1 < 0 {
            continue;
        }
        for y in y0..=y1 as usize {
            let fy = y as f32 + 0.5;
            for x in x0..=x1 as usize {
                let fx = x as f32 + 0.5;
                let w0 = ((b[0] - fx) * (cc[1] - fy) - (b[1] - fy) * (cc[0] - fx)) / area;
                let w1 = ((cc[0] - fx) * (a[1] - fy) - (cc[1] - fy) * (a[0] - fx)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * a[2] + w1 * b[2] + w2 * cc[2];
                let o = y * w + x;
                if z >= zbuf[o] || z < near || z > far {
                    continue;
                }
                zbuf[o] = z;
                let base = match img {
                    Some(im) => {
                        let tu = w0 * ua[0] + w1 * ub[0] + w2 * uc[0];
                        let tv = w0 * ua[1] + w1 * ub[1] + w2 * uc[1];
                        let ix = ((tu.rem_euclid(1.0) * im.size[0] as f32) as usize).min(im.size[0] - 1);
                        let iy = ((tv.rem_euclid(1.0) * im.size[1] as f32) as usize).min(im.size[1] - 1);
                        im.pixels[iy * im.size[0] + ix]
                    }
                    None => egui::Color32::from_rgb(219, 219, 230),
                };
                let f = |c: u8| (c as f32 * shade) as u8;
                px[o] = egui::Color32::from_rgb(f(base.r()), f(base.g()), f(base.b()));
            }
        }
    }
    egui::ColorImage { size: [w, h], pixels: px, source_size: egui::vec2(w as f32, h as f32) }
}

/// Center and radius to frame: the bounding box, unless a few far-out parts (sky spheres, backdrops,
/// planets around a level) make it much larger than where most vertices are; then the 5%..95% box.
fn framing(pos: &[[f32; 3]]) -> ([f32; 3], f32) {
    let (lo, hi) = pos.iter().fold(([f32::MAX; 3], [f32::MIN; 3]), |(lo, hi), p| {
        (std::array::from_fn(|k| lo[k].min(p[k])), std::array::from_fn(|k| hi[k].max(p[k])))
    });
    let full = |lo: [f32; 3], hi: [f32; 3]| -> ([f32; 3], f32) {
        (std::array::from_fn(|k| (lo[k] + hi[k]) / 2.0), (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max).max(1e-3) / 2.0)
    };
    let (c, r) = full(lo, hi);
    if pos.len() < 64 {
        return (c, r);
    }
    let step = (pos.len() / 20_000).max(1);
    let q: Vec<[f32; 2]> = (0..3)
        .map(|k| {
            let mut v: Vec<f32> = pos.iter().step_by(step).map(|p| p[k]).collect();
            v.sort_by(f32::total_cmp);
            [v[v.len() / 20], v[v.len() * 19 / 20]]
        })
        .collect();
    let (rc, rr) = full(std::array::from_fn(|k| q[k][0]), std::array::from_fn(|k| q[k][1]));
    if r > 4.0 * rr { (rc, rr * 1.2) } else { (c, r) }
}

/// Framing for an assembly with many parts (maps): where most parts are (10%..90% of their world
/// positions), so a sky sphere, planets or backdrop cards around the level don't shrink it to a dot.
pub fn part_framing(asm: &crate::bp::Assembly) -> Option<([f32; 3], f32)> {
    let t: Vec<[f64; 3]> = asm.parts.iter().filter(|p| p.mesh.is_some()).map(|p| p.world.trans).collect();
    if t.len() < 20 {
        return None;
    }
    let q: Vec<[f32; 2]> = (0..3)
        .map(|k| {
            let mut v: Vec<f64> = t.iter().map(|p| p[k]).collect();
            v.sort_by(f64::total_cmp);
            [v[v.len() / 10] as f32, v[v.len() * 9 / 10] as f32]
        })
        .collect();
    let c = std::array::from_fn(|k| (q[k][0] + q[k][1]) / 2.0);
    let r = (0..3).map(|k| q[k][1] - q[k][0]).fold(0.0f32, f32::max) / 2.0 * 2.0;
    Some((c, r.max(200.0)))
}

pub fn mesh_view(ui: &mut egui::Ui, m: &crate::mesh::Mesh, tex: &[Tex], cam: &mut crate::viewport::Camera) {
    mesh_view_framed(ui, m, tex, cam, None)
}

/// 3D view of `m`, framed on `(center, radius)` (default: framing of its vertices, see part_framing):
/// GPU viewport with Blender-like navigation, or the CPU rasterizer (orthographic) without a GPU.
pub fn mesh_view_framed(ui: &mut egui::Ui, m: &crate::mesh::Mesh, tex: &[Tex], cam: &mut crate::viewport::Camera, frame: Option<([f32; 3], f32)>) {
    let size = ui.available_size() - egui::vec2(4.0, 4.0);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(size.x, size.y.max(200.0)), egui::Sense::click_and_drag());
    // framing scans every vertex: computed once per mesh
    let fid = resp.id.with("framing");
    let mkey = (m.positions.as_ptr() as usize, m.positions.len());
    let (c, r) = frame.unwrap_or_else(|| match ui.ctx().data(|d| d.get_temp::<((usize, usize), ([f32; 3], f32))>(fid)) {
        Some((k, f)) if k == mkey => f,
        _ => {
            let f = framing(&m.positions);
            ui.ctx().data_mut(|d| d.insert_temp(fid, (mkey, f)));
            f
        }
    });
    if cam.navigate(ui, &resp, r) {
        ui.ctx().request_repaint();
    }
    if !crate::viewport::draw(ui, rect, m, tex, cam, (c, r)) {
        cpu_view(ui, rect, &resp, m, tex, cam, (c, r));
    }
    ui.painter_at(rect).text(
        rect.left_bottom() + egui::vec2(6.0, -6.0),
        egui::Align2::LEFT_BOTTOM,
        "LMB orbit · RMB/Shift pan · scroll zoom · WASD fly, Q/E down/up (Shift fast) · F frame",
        egui::FontId::proportional(11.0),
        ui.visuals().weak_text_color(),
    );
}

/// CPU fallback: orthographic, zoom/pan by shrinking/moving the framed region.
fn cpu_view(ui: &egui::Ui, rect: egui::Rect, resp: &egui::Response, m: &crate::mesh::Mesh, tex: &[Tex], cam: &crate::viewport::Camera, (c, r): ([f32; 3], f32)) {
    let frame = (std::array::from_fn(|k| c[k] + cam.target[k]), r * cam.dist / crate::viewport::Camera::default().dist);
    let ppp = ui.ctx().pixels_per_point();
    let (w, h) = (((rect.width() * ppp) as usize).clamp(16, 2048), ((rect.height() * ppp) as usize).clamp(16, 2048));
    let bg = ui.visuals().extreme_bg_color;
    // re-render only when something visible changed; framebuffer texture kept in egui memory
    let f = frame.0.iter().chain([&frame.1, &cam.yaw, &cam.pitch]).fold(0usize, |a, x| a.wrapping_mul(31).wrapping_add(x.to_bits() as usize));
    let key = (f, w, h, m.positions.as_ptr() as usize, m.indices.len(), tex.iter().map(|t| t.as_ref().map_or(0, |a| Arc::as_ptr(a) as usize)).sum::<usize>());
    let id = resp.id.with("mesh_fb");
    let cached: Option<(egui::TextureHandle, (usize, usize, usize, usize, usize, usize))> = ui.ctx().data(|dm| dm.get_temp(id));
    let handle = match cached {
        Some((hd, k)) if k == key => hd,
        other => {
            let img = raster(m, tex, (cam.yaw, cam.pitch), w, h, bg, Some(frame));
            let hd = match other {
                Some((mut hd, _)) => {
                    hd.set(img, egui::TextureOptions::LINEAR);
                    hd
                }
                None => ui.ctx().load_texture("mesh_fb", img, egui::TextureOptions::LINEAR),
            };
            ui.ctx().data_mut(|dm| dm.insert_temp(id, (hd.clone(), key)));
            hd
        }
    };
    ui.painter_at(rect).image(handle.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
}

#[cfg(test)]
mod tests {
    #[test]
    fn png_roundtrip() {
        let img = crate::texture::Image { width: 2, height: 1, rgba: vec![255, 0, 0, 255, 0, 0, 255, 128] };
        let ci = super::decode_png(&crate::texture::to_png(&img).unwrap()).unwrap();
        assert_eq!(ci.size, [2, 1]);
        assert_eq!(ci.pixels[0], eframe::egui::Color32::from_rgba_unmultiplied(255, 0, 0, 255));
        assert!(super::decode_png(b"not a png").is_none());
    }

    /// Renders the 3D view of one package exactly as the GUI does, to a PNG:
    /// `UCAS_TEST_PAK=x.pak PREVIEW_PKG=LVL_SpaceRig.umap PREVIEW_OUT=out.png cargo test preview_package -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn preview_package() {
        let pak = std::path::PathBuf::from(std::env::var("UCAS_TEST_PAK").unwrap());
        let want = std::env::var("PREVIEW_PKG").unwrap();
        let toc = crate::open_archive(&pak).unwrap();
        let res = crate::global::Resolver::load(&pak.with_file_name("global.utoc"), &toc).unwrap();
        let maps = crate::props::Mappings::default();
        let mut f = std::fs::File::open(&pak).unwrap();
        let (_, e) = toc.files.iter().find(|(p, _)| p.ends_with(&want)).expect("package not found").clone();
        let pkg = crate::zen::parse(crate::read_chunk(&toc, &mut f, e, u64::MAX).unwrap()).unwrap();
        let mut load = |q: &str| -> Option<Vec<u8>> {
            let (_, e) = toc.files.iter().find(|(p, _)| p == q)?;
            crate::read_chunk(&toc, &mut f, *e, u64::MAX).ok()
        };
        let t = std::time::Instant::now();
        let i = crate::bp::main_3d_export(&res, &pkg).unwrap_or_else(|| crate::bp::main_export(&pkg));
        let asm = crate::bp::assemble_any(&res, &maps, &pkg, i, &mut load).unwrap();
        asm.warnings.iter().filter(|w| w.contains("sublevel") || w.contains("merged")).for_each(|w| println!("\n  {w}"));
        let cap = if asm.parts.len() > 200 { 2_000 } else { 20_000 };
        let b = crate::bp::build(&res, &asm, cap, &mut load);
        let t_build = t.elapsed();
        let imgs: Vec<super::Tex> = b.slots.iter().map(|s| s.diffuse.as_ref().and_then(|d| crate::material::texture_to_png(&res, d, &mut load)).as_deref().and_then(super::decode_png).map(std::sync::Arc::new)).collect();
        let t = std::time::Instant::now();
        let img = super::raster(&b.mesh, &imgs, (0.6, 0.35), 900, 700, eframe::egui::Color32::from_gray(20), super::part_framing(&asm));
        println!("{} parts, {} tris, build {:?}, raster {:?}", asm.parts.len(), b.mesh.tris(), t_build, t.elapsed());
        let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
        let out = crate::texture::Image { width: 900, height: 700, rgba };
        std::fs::write(std::env::var("PREVIEW_OUT").unwrap(), crate::texture::to_png(&out).unwrap()).unwrap();
    }

    #[test]
    fn framing_ignores_far_shell() {
        // 1000 points in a unit box + a huge sky shell of 10 points: frame the box, not the shell
        let mut p: Vec<[f32; 3]> = (0..1000).map(|i| [(i % 10) as f32 / 10.0, (i / 10 % 10) as f32 / 10.0, (i / 100) as f32 / 10.0]).collect();
        let (c0, r0) = super::framing(&p);
        p.extend((0..10).map(|i| [if i % 2 == 0 { 1e5 } else { -1e5 }, 0.0, 0.0]));
        let (c, r) = super::framing(&p);
        assert!(r < 2.0 && (c[0] - c0[0]).abs() < 0.2, "{c:?} {r}");
        assert!(r0 < 1.0);
    }

    #[test]
    fn sphere_is_valid() {
        let m = super::sphere();
        assert!(m.indices.iter().all(|&i| (i as usize) < m.positions.len()));
        assert_eq!(m.sections, vec![(0, 0, m.tris() as u32)]);
        assert_eq!(m.uvs.len(), m.positions.len());
    }
}
