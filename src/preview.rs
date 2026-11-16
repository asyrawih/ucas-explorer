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
            // ponytail: 20k tris per mesh keeps the painter preview responsive; export uses LOD0
            let b = crate::bp::build(&res, &asm, 20_000, &mut load);
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

/// Uploads slot images as egui textures (UI thread); oversized ones are skipped.
pub fn upload(ctx: &egui::Context, imgs: SlotImages) -> Vec<Option<egui::TextureHandle>> {
    let max = ctx.input(|i| i.max_texture_side);
    imgs.into_iter()
        .enumerate()
        .map(|(k, img)| {
            let img = img.filter(|i| i.size[0] <= max && i.size[1] <= max)?;
            Some(ctx.load_texture(format!("slot{k}"), img, egui::TextureOptions::LINEAR_REPEAT))
        })
        .collect()
}

pub fn mesh_view(ui: &mut egui::Ui, m: &crate::mesh::Mesh, tex: &[Option<egui::TextureHandle>], rot: &mut (f32, f32)) {
    let size = ui.available_size() - egui::vec2(4.0, 4.0);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(size.x, size.y.max(200.0)), egui::Sense::drag());
    let d = resp.drag_delta();
    rot.0 += d.x * 0.01;
    rot.1 = (rot.1 + d.y * 0.01).clamp(-1.5, 1.5);
    let (lo, hi) = m.positions.iter().fold(([f32::MAX; 3], [f32::MIN; 3]), |(lo, hi), p| {
        (std::array::from_fn(|k| lo[k].min(p[k])), std::array::from_fn(|k| hi[k].max(p[k])))
    });
    let c: [f32; 3] = std::array::from_fn(|k| (lo[k] + hi[k]) / 2.0);
    let r = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max).max(1e-3) / 2.0;
    let scale = rect.width().min(rect.height()) * 0.42 / r;
    let (sy, cy, sp, cp) = (rot.0.sin(), rot.0.cos(), rot.1.sin(), rot.1.cos());
    // UE is Z-up: yaw around Z, then pitch around the view's X axis; depth = Y'
    let xf: Vec<[f32; 3]> = m
        .positions
        .iter()
        .map(|p| {
            let (x, y, z) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
            let (x1, y1) = (x * cy - y * sy, x * sy + y * cy);
            [x1, y1 * cp - z * sp, y1 * sp + z * cp]
        })
        .collect();
    // material slot per triangle, from the LOD's sections
    let mut slot = vec![0u32; m.indices.len() / 3];
    for &(mat, first, n) in &m.sections {
        let t0 = first as usize / 3;
        for s in slot.iter_mut().skip(t0).take(n as usize) {
            *s = mat;
        }
    }
    let mut tris: Vec<(f32, usize, f32)> = m
        .indices
        .chunks_exact(3)
        .enumerate()
        .map(|(k, t)| {
            let [a, b, cc] = [xf[t[0] as usize], xf[t[1] as usize], xf[t[2] as usize]];
            let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [cc[0] - a[0], cc[1] - a[1], cc[2] - a[2]]);
            let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
            (a[1] + b[1] + cc[1], k, 0.35 + 0.65 * (n[1] / len).abs()) // light from the camera
        })
        .collect();
    tris.sort_by(|a, b| b.0.total_cmp(&a.0)); // far to near

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
    let center = rect.center();
    // one egui mesh per run of same-texture triangles keeps the global far-to-near order
    let mut cur = egui::Mesh::default();
    for (_, k, shade) in tris {
        let t = tex.get(slot[k] as usize).and_then(|t| t.as_ref());
        let id = t.map_or(egui::TextureId::default(), |t| t.id());
        if id != cur.texture_id && !cur.is_empty() {
            painter.add(egui::Shape::mesh(std::mem::take(&mut cur)));
        }
        cur.texture_id = id;
        let g = (shade * 255.0) as u8;
        let color = if t.is_some() { egui::Color32::from_gray(g) } else { egui::Color32::from_rgb((g as f32 * 0.86) as u8, (g as f32 * 0.86) as u8, (g as f32 * 0.9) as u8) };
        let base = cur.vertices.len() as u32;
        for &vi in &m.indices[k * 3..k * 3 + 3] {
            let p = xf[vi as usize];
            let uv = match (t, m.uvs.get(vi as usize)) {
                (Some(_), Some(uv)) => egui::pos2(uv[0], uv[1]),
                _ => egui::epaint::WHITE_UV,
            };
            cur.vertices.push(egui::epaint::Vertex { pos: center + egui::vec2(p[0], -p[2]) * scale, uv, color });
        }
        cur.add_triangle(base, base + 1, base + 2);
    }
    if !cur.is_empty() {
        painter.add(egui::Shape::mesh(cur));
    }
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

    #[test]
    fn sphere_is_valid() {
        let m = super::sphere();
        assert!(m.indices.iter().all(|&i| (i as usize) < m.positions.len()));
        assert_eq!(m.sections, vec![(0, 0, m.tris() as u32)]);
        assert_eq!(m.uvs.len(), m.positions.len());
    }
}
