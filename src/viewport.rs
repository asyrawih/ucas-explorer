//! GPU 3D viewport (wgpu, the backend eframe already runs on): perspective camera with Blender-like
//! navigation, depth buffer, textured flat shading. Renders into its own texture shown as an egui image.
// ponytail: one scene cached globally (only one 3D view is on screen at a time); re-renders every
// frame the view is shown, which is cheap next to building the scene.
use crate::mesh::Mesh;
use crate::preview::Tex;
use eframe::egui;
use eframe::egui_wgpu::{wgpu, RenderState};
use std::sync::{Mutex, OnceLock};
use wgpu::util::DeviceExt;

/// Orbit camera around `target` (offset from the framed center, UE cm), `dist` in framing radii.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32, // positive looks down
    pub dist: f32,
    pub target: [f32; 3],
}

impl Default for Camera {
    fn default() -> Self {
        Camera { yaw: 0.6, pitch: 0.35, dist: 2.2, target: [0.0; 3] }
    }
}

impl Camera {
    /// (right, up, forward) unit vectors in UE space (Z up), as the CPU preview's projection.
    pub fn basis(&self) -> [[f32; 3]; 3] {
        let (sy, cy, sp, cp) = (self.yaw.sin(), self.yaw.cos(), self.pitch.sin(), self.pitch.cos());
        [[cy, -sy, 0.0], [sp * sy, sp * cy, cp], [cp * sy, cp * cy, -sp]]
    }

    /// Mouse/keyboard navigation over `resp` (radius `r` sets pan/fly speed):
    /// LMB drag orbit, RMB/MMB/Shift+LMB drag pan, scroll/pinch zoom, WASD fly, Q/E down/up
    /// (Shift = 4x), F frame all. Returns true while a fly key is held (keep repainting).
    pub fn navigate(&mut self, ui: &egui::Ui, resp: &egui::Response, r: f32) -> bool {
        let [right, up, fwd] = self.basis();
        let shift = ui.input(|i| i.modifiers.shift);
        let d = resp.drag_delta();
        let pan = resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) || (shift && resp.dragged_by(egui::PointerButton::Primary));
        if pan {
            let k = self.dist * r * 0.0025;
            self.target = std::array::from_fn(|i| self.target[i] - (d.x * right[i] - d.y * up[i]) * k);
        } else if resp.dragged_by(egui::PointerButton::Primary) {
            self.yaw += d.x * 0.008;
            self.pitch = (self.pitch + d.y * 0.008).clamp(-1.55, 1.55);
        }
        let mut flying = false;
        if resp.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            ui.ctx().input_mut(|i| i.smooth_scroll_delta = egui::Vec2::ZERO); // don't also scroll the panel
            let f = (-scroll * 0.003).exp() / pinch;
            let nd = self.dist * f;
            if nd < 0.02 {
                // zoomed all the way in: keep moving forward through the scene (Blender dolly)
                let step = (self.dist - nd) * r;
                self.target = std::array::from_fn(|i| self.target[i] + fwd[i] * step);
            } else {
                self.dist = nd.min(200.0);
            }
            if !ui.ctx().egui_wants_keyboard_input() {
                use egui::Key;
                let (dt, keys) = ui.input(|i| (i.stable_dt.min(0.1), [Key::W, Key::S, Key::A, Key::D, Key::E, Key::Q, Key::F].map(|k| i.key_down(k))));
                if keys[6] {
                    *self = Camera { yaw: self.yaw, pitch: self.pitch, ..Default::default() };
                }
                let axis = |p: bool, n: bool| p as i32 as f32 - n as i32 as f32;
                let (z, x, y) = (axis(keys[0], keys[1]), axis(keys[3], keys[2]), axis(keys[4], keys[5]));
                if z != 0.0 || x != 0.0 || y != 0.0 {
                    let speed = r * 0.5 * dt * if shift { 4.0 } else { 1.0 };
                    self.target = std::array::from_fn(|i| self.target[i] + (fwd[i] * z + right[i] * x + [0.0, 0.0, 1.0][i] * y) * speed);
                    flying = true;
                }
            }
        }
        flying
    }

    /// World -> clip matrix (column-major, wgpu depth 0..1) for framing `(center, radius)`.
    pub fn view_proj(&self, center: [f32; 3], r: f32, aspect: f32) -> ([f32; 16], [f32; 3]) {
        let [rt, up, fw] = self.basis();
        let t: [f32; 3] = std::array::from_fn(|i| center[i] + self.target[i]);
        let eye: [f32; 3] = std::array::from_fn(|i| t[i] - fw[i] * self.dist * r);
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let f = 1.0 / (30f32.to_radians()).tan(); // 60 degree vertical FOV
        let (near, far) = (r * 2e-4, r * (self.dist + 60.0));
        let (a, b) = (far / (far - near), -far * near / (far - near));
        let row = |v: [f32; 3], s: f32| [v[0] * s, v[1] * s, v[2] * s, -dot(v, eye) * s];
        let r0 = row(rt, f / aspect);
        let r1 = row(up, f);
        let r3 = row(fw, 1.0);
        let r2 = [r3[0] * a, r3[1] * a, r3[2] * a, r3[3] * a + b];
        let rows = [r0, r1, r2, r3];
        (std::array::from_fn(|k| rows[k % 4][k / 4]), eye)
    }
}

/// Direction toward the sun (UE space, Z up).
const SUN: [f32; 3] = [0.4472, 0.2981, 0.8433];
/// Shadow map resolution (one map over the framed region).
// ponytail: single shadow map around the framing; cascades around the camera if walking through big
// levels shows blocky shadows.
const SHADOW: u32 = 4096;

/// Orthographic world -> shadow-map matrix (column-major, depth 0..1) covering a sphere of radius
/// `ext` around `c`, looking along -SUN.
fn light_proj(c: [f32; 3], ext: f32) -> [f32; 16] {
    let f = SUN.map(|x| -x);
    let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let norm = |v: [f32; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        v.map(|x| x / l)
    };
    let rt = norm(cross([0.0, 0.0, 1.0], f));
    let up = cross(f, rt);
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let row = |v: [f32; 3], s: f32, o: f32| [v[0] * s, v[1] * s, v[2] * s, -dot(v, c) * s + o];
    let rows = [row(rt, 1.0 / ext, 0.0), row(up, 1.0 / ext, 0.0), row(f, 0.5 / ext, 0.5), [0.0, 0.0, 0.0, 1.0]];
    std::array::from_fn(|k| rows[k % 4][k / 4])
}

struct Gpu {
    rs: RenderState,
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    tex_layout: wgpu::BindGroupLayout,
    uni_layout: wgpu::BindGroupLayout,
    shadow_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
}

struct Target {
    size: (u32, u32),
    msaa: wgpu::TextureView,
    color: wgpu::TextureView, // resolved, shown by egui
    depth: wgpu::TextureView,
    id: egui::TextureId,
}

struct Scene {
    geo_key: (usize, usize),
    tex_key: usize,
    pos: wgpu::Buffer,
    uv: wgpu::Buffer,
    nrm: wgpu::Buffer,
    idx: wgpu::Buffer,
    draws: Vec<(u32, u32, u32)>, // (slot, first index, index count)
    binds: Vec<wgpu::BindGroup>, // per slot; last = untextured fallback
    uni: wgpu::Buffer,
    uni_bind: wgpu::BindGroup,
    shadow_bind: wgpu::BindGroup,
    shadow: wgpu::TextureView,
    /// framing the shadow map was rendered for (geometry and sun are static: rendered once)
    shadow_frame: Option<([f32; 3], f32)>,
    target: Option<Target>,
}

static GPU: OnceLock<Gpu> = OnceLock::new();
static SCENE: Mutex<Option<Scene>> = Mutex::new(None);

const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MSAA: u32 = 4;

const SHADER: &str = r#"
struct U { vp: mat4x4<f32>, light: mat4x4<f32>, eye: vec4<f32>, sun: vec4<f32> }; // sun.w = shadow texel (world)
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_cmp: sampler_comparison;
@group(1) @binding(0) var t: texture_2d<f32>;
@group(1) @binding(1) var s: sampler;
struct O { @builtin(position) p: vec4<f32>, @location(0) w: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) n: vec3<f32> };
@vertex fn vs(@location(0) p: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) n: vec3<f32>) -> O {
    var o: O;
    o.p = u.vp * vec4<f32>(p, 1.0);
    o.w = p;
    o.uv = uv;
    o.n = n;
    return o;
}
// fraction of the sun reaching `w`: 3x3 PCF over the shadow map (outside it: lit)
fn lit(w: vec3<f32>) -> f32 {
    let l = u.light * vec4<f32>(w, 1.0);
    let uv = l.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || l.z > 1.0) { return 1.0; }
    let texel = 1.0 / f32(textureDimensions(shadow_map).x);
    var sum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            sum += textureSampleCompareLevel(shadow_map, shadow_cmp, uv + vec2<f32>(f32(x), f32(y)) * texel, l.z);
        }
    }
    return sum / 9.0;
}
@fragment fn fs(i: O) -> @location(0) vec4<f32> {
    let albedo = textureSample(t, s, i.uv).rgb;
    let v = normalize(u.eye.xyz - i.w);
    // smooth vertex normal; face normal where the mesh has none
    var n = cross(dpdx(i.w), dpdy(i.w));
    if (dot(i.n, i.n) > 0.25) { n = i.n; }
    n = normalize(n);
    if (dot(n, v) < 0.0) { n = -n; } // both faces are drawn: light the side we see
    let sun = vec3<f32>(1.0, 0.95, 0.85) * max(dot(n, u.sun.xyz), 0.0) * lit(i.w + n * u.sun.w * 1.5);
    let sky = mix(vec3<f32>(0.10, 0.09, 0.08), vec3<f32>(0.28, 0.32, 0.40), n.z * 0.5 + 0.5); // hemisphere ambient
    let head = vec3<f32>(0.25) * max(dot(n, v), 0.0); // camera fill, so interiors aren't black
    return vec4<f32>(albedo * (sun + sky + head), 1.0);
}
"#;

const SHADOW_SHADER: &str = r#"
struct U { vp: mat4x4<f32>, light: mat4x4<f32>, eye: vec4<f32>, sun: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;
@vertex fn vs(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
    return u.light * vec4<f32>(p, 1.0);
}
"#;

/// Called once at startup with eframe's wgpu state; without it the CPU preview is used.
pub fn init(rs: RenderState) {
    let d = &rs.device;
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("viewport"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let uniform = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let shadow_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[uniform] });
    let uni_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            uniform,
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison), count: None },
        ],
    });
    let tex_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
        ],
    });
    let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&uni_layout), Some(&tex_layout)], immediate_size: 0 });
    let pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewport"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            buffers: &[
                Some(wgpu::VertexBufferLayout { array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3] }),
                Some(wgpu::VertexBufferLayout { array_stride: 8, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![1 => Float32x2] }),
                Some(wgpu::VertexBufferLayout { array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![2 => Float32x3] }),
            ],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() }, // both faces, like the CPU preview
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: MSAA, ..Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState { format: COLOR, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    });
    let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: 8,
        ..Default::default()
    });
    let smodule = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("shadow"), source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()) });
    let slayout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&shadow_layout)], immediate_size: 0 });
    let shadow_pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow"),
        layout: Some(&slayout),
        vertex: wgpu::VertexState {
            module: &smodule,
            entry_point: Some("vs"),
            buffers: &[Some(wgpu::VertexBufferLayout { array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3] })],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 }, // against shadow acne
        }),
        multisample: Default::default(),
        fragment: None,
        multiview_mask: None,
        cache: None,
    });
    let shadow_sampler = d.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });
    let _ = GPU.set(Gpu { rs, pipeline, shadow_pipeline, tex_layout, uni_layout, shadow_layout, sampler, shadow_sampler });
}

/// RGBA8 mip chain (box filter), most detailed first, starting at the first level that fits `max`.
fn mips(img: &egui::ColorImage, max: u32) -> (u32, u32, Vec<Vec<u8>>) {
    let (mut w, mut h) = (img.size[0] as u32, img.size[1] as u32);
    let mut cur: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
    let (mut out, mut first) = (vec![], (1, 1));
    loop {
        if w <= max && h <= max {
            if out.is_empty() {
                first = (w, h);
            }
            out.push(cur.clone());
        }
        if w == 1 && h == 1 {
            break;
        }
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let px = |x: u32, y: u32, k: usize| cur[((y.min(h - 1) * w + x.min(w - 1)) * 4) as usize + k] as u32;
        cur = (0..nh)
            .flat_map(|y| (0..nw).map(move |x| (x, y)))
            .flat_map(|(x, y)| {
                let (sx, sy) = (x * w / nw, y * h / nh);
                (0..4).map(move |k| ((px(sx, sy, k) + px(sx + 1, sy, k) + px(sx, sy + 1, k) + px(sx + 1, sy + 1, k)) / 4) as u8).collect::<Vec<_>>()
            })
            .collect();
        (w, h) = (nw, nh);
    }
    (first.0, first.1, out)
}

fn upload_tex(g: &Gpu, img: Option<&egui::ColorImage>) -> wgpu::BindGroup {
    let (d, q) = (&g.rs.device, &g.rs.queue);
    let fallback = egui::ColorImage::new([1, 1], vec![egui::Color32::from_rgb(219, 219, 230)]);
    let (w, h, levels) = mips(img.unwrap_or(&fallback), d.limits().max_texture_dimension_2d);
    let tex = d.create_texture_with_data(
        q,
        &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &levels.concat(),
    );
    let view = tex.create_view(&Default::default());
    d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g.tex_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&g.sampler) }],
    })
}

fn tex_key(tex: &[Tex]) -> usize {
    tex.iter().enumerate().map(|(i, t)| t.as_ref().map_or(0, |a| std::sync::Arc::as_ptr(a) as usize).wrapping_mul(i + 1)).fold(tex.len(), usize::wrapping_add)
}

/// Builds or updates the cached scene for `m`/`tex`. None if the mesh doesn't fit the GPU's buffers.
fn scene<'a>(g: &Gpu, slot: &'a mut Option<Scene>, m: &Mesh, tex: &[Tex]) -> Option<&'a mut Scene> {
    let d = &g.rs.device;
    let geo_key = (m.positions.as_ptr() as usize, m.indices.len());
    if slot.as_ref().is_none_or(|s| s.geo_key != geo_key) {
        let max = d.limits().max_buffer_size;
        if (m.positions.len() * 12) as u64 > max || (m.indices.len() * 4) as u64 > max {
            return None;
        }
        let bytes = |v: &mut dyn Iterator<Item = f32>| v.flat_map(f32::to_le_bytes).collect::<Vec<u8>>();
        let pos = bytes(&mut m.positions.iter().flatten().copied());
        let uv = bytes(&mut (0..m.positions.len()).flat_map(|i| m.uvs.get(i).copied().unwrap_or([0.0; 2])));
        let nrm = bytes(&mut (0..m.positions.len()).flat_map(|i| m.normals.get(i).copied().unwrap_or([0.0; 3])));
        let idx: Vec<u8> = m.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let buf = |data: &[u8], usage| d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: data, usage });
        let uni = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: 160, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let shadow = d
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("shadow"),
                size: wgpu::Extent3d { width: SHADOW, height: SHADOW, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let uni_bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &g.uni_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uni.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&g.shadow_sampler) },
            ],
        });
        let shadow_bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &g.shadow_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: uni.as_entire_binding() }] });
        let target = slot.take().and_then(|s| s.target);
        *slot = Some(Scene {
            geo_key,
            tex_key: usize::MAX,
            pos: buf(&pos, wgpu::BufferUsages::VERTEX),
            uv: buf(&uv, wgpu::BufferUsages::VERTEX),
            nrm: buf(&nrm, wgpu::BufferUsages::VERTEX),
            idx: buf(&idx, wgpu::BufferUsages::INDEX),
            draws: m.sections.iter().map(|&(mat, first, n)| (mat, first, n * 3)).collect(),
            binds: vec![],
            uni,
            uni_bind,
            shadow_bind,
            shadow,
            shadow_frame: None,
            target,
        });
    }
    let s = slot.as_mut()?;
    let tk = tex_key(tex);
    if s.tex_key != tk {
        s.binds = tex.iter().map(|t| upload_tex(g, t.as_deref())).chain(std::iter::once(upload_tex(g, None))).collect();
        s.tex_key = tk;
    }
    Some(s)
}

/// Draws `m` into `rect` with camera `cam` framed on `(center, radius)`. False: no GPU (or the mesh
/// is too big for its buffers); the caller falls back to the CPU preview.
pub fn draw(ui: &egui::Ui, rect: egui::Rect, m: &Mesh, tex: &[Tex], cam: &Camera, frame: ([f32; 3], f32)) -> bool {
    let Some(g) = GPU.get() else { return false };
    let mut lock = SCENE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = scene(g, &mut lock, m, tex) else { return false };
    let (d, q) = (&g.rs.device, &g.rs.queue);
    let ppp = ui.ctx().pixels_per_point();
    let max = d.limits().max_texture_dimension_2d;
    let (w, h) = (((rect.width() * ppp) as u32).clamp(16, max), ((rect.height() * ppp) as u32).clamp(16, max));
    if s.target.as_ref().is_none_or(|t| t.size != (w, h)) {
        let mk = |format, usage, samples| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
            .create_view(&Default::default())
        };
        let msaa = mk(COLOR, wgpu::TextureUsages::RENDER_ATTACHMENT, MSAA);
        let color = mk(COLOR, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, 1);
        let depth = mk(DEPTH, wgpu::TextureUsages::RENDER_ATTACHMENT, MSAA);
        let mut r = g.rs.renderer.write();
        let id = match s.target.take() {
            Some(t) => {
                r.update_egui_texture_from_wgpu_texture(d, &color, wgpu::FilterMode::Linear, t.id);
                t.id
            }
            None => r.register_native_texture(d, &color, wgpu::FilterMode::Linear),
        };
        s.target = Some(Target { size: (w, h), msaa, color, depth, id });
    }
    let (vp, eye) = cam.view_proj(frame.0, frame.1, w as f32 / h as f32);
    let ext = frame.1 * 1.8;
    let light = light_proj(frame.0, ext);
    let uni: Vec<u8> = vp.iter().chain(&light).chain(&eye).chain(&[0.0]).chain(&SUN).chain(&[2.0 * ext / SHADOW as f32]).flat_map(|f| f.to_le_bytes()).collect();
    q.write_buffer(&s.uni, 0, &uni);
    let t = s.target.as_ref().unwrap();
    let bg = egui::Rgba::from(ui.visuals().extreme_bg_color);
    let mut enc = d.create_command_encoder(&Default::default());
    let fallback = s.binds.len() - 1;
    if s.shadow_frame != Some(frame) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &s.shadow,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&g.shadow_pipeline);
        pass.set_bind_group(0, &s.shadow_bind, &[]);
        pass.set_vertex_buffer(0, s.pos.slice(..));
        pass.set_index_buffer(s.idx.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..s.draws.iter().map(|d| d.1 + d.2).max().unwrap_or(0), 0, 0..1);
        drop(pass);
        s.shadow_frame = Some(frame);
    }
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewport"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &t.msaa,
                resolve_target: Some(&t.color),
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: bg.r() as f64, g: bg.g() as f64, b: bg.b() as f64, a: 1.0 }), store: wgpu::StoreOp::Discard },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &t.depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&g.pipeline);
        pass.set_bind_group(0, &s.uni_bind, &[]);
        pass.set_vertex_buffer(0, s.pos.slice(..));
        pass.set_vertex_buffer(1, s.uv.slice(..));
        pass.set_vertex_buffer(2, s.nrm.slice(..));
        pass.set_index_buffer(s.idx.slice(..), wgpu::IndexFormat::Uint32);
        for &(slot, first, count) in &s.draws {
            let b = if (slot as usize) < fallback { slot as usize } else { fallback };
            pass.set_bind_group(1, &s.binds[b], &[]);
            pass.draw_indexed(first..first + count, 0, 0..1);
        }
    }
    q.submit([enc.finish()]);
    ui.painter_at(rect).image(t.id, rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
    true
}

#[cfg(test)]
mod tests {
    use super::Camera;

    #[test]
    fn view_proj_puts_target_center_and_depth_in_range() {
        let c = Camera::default();
        let (m, eye) = c.view_proj([100.0, 50.0, 10.0], 100.0, 1.5);
        let clip = |p: [f32; 3]| -> [f32; 4] { std::array::from_fn(|r| m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]) };
        let t = clip([100.0, 50.0, 10.0]); // the orbit target lands mid-screen, between the depth planes
        assert!((t[0] / t[3]).abs() < 1e-4 && (t[1] / t[3]).abs() < 1e-4, "{t:?}");
        assert!((0.0..1.0).contains(&(t[2] / t[3])));
        let dist = ((eye[0] - 100.0).powi(2) + (eye[1] - 50.0).powi(2) + (eye[2] - 10.0).powi(2)).sqrt();
        assert!((dist - 220.0).abs() < 1e-2);
        // world up (+Z) projects upward on screen
        let u = clip([100.0, 50.0, 60.0]);
        assert!(u[1] / u[3] > 0.0);
    }

    #[test]
    fn light_proj_maps_framed_sphere_into_shadow_map() {
        let (c, ext) = ([1000.0, -500.0, 200.0], 300.0);
        let m = super::light_proj(c, ext);
        let tf = |p: [f32; 3]| -> [f32; 3] { std::array::from_fn(|r| m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]) };
        let mid = tf(c);
        assert!(mid[0].abs() < 1e-4 && mid[1].abs() < 1e-4 && (mid[2] - 0.5).abs() < 1e-4, "{mid:?}");
        // a point toward the sun is nearer the light (smaller depth) than one away from it
        let toward = tf(std::array::from_fn(|k| c[k] + super::SUN[k] * 250.0));
        let away = tf(std::array::from_fn(|k| c[k] - super::SUN[k] * 250.0));
        assert!(toward[2] < 0.5 && away[2] > 0.5 && toward[2] >= 0.0 && away[2] <= 1.0);
    }

    #[test]
    fn mips_halve_down_to_one_pixel_and_respect_max() {
        let img = eframe::egui::ColorImage::new([8, 4], vec![eframe::egui::Color32::WHITE; 32]);
        let (w, h, l) = super::mips(&img, 4);
        assert_eq!((w, h, l.len()), (4, 2, 3)); // 4x2, 2x1, 1x1
        assert_eq!(l.iter().map(Vec::len).collect::<Vec<_>>(), [32, 8, 4]);
        assert!(l[2].iter().all(|&b| b == 255));
    }
}
