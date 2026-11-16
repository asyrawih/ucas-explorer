//! Cooked USkeletalMesh -> geometry + skeleton + skin weights -> binary glTF, without .usmap.
// ponytail: properties are unversioned, so like mesh.rs we don't walk USkeletalMesh::Serialize. We scan for
// three independently validated signatures and stitch them by consistency checks:
//  1. FReferenceSkeleton: TArray<FMeshBoneInfo{FName, i32 parent}> (first parent -1, others in [0, i))
//     followed by an equal-length TArray<FTransform> of doubles (unit quaternions).
//  2. FSkeletalMeshLODRenderData streamed data: FMultiSizeIndexContainer, FPositionVertexBuffer,
//     FStaticMeshVertexBuffer, FSkinWeightVertexBuffer (found via the position-buffer signature).
//  3. TArray<FSkelMeshRenderSection> whose triangle/vertex totals match that LOD (gives BoneMap + materials).
// Layout verified on Palia (UE5.3+). Upgrade path: real FSkeletalMeshRenderData walk once property sizes are known.
use crate::Res;

pub struct Bone {
    pub name: String,
    pub parent: i32,
    pub rotation: [f32; 4], // xyzw
    pub translation: [f32; 3],
    pub scale: [f32; 3],
}

pub struct SkelMesh {
    pub bones: Vec<Bone>, // reference skeleton (bind pose, local transforms, UE space)
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub joints: Vec<[u16; 4]>, // indices into `bones` (top 4 influences)
    pub weights: Vec<[f32; 4]>, // normalized
    pub sections: Vec<(u32, u32, u32)>, // (material index, first index, num triangles)
}

struct Rd<'a>(&'a [u8], usize);
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.0.get(self.1..self.1.checked_add(n)?)?;
        self.1 += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn bool(&mut self) -> Option<bool> {
        match self.u32()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
    /// TBulkArray: i32 element size, i32 count, data
    fn bulk(&mut self) -> Option<(usize, usize, &'a [u8])> {
        let (es, n) = (self.u32()? as usize, self.u32()? as usize);
        Some((es, n, self.take(es.checked_mul(n)?)?))
    }
}

fn f16(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let mag = match e {
        0 => m as f32 / 16_777_216.0,
        31 => f32::INFINITY,
        _ => f32::from_bits(((e + 112) << 23) | (m << 13)),
    };
    if s == 1 { -mag } else { mag }
}

/// FReferenceSkeleton at `at`: bone infos + FTransform (double) pose.
fn parse_skeleton(d: &[u8], at: usize, names: &[String]) -> Option<Vec<Bone>> {
    let mut r = Rd(d, at);
    let n = r.u32()? as usize;
    if !(1..=4096).contains(&n) {
        return None;
    }
    let mut info = Vec::with_capacity(n);
    for i in 0..n {
        let (ni, num, parent) = (r.u32()?, r.u32()?, r.u32()? as i32);
        let base = names.get(ni as usize)?;
        if num > 1000 || if i == 0 { parent != -1 } else { parent < 0 || parent >= i as i32 } {
            return None;
        }
        info.push((if num == 0 { base.clone() } else { format!("{base}_{}", num - 1) }, parent));
    }
    if r.u32()? as usize != n {
        return None;
    }
    let mut bones = Vec::with_capacity(n);
    for (name, parent) in info {
        let mut v = [0f64; 10];
        for x in &mut v {
            *x = r.f64()?;
        }
        let ql = v[..4].iter().map(|x| x * x).sum::<f64>();
        if !v.iter().all(|x| x.is_finite() && x.abs() < 1e7) || (ql - 1.0).abs() > 1e-3 {
            return None;
        }
        let f = |a: &[f64]| a.iter().map(|&x| x as f32).collect::<Vec<_>>();
        bones.push(Bone {
            name,
            parent,
            rotation: f(&v[..4]).try_into().ok()?,
            translation: f(&v[4..7]).try_into().ok()?,
            scale: f(&v[7..]).try_into().ok()?,
        });
    }
    Some(bones)
}

/// First FReferenceSkeleton found in `d` (SkeletalMesh or Skeleton export data).
pub fn find_reference_skeleton(d: &[u8], names: &[String]) -> Option<Vec<Bone>> {
    (0..d.len().saturating_sub(16)).find_map(|i| parse_skeleton(d, i, names))
}

struct Lod {
    idx_at: usize,
    indices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    infl: Vec<Vec<(u16, f32)>>, // per vertex: (section-local bone index, weight)
}

/// FSkeletalMeshLODRenderData streamed data around the FPositionVertexBuffer at `at`.
fn parse_lod(d: &[u8], at: usize) -> Option<Lod> {
    let mut r = Rd(d, at);
    let (stride, n) = (r.u32()?, r.u32()? as usize);
    let (es, n2, pos) = r.bulk()?;
    if stride != 12 || es != 12 || n2 != n || n == 0 {
        return None;
    }
    let positions: Vec<[f32; 3]> = pos
        .chunks_exact(12)
        .map(|c| std::array::from_fn(|k| f32::from_le_bytes(c[k * 4..k * 4 + 4].try_into().unwrap())))
        .collect();
    if positions.iter().flatten().any(|v| !v.is_finite() || v.abs() > 1e7) {
        return None;
    }

    // FStaticMeshVertexBuffer: strip flags (2), NumTexCoords, NumVertices, bFullPrecisionUVs, bHighPrecisionTangents
    r.take(2)?;
    let (ntc, nv) = (r.u32()? as usize, r.u32()? as usize);
    let (full_uv, hi_tan) = (r.bool()?, r.bool()?);
    if nv != n || !(1..=8).contains(&ntc) {
        return None;
    }
    let (tes, tn, tan) = r.bulk()?;
    let (ues, un, uv) = r.bulk()?;
    if tn != n || tes != if hi_tan { 16 } else { 8 } || un != n * ntc || ues != if full_uv { 8 } else { 4 } {
        return None;
    }
    let normals = tan
        .chunks_exact(tes)
        .map(|c| {
            let h = &c[tes / 2..]; // (TangentX, TangentZ=normal)
            let v: [f32; 3] = if hi_tan {
                std::array::from_fn(|k| i16::from_le_bytes([h[k * 2], h[k * 2 + 1]]) as f32 / 32767.0)
            } else {
                std::array::from_fn(|k| h[k] as i8 as f32 / 127.0)
            };
            let l = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if l > 1e-6 { v.map(|x| x / l) } else { [0.0, 0.0, 1.0] }
        })
        .collect();
    let uvs = uv
        .chunks_exact(ues * ntc)
        .map(|c| {
            if full_uv {
                [f32::from_le_bytes(c[0..4].try_into().unwrap()), f32::from_le_bytes(c[4..8].try_into().unwrap())]
            } else {
                [f16(u16::from_le_bytes([c[0], c[1]])), f16(u16::from_le_bytes([c[2], c[3]]))]
            }
        })
        .collect();

    // FSkinWeightVertexBuffer (UE5): strip(2), bVariableBonesPerVertex, MaxBoneInfluences, NumBones, NumVertices,
    // bUse16BitBoneIndex, bUse16BitBoneWeight, bulk<u8> data, then lookup: strip(2), NumVertices, bulk<u32>
    r.take(2)?;
    let var = r.bool()?;
    let (maxi, _nbones, snv) = (r.u32()? as usize, r.u32()?, r.u32()? as usize);
    let (b16i, b16w) = (r.bool()?, r.bool()?);
    let (wes, _, wdata) = r.bulk()?;
    r.take(2)?;
    let _ = r.u32()?;
    let (les, ln, lookup) = r.bulk()?;
    if snv != n || wes != 1 || !(1..=12).contains(&maxi) {
        return None;
    }
    let (isz, wsz) = (if b16i { 2 } else { 1 }, if b16w { 2 } else { 1 });
    let get = |o: usize, sz: usize| -> Option<u16> {
        let b = wdata.get(o..o + sz)?;
        Some(if sz == 2 { u16::from_le_bytes([b[0], b[1]]) } else { b[0] as u16 })
    };
    let wmax = if b16w { 65535.0 } else { 255.0 };
    let mut infl = Vec::with_capacity(n);
    for v in 0..n {
        let (off, cnt) = if var {
            if les != 4 || ln != n {
                return None;
            }
            let l = u32::from_le_bytes(lookup[v * 4..v * 4 + 4].try_into().unwrap()) as usize;
            (l >> 8, l & 0xff)
        } else {
            (v * maxi * (isz + wsz), maxi)
        };
        let mut w = Vec::with_capacity(cnt);
        for k in 0..cnt {
            let (b, wt) = (get(off + k * isz, isz)?, get(off + cnt * isz + k * wsz, wsz)?);
            if wt > 0 {
                w.push((b, wt as f32 / wmax));
            }
        }
        infl.push(w);
    }

    // FMultiSizeIndexContainer precedes the position buffer: u8 DataTypeSize, bulk<u16|u32>. Search backward for
    // the start whose length ends exactly at `at`.
    // ponytail: O(index bytes) backward search; fine for one hit per LOD.
    let u = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap()) as usize;
    let idx_at = (at.saturating_sub(4 * 4_000_000 + 9)..at.checked_sub(9)?)
        .rev()
        .find(|&p| matches!(d[p], 2 | 4) && u(p + 1) == d[p] as usize && (d[p] as usize).checked_mul(u(p + 5)).and_then(|s| s.checked_add(p + 9)) == Some(at))?;
    let raw = &d[idx_at + 9..at];
    let indices: Vec<u32> = if d[idx_at] == 4 {
        raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    } else {
        raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect()
    };
    if indices.is_empty() || indices.len() % 3 != 0 || indices.iter().any(|&i| i as usize >= n) {
        return None;
    }
    Some(Lod { idx_at, indices, positions, normals, uvs, infl })
}

struct Section {
    material: u32,
    base_index: u32,
    tris: u32,
    base_vertex: u32,
    verts: u32,
    bone_map: Vec<u16>,
}

/// TArray<FSkelMeshRenderSection> at `at`.
fn parse_sections(d: &[u8], at: usize) -> Option<Vec<Section>> {
    let mut r = Rd(d, at);
    let n = r.u32()?;
    if !(1..=256).contains(&n) {
        return None;
    }
    let mut out = Vec::new();
    for _ in 0..n {
        let (_, class_strip) = (r.u8()?, r.u8()?);
        let material = r.u16()? as i16;
        let (base_index, tris) = (r.u32()?, r.u32()?);
        r.bool()?; // bRecomputeTangent
        if r.u8()? > 4 {
            return None; // RecomputeTangentsVertexMaskChannel
        }
        r.bool()?; // bCastShadow
        r.bool()?; // bVisibleInRayTracing
        let base_vertex = r.u32()?;
        for _ in 0..r.u32()? {
            let k = r.u32()? as usize; // ClothMappingDataLODs: TArray<TArray<FMeshToMeshVertData (64 bytes)>>
            r.take(k.checked_mul(64)?)?;
        }
        let nb = r.u32()? as usize;
        let bone_map = r.take(nb.checked_mul(2)?)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let verts = r.u32()?;
        let max_infl = r.u32()? & 0x7fff_ffff;
        r.take(2 + 20)?; // CorrespondClothAssetIndex, FClothingSectionData
        if class_strip & 1 == 0 {
            r.bulk()?; // DuplicatedVertices: DupVertData, DupVertIndexData
            r.bulk()?;
        }
        r.bool()?; // bDisabled
        if material < 0 || tris == 0 || !(1..=12).contains(&max_infl) {
            return None;
        }
        out.push(Section { material: material as u32, base_index, tris, base_vertex, verts, bone_map });
    }
    Some(out)
}

fn scan_lods(d: &[u8], out: &mut Vec<Lod>) {
    let u = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
    let mut i = 0;
    while i + 16 <= d.len() {
        if u(i) == 12 && u(i + 8) == 12 && u(i + 4) == u(i + 12) && u(i + 4) > 0 {
            if let Some(m) = parse_lod(d, i) {
                i += 16 + m.positions.len() * 12;
                out.push(m);
                continue;
            }
        }
        i += 1;
    }
}

/// Most detailed LOD fully decodable from the SkeletalMesh export (and its .ubulk, if any).
pub fn decode_skeletal_mesh(pkg: &crate::zen::ZenPackage, export: usize, ubulk: Option<&[u8]>) -> Res<SkelMesh> {
    if export >= pkg.exports.len() {
        return Err("skeletal: bad export index".into());
    }
    let d = pkg.export_data(export);
    let bones = find_reference_skeleton(d, &pkg.names).ok_or("skeletal: reference skeleton not found")?;

    let mut lods = Vec::new();
    scan_lods(d, &mut lods);
    let inline = lods.len();
    if let Some(b) = ubulk {
        scan_lods(b, &mut lods);
    }
    if lods.is_empty() {
        return Err("skeletal: no LOD vertex buffers found".into());
    }
    // section arrays live in the export (LOD headers); pair each LOD with the nearest matching one before it
    let candidates: Vec<(usize, Vec<Section>)> = (0..d.len()).filter_map(|i| Some((i, parse_sections(d, i)?))).collect();
    let mut order: Vec<usize> = (0..lods.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(lods[i].indices.len()));
    for li in order {
        let lod = &lods[li];
        let fits = |s: &[Section]| {
            s.iter().map(|s| s.tris as usize * 3).sum::<usize>() == lod.indices.len()
                && s.iter().map(|s| s.verts as usize).sum::<usize>() == lod.positions.len()
                && s.iter().all(|s| {
                    (s.base_index as usize + s.tris as usize * 3) <= lod.indices.len()
                        && (s.base_vertex as usize + s.verts as usize) <= lod.positions.len()
                        && s.bone_map.iter().all(|&b| (b as usize) < bones.len())
                })
        };
        let near = |&&(o, _): &&(usize, Vec<Section>)| if li < inline && o < lod.idx_at { lod.idx_at - o } else { usize::MAX };
        let Some((_, secs)) = candidates.iter().filter(|(_, s)| fits(s)).min_by_key(near) else { continue };
        let mut joints = vec![[0u16; 4]; lod.positions.len()];
        let mut weights = vec![[0f32; 4]; lod.positions.len()];
        let mut bad = false;
        for s in secs {
            for v in s.base_vertex as usize..(s.base_vertex + s.verts) as usize {
                let mut w = lod.infl[v].clone();
                w.sort_by(|a, b| b.1.total_cmp(&a.1));
                w.truncate(4);
                let sum: f32 = w.iter().map(|x| x.1).sum();
                for (k, &(b, wt)) in w.iter().enumerate() {
                    let Some(&g) = s.bone_map.get(b as usize) else {
                        bad = true;
                        break;
                    };
                    joints[v][k] = g;
                    weights[v][k] = wt / sum;
                }
                if w.is_empty() {
                    weights[v][0] = 1.0; // unweighted -> root
                    joints[v][0] = s.bone_map.first().copied().unwrap_or(0);
                }
            }
        }
        if bad {
            continue;
        }
        let lod = lods.swap_remove(li);
        return Ok(SkelMesh {
            bones,
            positions: lod.positions,
            normals: lod.normals,
            uvs: lod.uvs,
            indices: lod.indices,
            joints,
            weights,
            sections: secs.iter().map(|s| (s.material, s.base_index, s.tris)).collect(),
        });
    }
    Err("skeletal: no LOD with matching render sections".into())
}

// --- glTF ---

type M4 = [f32; 16]; // column-major

fn trs(t: [f32; 3], q: [f32; 4], s: [f32; 3]) -> M4 {
    let [x, y, z, w] = q;
    let r = [
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w),
        2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w),
        2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y),
    ];
    let mut m = [0.0; 16];
    for c in 0..3 {
        for row in 0..3 {
            m[c * 4 + row] = r[c * 3 + row] * s[c];
        }
    }
    m[12..15].copy_from_slice(&t);
    m[15] = 1.0;
    m
}

fn mul(a: &M4, b: &M4) -> M4 {
    std::array::from_fn(|i| (0..4).map(|k| a[k * 4 + i % 4] * b[(i / 4) * 4 + k]).sum())
}

/// Inverse of an affine matrix (last row 0,0,0,1).
fn inv_affine(m: &M4) -> M4 {
    let a = |r: usize, c: usize| m[c * 4 + r] as f64;
    let det = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
        + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
    let d = if det.abs() < 1e-12 { 1.0 } else { 1.0 / det };
    let mut inv = [[0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let (r1, r2, c1, c2) = ((c + 1) % 3, (c + 2) % 3, (r + 1) % 3, (r + 2) % 3);
            inv[r][c] = (a(r1, c1) * a(r2, c2) - a(r1, c2) * a(r2, c1)) * d;
        }
    }
    let mut o = [0f32; 16];
    for r in 0..3 {
        for c in 0..3 {
            o[c * 4 + r] = inv[r][c] as f32;
        }
        o[12 + r] = -(0..3).map(|k| inv[r][k] * a(k, 3)).sum::<f64>() as f32;
    }
    o[15] = 1.0;
    o
}

/// UE (Z-up, left-handed, cm) -> glTF (Y-up, right-handed, m): swap Y/Z, /100. A mirror, so quaternion axes
/// flip sign: (x,y,z,w) -> (-x,-z,-y,w). It also flips UE's clockwise winding to glTF's counter-clockwise.
fn conv_bone(b: &Bone) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let (t, q, s) = (b.translation, b.rotation, b.scale);
    ([t[0] / 100.0, t[2] / 100.0, t[1] / 100.0], [-q[0], -q[2], -q[1], q[3]], [s[0], s[2], s[1]])
}

/// `textures[material index]` = optional base color PNG, embedded in the .glb.
pub fn to_glb(m: &SkelMesh, name: &str, textures: &[Option<Vec<u8>>]) -> Res<Vec<u8>> {
    write_glb(m, name, textures, None)
}

/// Like `to_glb`, plus one glTF animation (`anim.tracks[..].bone` index `m.bones`). With an empty mesh
/// (no positions) only the skeleton (a skin without a mesh) and the animation are written.
pub fn to_glb_with_animation(m: &SkelMesh, name: &str, textures: &[Option<Vec<u8>>], anim: &crate::anim::Anim) -> Res<Vec<u8>> {
    write_glb(m, name, textures, Some(anim))
}

fn write_glb(m: &SkelMesh, name: &str, textures: &[Option<Vec<u8>>], anim: Option<&crate::anim::Anim>) -> Res<Vec<u8>> {
    let (nv, nb) = (m.positions.len(), m.bones.len());
    let has_mesh = nv > 0;
    if nb == 0 || m.normals.len() != nv || m.uvs.len() != nv || m.joints.len() != nv || m.weights.len() != nv || (!has_mesh && anim.is_none()) {
        return Err("skeletal: inconsistent mesh".into());
    }
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new(); // (offset, len)
    // appends a 4-byte aligned buffer view, returns its index
    let mut push = |bin: &mut Vec<u8>, bytes: Vec<u8>| {
        views.push((bin.len(), bytes.len()));
        bin.extend(bytes);
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        views.len() - 1
    };
    let f32s = |it: &mut dyn Iterator<Item = f32>| it.flat_map(f32::to_le_bytes).collect::<Vec<u8>>();
    let mut acc = Vec::new();
    let mut prims = Vec::new();
    let mut mats: Vec<u32> = m.sections.iter().map(|s| s.0).collect();
    mats.sort();
    mats.dedup();
    if has_mesh {
        let pos: Vec<[f32; 3]> = m.positions.iter().map(|p| [p[0] / 100.0, p[2] / 100.0, p[1] / 100.0]).collect();
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &pos {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        push(&mut bin, f32s(&mut pos.iter().flatten().copied()));
        push(&mut bin, f32s(&mut m.normals.iter().flat_map(|n| [n[0], n[2], n[1]])));
        push(&mut bin, f32s(&mut m.uvs.iter().flatten().copied()));
        push(&mut bin, m.joints.iter().flatten().flat_map(|j| j.to_le_bytes()).collect());
        push(&mut bin, f32s(&mut m.weights.iter().flatten().copied()));
        push(&mut bin, m.indices.iter().flat_map(|i| i.to_le_bytes()).collect());
        acc.extend([
            format!(r#"{{"bufferView":0,"componentType":5126,"count":{nv},"type":"VEC3","min":{lo:?},"max":{hi:?}}}"#),
            format!(r#"{{"bufferView":1,"componentType":5126,"count":{nv},"type":"VEC3"}}"#),
            format!(r#"{{"bufferView":2,"componentType":5126,"count":{nv},"type":"VEC2"}}"#),
            format!(r#"{{"bufferView":3,"componentType":5123,"count":{nv},"type":"VEC4"}}"#),
            format!(r#"{{"bufferView":4,"componentType":5126,"count":{nv},"type":"VEC4"}}"#),
        ]);
    }

    // bind pose: global = parent_global * local, all in glTF space
    let locals: Vec<_> = m.bones.iter().map(conv_bone).collect();
    let mut globals: Vec<M4> = Vec::with_capacity(nb);
    for (i, b) in m.bones.iter().enumerate() {
        let l = trs(locals[i].0, locals[i].1, locals[i].2);
        let g = if b.parent >= 0 && (b.parent as usize) < i { mul(&globals[b.parent as usize], &l) } else { l };
        globals.push(g);
    }
    let ibm_view = push(&mut bin, f32s(&mut globals.iter().flat_map(inv_affine)));
    acc.push(format!(r#"{{"bufferView":{ibm_view},"componentType":5126,"count":{nb},"type":"MAT4"}}"#));
    let ibm_acc = acc.len() - 1;

    if has_mesh {
        let full = [(0u32, 0u32, (m.indices.len() / 3) as u32)];
        let secs = if m.sections.is_empty() { &full[..] } else { &m.sections[..] };
        for &(mat, first, tris) in secs {
            if (first as usize + tris as usize * 3) > m.indices.len() {
                return Err("skeletal: section out of range".into());
            }
            acc.push(format!(r#"{{"bufferView":5,"byteOffset":{},"componentType":5125,"count":{},"type":"SCALAR"}}"#, first * 4, tris * 3));
            let mi = mats.iter().position(|&x| x == mat).unwrap_or(0);
            prims.push(format!(
                r#"{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2,"JOINTS_0":3,"WEIGHTS_0":4}},"indices":{},"material":{mi}}}"#,
                acc.len() - 1
            ));
        }
        if mats.is_empty() {
            mats.push(0);
        }
    }

    // animation: one sampler per animated channel; constant channels get a single key at t=0
    let mut anim_part = String::new();
    if let Some(a) = anim {
        let rate = if a.rate > 0.0 { a.rate } else { 30.0 };
        let times: Vec<f32> = (0..a.frames.max(1)).map(|k| k as f32 / rate).collect();
        let tv = push(&mut bin, f32s(&mut times.iter().copied()));
        acc.push(format!(r#"{{"bufferView":{tv},"componentType":5126,"count":{},"type":"SCALAR","min":[0],"max":[{}]}}"#, times.len(), times[times.len() - 1]));
        let t_full = acc.len() - 1;
        let tv = push(&mut bin, f32s(&mut [0f32].into_iter()));
        acc.push(format!(r#"{{"bufferView":{tv},"componentType":5126,"count":1,"type":"SCALAR","min":[0],"max":[0]}}"#));
        let t_one = acc.len() - 1;
        let (mut samplers, mut channels) = (Vec::new(), Vec::new());
        for t in &a.tracks {
            if t.bone >= nb {
                return Err("skeletal: animation track bone out of range".into());
            }
            let chans: [(&str, &str, usize, Vec<f32>); 3] = [
                ("translation", "VEC3", t.trans.len(), t.trans.iter().flat_map(|v| [v[0] / 100.0, v[2] / 100.0, v[1] / 100.0]).collect()),
                ("rotation", "VEC4", t.rot.len(), t.rot.iter().flat_map(|q| [-q[0], -q[2], -q[1], q[3]]).collect()),
                ("scale", "VEC3", t.scale.len(), t.scale.iter().flat_map(|s| [s[0], s[2], s[1]]).collect()),
            ];
            for (path, ty, n, vals) in chans {
                let input = match n {
                    0 => continue, // bind pose
                    1 => t_one,
                    n if n == times.len() => t_full,
                    _ => return Err("skeletal: animation track length mismatch".into()),
                };
                let v = push(&mut bin, f32s(&mut vals.into_iter()));
                acc.push(format!(r#"{{"bufferView":{v},"componentType":5126,"count":{n},"type":"{ty}"}}"#));
                samplers.push(format!(r#"{{"input":{input},"output":{},"interpolation":"LINEAR"}}"#, acc.len() - 1));
                channels.push(format!(r#"{{"sampler":{},"target":{{"node":{},"path":"{path}"}}}}"#, samplers.len() - 1, t.bone));
            }
        }
        anim_part = format!(r#","animations":[{{"name":"{}","samplers":[{}],"channels":[{}]}}]"#, esc(name), samplers.join(","), channels.join(","));
    }

    let mut nodes: Vec<String> = m
        .bones
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let (t, r, s) = locals[i];
            let kids: Vec<String> = (0..nb).filter(|&k| m.bones[k].parent == i as i32).map(|k| k.to_string()).collect();
            let ch = if kids.is_empty() { String::new() } else { format!(r#","children":[{}]"#, kids.join(",")) };
            format!(r#"{{"name":"{}","translation":{t:?},"rotation":{r:?},"scale":{s:?}{ch}}}"#, esc(&b.name))
        })
        .collect();
    let mut roots: Vec<String> = (0..nb).filter(|&i| m.bones[i].parent < 0).map(|i| i.to_string()).collect();
    if has_mesh {
        nodes.push(format!(r#"{{"name":"{}","mesh":0,"skin":0}}"#, esc(name)));
        roots.push(nb.to_string());
    }
    let (mut images, mut mats_json) = (Vec::new(), Vec::new());
    for &i in &mats {
        match textures.get(i as usize).and_then(|t| t.as_ref()) {
            Some(png) => {
                let view = push(&mut bin, png.clone());
                images.push(format!(r#"{{"bufferView":{view},"mimeType":"image/png"}}"#));
                let t = images.len() - 1;
                mats_json.push(format!(r#"{{"name":"mat_{i}","pbrMetallicRoughness":{{"baseColorTexture":{{"index":{t}}},"metallicFactor":0}}}}"#));
            }
            None => mats_json.push(format!(r#"{{"name":"mat_{i}"}}"#)),
        }
    }
    let tex_json: Vec<String> = (0..images.len()).map(|i| format!(r#"{{"source":{i}}}"#)).collect();
    let img_part = if images.is_empty() { String::new() } else { format!(r#","images":[{}],"textures":[{}]"#, images.join(","), tex_json.join(",")) };
    let mesh_part = if has_mesh {
        format!(r#","meshes":[{{"name":"{}","primitives":[{}]}}],"materials":[{}]{img_part}"#, esc(name), prims.join(","), mats_json.join(","))
    } else {
        String::new()
    };
    let views_json: Vec<String> = views.iter().map(|(o, l)| format!(r#"{{"buffer":0,"byteOffset":{o},"byteLength":{l}}}"#)).collect();
    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"ucas-explorer"}},"scene":0,"scenes":[{{"nodes":[{}]}}],"nodes":[{}]{mesh_part},"skins":[{{"joints":[{}],"inverseBindMatrices":{ibm_acc},"skeleton":{}}}]{anim_part},"accessors":[{}],"bufferViews":[{}],"buffers":[{{"byteLength":{}}}]}}"#,
        roots.join(","),
        nodes.join(","),
        (0..nb).map(|i| i.to_string()).collect::<Vec<_>>().join(","),
        roots[0],
        acc.join(","),
        views_json.join(","),
        bin.len()
    );
    let mut j = json.into_bytes();
    while j.len() % 4 != 0 {
        j.push(b' ');
    }
    let total = 12 + 8 + j.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend(b"glTF");
    out.extend(2u32.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((j.len() as u32).to_le_bytes());
    out.extend(b"JSON");
    out.extend(j);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(b"BIN\0");
    out.extend(bin);
    Ok(out)
}

fn esc(s: &str) -> String {
    s.chars().filter(|c| !c.is_control()).collect::<String>().replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_matrices_invert_and_skeleton_parses() {
        // skeleton: root + child at (0,0,100) cm UE -> (0,1,0) m glTF
        let mut d = vec![0xAAu8; 3];
        let w = |d: &mut Vec<u8>, v: u32| d.extend(v.to_le_bytes());
        w(&mut d, 2);
        for v in [0, 0, u32::MAX, 1, 0, 0] {
            w(&mut d, v);
        }
        w(&mut d, 2);
        for t in [[0.0f64, 0.0, 0.0], [0.0, 0.0, 100.0]] {
            for v in [0.0, 0.0, 0.0, 1.0].iter().chain(&t).chain(&[1.0, 1.0, 1.0]) {
                d.extend(v.to_le_bytes());
            }
        }
        let names = ["root".to_string(), "spine".to_string()];
        assert!(parse_skeleton(&d, 0, &names).is_none());
        let b = parse_skeleton(&d, 3, &names).unwrap();
        assert_eq!((b[1].name.as_str(), b[1].parent), ("spine", 0));
        let (t, q, s) = conv_bone(&b[1]);
        let m = trs(t, q, s);
        assert_eq!(&m[12..15], &[0.0, 1.0, 0.0]);
        let id = mul(&m, &inv_affine(&m));
        assert!(id.iter().enumerate().all(|(i, v)| (v - if i % 5 == 0 { 1.0 } else { 0.0 }).abs() < 1e-5));
        let sm = SkelMesh {
            bones: b,
            positions: vec![[0.0; 3]; 3],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0; 2]; 3],
            indices: vec![0, 1, 2],
            joints: vec![[1, 0, 0, 0]; 3],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
            sections: vec![(2, 0, 1)],
        };
        let g = to_glb(&sm, "t", &[]).unwrap();
        assert_eq!((&g[..4], u32::from_le_bytes(g[8..12].try_into().unwrap()) as usize), (&b"glTF"[..], g.len()));
    }

    /// End-to-end: decode real packages from the container and write .glb files to SKEL_OUT (default: scratch).
    /// Run: SKEL_FILTER=SK_Female_Base.uasset cargo test --release skel_e2e -- --ignored --nocapture
    #[test]
    #[ignore]
    fn skel_e2e() {
        let utoc_path = crate::test_utoc();
        let utoc = std::path::Path::new(&utoc_path);
        let toc = crate::open_archive(utoc).unwrap();
        let res = crate::global::Resolver::load(&utoc.with_file_name("global.utoc"), &toc).unwrap();
        let mut ucas = std::fs::File::open(utoc.with_extension("ucas")).unwrap();
        let filter = std::env::var("SKEL_FILTER").unwrap_or("/SK_".into());
        let out = std::env::var("SKEL_OUT").ok();
        let limit: usize = std::env::var("SKEL_LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
        let (mut ok, mut seen) = (0, 0);
        let mut errs = std::collections::BTreeMap::<String, usize>::new();
        for (p, e) in toc.files.iter().filter(|(p, _)| p.contains(&filter) && p.ends_with(".uasset")) {
            let Ok(pkg) = crate::zen::parse(crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).unwrap()) else { continue };
            let Some(i) = (0..pkg.exports.len()).find(|&i| res.class_name(&pkg, i) == "SkeletalMesh") else { continue };
            seen += 1;
            if seen > limit {
                break;
            }
            match decode_skeletal_mesh(&pkg, i, None) {
                Ok(m) => {
                    ok += 1;
                    eprintln!("ok {p}: {} bones, {} verts, {} tris, {} sections", m.bones.len(), m.positions.len(), m.indices.len() / 3, m.sections.len());
                    if let Some(dir) = &out {
                        let stem = p.rsplit('/').next().unwrap().trim_end_matches(".uasset");
                        std::fs::write(format!("{dir}/{stem}.glb"), to_glb(&m, stem, &[]).unwrap()).unwrap();
                    }
                }
                Err(err) => {
                    eprintln!("FAIL {p}: {err}");
                    *errs.entry(err.to_string()).or_default() += 1;
                }
            }
        }
        eprintln!("decoded {ok}/{} {errs:?}", seen.min(limit));
    }
}
