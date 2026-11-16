//! Cooked UStaticMesh LOD geometry -> OBJ, without .usmap.
// ponytail: properties are unversioned (no schema), so instead of walking UStaticMesh::Serialize we
// scan for each LOD's FPositionVertexBuffer signature (stride=12, n, elemSize=12, n) and parse the
// fixed buffer chain after it (StaticMeshVertexBuffer, ColorVertexBuffer, IndexBuffer). Each hit is
// fully validated (lengths + index range). Material sections: see find_sections.
// Upgrade path: parse FStaticMeshRenderData properly once property sizes are known.
use crate::Res;

pub struct Mesh {
    pub positions: Vec<[f32; 3]>, // UE space, cm
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>, // UV channel 0
    pub indices: Vec<u32>,
    /// (material index, first index, num triangles); one section covering everything if not found
    pub sections: Vec<(u32, u32, u32)>,
}

struct Rd<'a>(&'a [u8], usize);
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.0.get(self.1..self.1.checked_add(n)?)?;
        self.1 += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
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

/// Parses one LOD whose FPositionVertexBuffer starts at `at`. None if anything doesn't add up.
fn parse_lod(d: &[u8], at: usize) -> Option<Mesh> {
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
    let (full_uv, hi_tan) = (r.u32()?, r.u32()?);
    if nv != n || !(1..=8).contains(&ntc) || full_uv > 1 || hi_tan > 1 {
        return None;
    }
    let (tes, tn, tan) = r.bulk()?;
    let (ues, un, uv) = r.bulk()?;
    if tn != n || tes != if hi_tan == 1 { 16 } else { 8 } || un != n * ntc || ues != if full_uv == 1 { 8 } else { 4 } {
        return None;
    }
    // tangent pair = (TangentX, TangentZ=normal); normal is the second half
    let normals = tan
        .chunks_exact(tes)
        .map(|c| {
            let h = &c[tes / 2..];
            if hi_tan == 1 {
                std::array::from_fn(|k| i16::from_le_bytes([h[k * 2], h[k * 2 + 1]]) as f32 / 32767.0)
            } else {
                std::array::from_fn(|k| h[k] as i8 as f32 / 127.0)
            }
        })
        .collect();
    let uvs = uv
        .chunks_exact(ues * ntc)
        .map(|c| {
            if full_uv == 1 {
                [f32::from_le_bytes(c[0..4].try_into().unwrap()), f32::from_le_bytes(c[4..8].try_into().unwrap())]
            } else {
                [f16(u16::from_le_bytes([c[0], c[1]])), f16(u16::from_le_bytes([c[2], c[3]]))]
            }
        })
        .collect();

    // FColorVertexBuffer: strip flags (2), Stride, NumVertices, [bulk]
    r.take(2)?;
    let (_, cn) = (r.u32()?, r.u32()? as usize);
    if cn != 0 {
        let (ces, cn2, _) = r.bulk()?;
        if ces != 4 || cn2 != n {
            return None;
        }
    }

    // FRawStaticIndexBuffer: b32Bit, bulk<u8>
    let b32 = r.u32()?;
    let (ies, _, raw) = r.bulk()?;
    if b32 > 1 || ies != 1 {
        return None;
    }
    let indices: Vec<u32> = if b32 == 1 {
        raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    } else {
        raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect()
    };
    if indices.is_empty() || indices.len() % 3 != 0 || indices.iter().any(|&i| i as usize >= n) {
        return None;
    }
    let sections = vec![(0, 0, (indices.len() / 3) as u32)];
    Some(Mesh { positions, normals, uvs, indices, sections })
}

fn scan(d: &[u8], out: &mut Vec<Mesh>) {
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

/// All LODs found in the StaticMesh export (and its .ubulk, for streamed LODs), most detailed first.
pub fn decode_static_mesh(export: &[u8], ubulk: Option<&[u8]>) -> Res<Vec<Mesh>> {
    let mut lods = Vec::new();
    scan(export, &mut lods);
    if let Some(b) = ubulk {
        scan(b, &mut lods);
    }
    if lods.is_empty() {
        return Err("mesh: no LOD vertex buffers found (Nanite-only or unsupported layout?)".into());
    }
    lods.sort_by_key(|m| std::cmp::Reverse(m.indices.len()));
    for m in &mut lods {
        if let Some((_, s)) = find_sections(export, m) {
            m.sections = s;
        }
    }
    Ok(lods)
}

/// Finds this LOD's TArray<FStaticMeshSection> in the export: i32 count, then per section
/// MaterialIndex, FirstIndex, NumTriangles, MinVertexIndex, MaxVertexIndex + bool32 flags
/// (UE5.3: Collision, CastShadow, ForceOpaque, VisibleInRayTracing, AffectDistanceFieldLighting).
/// Returns (offset of the count, sections). Streamed LODs keep their sections inline too.
// ponytail: signature scan, validated by contiguous FirstIndex, triangle sum and vertex range;
// a coincidental byte match is possible but needs all of those to line up.
pub fn find_sections(d: &[u8], m: &Mesh) -> Option<(usize, Vec<(u32, u32, u32)>)> {
    let u = |o: usize| d.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let (tris, nv) = (m.tris() as u64, m.positions.len() as u32);
    for o in 0..d.len().saturating_sub(48) {
        let n = u(o)? as usize;
        if !(1..=256).contains(&n) || u(o + 8)? != 0 {
            continue;
        }
        'stride: for stride in [40, 36, 32] {
            let (mut next, mut sum, mut out) = (0u32, 0u64, Vec::with_capacity(n));
            for k in 0..n {
                let b = o + 4 + k * stride;
                let Some(f) = (0..stride / 4).map(|j| u(b + j * 4)).collect::<Option<Vec<u32>>>() else { continue 'stride };
                if f[0] > 255 || f[1] != next || f[3] > f[4] || f[4] >= nv || f[5..].iter().any(|&x| x > 1) {
                    continue 'stride;
                }
                let Some(nx) = f[2].checked_mul(3).and_then(|t| t.checked_add(f[1])) else { continue 'stride };
                next = nx;
                sum += f[2] as u64;
                out.push((f[0], f[1], f[2]));
            }
            if sum == tris {
                return Some((o, out));
            }
        }
    }
    None
}

impl Mesh {
    pub fn tris(&self) -> usize {
        self.indices.len() / 3
    }

    /// Wavefront OBJ, converted to Y-up right-handed meters (UE is Z-up left-handed cm).
    pub fn to_obj(&self, name: &str) -> String {
        use std::fmt::Write;
        let mut s = format!("# exported by ucas-explorer\no {name}\n");
        for p in &self.positions {
            let _ = writeln!(s, "v {} {} {}", p[0] / 100.0, p[2] / 100.0, p[1] / 100.0);
        }
        for t in &self.uvs {
            let _ = writeln!(s, "vt {} {}", t[0], 1.0 - t[1]);
        }
        for n in &self.normals {
            let _ = writeln!(s, "vn {} {} {}", n[0], n[2], n[1]);
        }
        // swapping Y/Z mirrors the mesh, which flips UE's clockwise winding to OBJ's counter-clockwise
        for t in self.indices.chunks_exact(3) {
            let [a, b, c] = [t[0] + 1, t[1] + 1, t[2] + 1];
            let _ = writeln!(s, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}");
        }
        s
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parses_synthetic_lod() {
        let mut d = vec![0xAAu8; 5]; // junk before the signature
        let w = |d: &mut Vec<u8>, v: u32| d.extend(v.to_le_bytes());
        w(&mut d, 12);
        w(&mut d, 3);
        w(&mut d, 12);
        w(&mut d, 3);
        for v in [0.0f32, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0] {
            d.extend(v.to_le_bytes());
        }
        d.extend([0, 0]); // strip flags
        for v in [1, 3, 0, 0, 8, 3] {
            w(&mut d, v);
        }
        d.extend([0u8; 24]); // tangents
        w(&mut d, 4);
        w(&mut d, 3);
        d.extend([0, 0x3c, 0, 0].repeat(3)); // uv (1.0, 0.0) as halfs
        d.extend([0, 0]);
        w(&mut d, 4);
        w(&mut d, 0); // no colors
        w(&mut d, 0); // 16-bit indices
        w(&mut d, 1);
        w(&mut d, 6);
        d.extend([0, 0, 1, 0, 2, 0]);
        let lods = super::decode_static_mesh(&d, None).unwrap();
        assert_eq!((lods.len(), lods[0].tris(), lods[0].uvs[0]), (1, 1, [1.0, 0.0]));
        assert!(lods[0].to_obj("t").contains("v 1 0 0\n"));
        // corrupt index -> rejected
        let n = d.len();
        d[n - 2] = 9;
        assert!(super::decode_static_mesh(&d, None).is_err());
    }
}
