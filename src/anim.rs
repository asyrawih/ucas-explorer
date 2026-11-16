//! Cooked UAnimSequence -> per-bone sampled tracks, without .usmap.
// ponytail: properties are unversioned, so we don't walk them. The native tail of a cooked UE5 AnimSequence
// (FCompressedAnimSequence::SerializeCompressedData) is found from the codec's FString (bone codec DDC handle,
// e.g. "AnimCompress_PerTrackCompression_3") and walked outwards:
//   ... i32 RawDataSize, TArray<i32> TrackToSkeleton, TArray<FName> CurveNames,
//   i32 NumBytes, bool bUseBulk, u8[NumBytes] stream, FString BoneCodec, FString CurveCodec, TArray<u8> curves,
//   codec data (UE codecs: i32 NumFrames, u8 KeyFormat, 3x u8 formats, i32 ByteStream/TrackOffsets/ScaleOffsets
//   counts, i32 ScaleStripSize).
// Layout verified on Palia (UE5.3+). Reference: CUE4Parse UAnimSequence.cs / AnimConverter.cs (UE4 codecs).
// Upgrade path: property parsing with a .usmap would give SequenceLength, frame rate and additive type directly.
use crate::Res;

/// One bone's local transform per frame (UE space). Each Vec holds 1 (constant) or `Anim::frames` values.
pub struct Track {
    pub bone: usize, // index into the USkeleton reference skeleton
    pub rot: Vec<[f32; 4]>, // xyzw
    pub trans: Vec<[f32; 3]>,
    pub scale: Vec<[f32; 3]>,
}

pub struct Anim {
    pub codec: String,
    pub frames: usize,
    pub rate: f32, // frames per second
    pub tracks: Vec<Track>,
}

impl Anim {
    pub fn length(&self) -> f32 {
        self.frames.saturating_sub(1) as f32 / self.rate
    }
}

/// AnimSequence export -> .glb with skeleton + animation, plus a skinned mesh when one is found: the package
/// matching `mesh_hint` (container path substring), else a SkeletalMesh next to the Skeleton that imports it.
/// Returns (glb, mesh container path used).
pub fn anim_to_glb(
    res: &crate::global::Resolver,
    pkg: &crate::zen::ZenPackage,
    export: usize,
    mesh_hint: Option<&str>,
    load: &mut dyn FnMut(&str) -> Option<Vec<u8>>,
) -> Res<(Vec<u8>, Option<String>)> {
    use crate::{global, skeletal, zen};
    let mut anim = decode_anim_sequence(pkg, export)?;
    // the USkeleton the track table indexes: the imported package whose export is a Skeleton
    let parse = |load: &mut dyn FnMut(&str) -> Option<Vec<u8>>, path: &str| load(path).and_then(|d| zen::parse(d).ok());
    let container = |package: &str| res.packages.get(&global::package_id(package)).cloned();
    let (skel_pkg, skel_bones) = pkg
        .imported_package_names
        .iter()
        .filter(|n| !n.starts_with("/Script/"))
        .find_map(|n| {
            let p = parse(load, &container(n)?)?;
            let i = (0..p.exports.len()).find(|&i| res.class_name(&p, i) == "Skeleton")?;
            Some((n.clone(), skeletal::find_reference_skeleton(p.export_data(i), &p.names)?))
        })
        .ok_or("anim: Skeleton import not found or undecodable")?;

    // ponytail: mesh auto-pick scans packages in the Skeleton's folder (then its parent), capped at 200 loads;
    // meshes elsewhere need `mesh_hint`.
    let skel_path = container(&skel_pkg).unwrap_or_default();
    let dir = skel_path.rsplit_once('/').map_or("", |(d, _)| d).to_string();
    let parent = dir.rsplit_once('/').map_or("", |(d, _)| d).to_string();
    let mut cands: Vec<&String> = match mesh_hint {
        Some(h) => res.packages.values().filter(|p| p.to_lowercase().contains(&h.to_lowercase())).collect(),
        None => {
            let mut c: Vec<_> = res.packages.values().filter(|p| p.rsplit_once('/').is_some_and(|(d, _)| d == dir)).collect();
            c.sort();
            let mut up: Vec<_> = res.packages.values().filter(|p| !parent.is_empty() && p.starts_with(&format!("{parent}/")) && !c.contains(p)).collect();
            up.sort();
            c.extend(up);
            c
        }
    };
    cands.retain(|p| p.ends_with(".uasset"));
    let mut mesh = None;
    for path in cands.into_iter().take(200) {
        let Some(p) = parse(load, path) else { continue };
        if mesh_hint.is_none() && !p.imported_package_names.contains(&skel_pkg) {
            continue;
        }
        let Some(i) = (0..p.exports.len()).find(|&i| res.class_name(&p, i) == "SkeletalMesh") else { continue };
        let ubulk = load(&format!("{}.ubulk", path.trim_end_matches(".uasset")));
        if let Ok(m) = skeletal::decode_skeletal_mesh(&p, i, ubulk.as_deref()) {
            let need = m.sections.iter().map(|s| s.0 as usize + 1).max().unwrap_or(1);
            let tex = crate::material::slot_diffuse_pngs(res, &p, i, need, load);
            mesh = Some((m, tex, path.clone()));
            break;
        }
    }
    let (m, tex, used) = match mesh {
        Some((m, t, u)) => (m, t, Some(u)),
        None => {
            let empty = skeletal::SkelMesh { bones: Vec::new(), positions: vec![], normals: vec![], uvs: vec![], indices: vec![], joints: vec![], weights: vec![], sections: vec![] };
            (skeletal::SkelMesh { bones: skel_bones.iter().map(|b| skeletal::Bone { name: b.name.clone(), ..*b }).collect(), ..empty }, vec![], None)
        }
    };
    // ponytail: additive detection by Palia naming (ADD_ / Additive); applied as local-space additive on the
    // Skeleton ref pose (ABPT_RefPose). Additives based on another anim's frame come out approximate.
    let name = &pkg.exports[export].name;
    if name.starts_with("ADD_") || name.contains("Additive") {
        for t in &mut anim.tracks {
            let Some(b) = skel_bones.get(t.bone) else { continue };
            t.rot = t.rot.iter().map(|q| qmul(*q, b.rotation)).collect();
            t.trans = t.trans.iter().map(|v| std::array::from_fn(|k| v[k] + b.translation[k])).collect();
            // per-track "no scale" decodes as 1; additive scale is a delta around 0
            t.scale = t.scale.iter().map(|s| std::array::from_fn(|k| b.scale[k] * if s == &[1.0; 3] { 1.0 } else { 1.0 + s[k] })).collect();
        }
    }
    // skeleton bone index -> mesh bone index by name; tracks for bones the mesh lacks are dropped
    anim.tracks.retain_mut(|t| {
        let Some(n) = skel_bones.get(t.bone).map(|b| &b.name) else { return false };
        m.bones.iter().position(|b| &b.name == n).map(|i| t.bone = i).is_some()
    });
    if anim.tracks.is_empty() {
        return Err("anim: no tracks map onto the skeleton".into());
    }
    Ok((skeletal::to_glb_with_animation(&m, name, &tex, &anim)?, used))
}

fn qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

struct Rd<'a>(&'a [u8], usize);
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let s = self.1.checked_add(n).and_then(|e| self.0.get(self.1..e)).ok_or("anim: truncated")?;
        self.1 += n;
        Ok(s)
    }
    fn u8(&mut self) -> Res<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Res<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn f32(&mut self) -> Res<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn v3(&mut self) -> Res<[f32; 3]> {
        Ok([self.f32()?, self.f32()?, self.f32()?])
    }
    fn fstring(&mut self) -> Res<String> {
        let n = self.u32()? as usize;
        if n > 4096 {
            return Err("anim: bad FString".into());
        }
        Ok(String::from_utf8_lossy(self.take(n)?).trim_end_matches('\0').to_string())
    }
    fn align4(&mut self) {
        self.1 = self.1.next_multiple_of(4);
    }
}

fn u32_at(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(o..o.checked_add(4)?)?.try_into().ok()?))
}

/// Decodes the bone tracks of an AnimSequence export.
pub fn decode_anim_sequence(pkg: &crate::zen::ZenPackage, export: usize) -> Res<Anim> {
    let d = pkg.exports.get(export).map(|_| pkg.export_data(export)).ok_or("anim: bad export index")?;
    // 1. bone codec FString
    let codec_at = ["AnimCompress_", "AnimBoneCompressionCodec"]
        .iter()
        .filter_map(|pat| (4..d.len()).find(|&k| d[k..].starts_with(pat.as_bytes())))
        .filter_map(|k| {
            let n = u32_at(d, k - 4)? as usize;
            (n > 1 && d.get(k + n - 1) == Some(&0) && !d[k..k + n - 1].contains(&0)).then_some(k - 4)
        })
        .min()
        .ok_or("anim: no bone codec string (no compressed data?)")?;
    let mut r = Rd(d, codec_at);
    let codec = r.fstring()?;
    let _curve_codec = r.fstring()?;
    let n = r.u32()? as usize;
    r.take(n)?; // compressed curves (ignored)
    let frames = r.u32()? as usize;
    if frames == 0 || frames > 1_000_000 {
        return Err("anim: bad frame count".into());
    }

    // 2. serialized byte stream ends right at the codec string: [i32 NumBytes][bool bUseBulk][bytes]. Its length
    // (and the track count) comes from the codec data: UE codecs list their array sizes, ACL buffers start with
    // their own size and the compressed_tracks tag.
    let ue = codec.starts_with("AnimCompress_");
    let (stream_len, expected) = if ue {
        let kf = *d.get(r.1).ok_or("anim: truncated")?;
        let u = |k: usize| u32_at(d, r.1 + 4 + k * 4).map(|v| v as usize).ok_or("anim: truncated");
        let (nbytes, noff, nscale) = (u(0)?, u(1)?, u(2)?);
        (nbytes.checked_add(noff.checked_add(nscale).ok_or("anim: bad sizes")? * 4).ok_or("anim: bad sizes")?, noff / if kf == 2 { 2 } else { 4 })
    } else if codec.starts_with("AnimBoneCompressionCodec_ACL") {
        let k = (16..codec_at.saturating_sub(16))
            .rev()
            .find(|&k| u32_at(d, k) == Some(ACL_TAG) && u32_at(d, k - 16) == Some((codec_at - (k - 16) - 8) as u32))
            .ok_or("anim: ACL buffer not found")?;
        (codec_at - (k - 8), u32_at(d, k + 8).unwrap_or(0) as usize)
    } else {
        return Err(format!("anim: unsupported codec {codec}").into());
    };
    let stream_at = codec_at.checked_sub(stream_len + 8).ok_or("anim: byte stream out of range")?;
    match u32_at(d, stream_at + 4) {
        _ if u32_at(d, stream_at) != Some(stream_len as u32) => return Err("anim: byte stream header mismatch".into()),
        Some(0) => {}
        _ => return Err("anim: bulk-data byte stream not supported".into()),
    }
    let stream = &d[stream_at + 8..codec_at];

    // 3. TArray<i32> track->bone, TArray<FName> curve names, ending at the stream header
    // ponytail: nearest backward candidate with the codec's track count, sane bone indices and a curve array that
    // fits exactly.
    let track_map = (4..stream_at.saturating_sub(7))
        .rev()
        .find_map(|q| {
            let nt = u32_at(d, q)? as usize;
            if nt != expected || nt == 0 {
                return None;
            }
            let m = q + 4 + nt * 4;
            let nc = u32_at(d, m)? as usize;
            if m + 4 + nc.checked_mul(8)? != stream_at {
                return None;
            }
            (0..nt).map(|i| u32_at(d, q + 4 + i * 4).filter(|&b| b < 4096).map(|b| b as usize)).collect::<Option<Vec<_>>>()
        })
        .ok_or("anim: track-to-skeleton table not found")?;

    if ue {
        let tracks = decode_ue(&mut r, stream, stream_len, &track_map, frames)?;
        Ok(Anim { codec, frames, rate: guess_rate(&d[..stream_at], frames), tracks })
    } else {
        let (tracks, frames, rate) = decode_acl(stream, &track_map)?;
        Ok(Anim { codec, frames, rate, tracks })
    }
}

// ponytail: SequenceLength is an unversioned property we can't locate by name; pick the common frame rate whose
// implied length (frames-1)/rate appears as an f32 in the property bytes. Falls back to 30 fps.
fn guess_rate(props: &[u8], frames: usize) -> f32 {
    let has = |len: f32| props.windows(4).any(|w| (f32::from_le_bytes(w.try_into().unwrap()) - len).abs() <= len * 1e-5);
    [30.0, 60.0, 24.0, 25.0, 50.0, 120.0, 15.0, 48.0, 20.0]
        .into_iter()
        .find(|&r: &f32| frames > 1 && has((frames - 1) as f32 / r))
        .unwrap_or(30.0)
}

// AnimationCompressionFormat
const F_NONE: u8 = 0;
const F_FLOAT96: u8 = 1;
const F_FIXED48: u8 = 2;
const F_INTERVAL32: u8 = 3;
const F_FIXED32: u8 = 4;
const F_FLOAT32: u8 = 5;
const F_IDENTITY: u8 = 6;

/// UE4 codecs (FUECompressedAnimData): AKF_ConstantKeyLerp / AKF_VariableKeyLerp (bitwise, remove-linear-keys) and
/// AKF_PerTrackCompression.
fn decode_ue(r: &mut Rd, stream: &[u8], stream_len: usize, map: &[usize], frames: usize) -> Res<Vec<Track>> {
    let key_format = r.u8()?;
    let (tf, rf, sf) = (r.u8()?, r.u8()?, r.u8()?);
    let (nbytes, noff, nscale, strip) = (r.u32()? as usize, r.u32()? as usize, r.u32()? as usize, r.u32()? as usize);
    if noff.checked_add(nscale).and_then(|x| x.checked_mul(4)).and_then(|x| x.checked_add(nbytes)) != Some(stream_len) {
        return Err("anim: UE codec stream size mismatch".into());
    }
    let ints = |o: usize, n: usize| -> Vec<i32> { (0..n).map(|i| u32_at(stream, o + i * 4).unwrap_or(u32::MAX) as i32).collect() };
    let offs = ints(0, noff);
    let scale_offs = ints(noff * 4, nscale);
    let bytes = &stream[(noff + nscale) * 4..];
    let nt = map.len();
    let per_track = key_format == 2;
    if noff != nt * if per_track { 2 } else { 4 } || (nscale != 0 && (strip == 0 || nscale != nt * strip)) {
        return Err(format!("anim: UE codec offset table size mismatch nt={nt} noff={noff} nscale={nscale} strip={strip} kf={key_format}").into());
    }
    let mut out = Vec::with_capacity(nt);
    for (t, &bone) in map.iter().enumerate() {
        let track = if per_track {
            let at = |o: i32| -> Res<Rd> { if (o as usize) < bytes.len() { Ok(Rd(bytes, o as usize)) } else { Err("anim: bad track offset".into()) } };
            let (to, ro, so) = (offs[t * 2], offs[t * 2 + 1], if nscale > 0 { scale_offs[t * strip] } else { -1 });
            Track {
                bone,
                trans: if to < 0 { vec![[0.0; 3]] } else { pt_vec(&mut at(to)?, frames)? },
                rot: if ro < 0 { vec![[0.0, 0.0, 0.0, 1.0]] } else { pt_quat(&mut at(ro)?, frames)? },
                scale: if so < 0 { vec![[1.0; 3]] } else { pt_vec(&mut at(so)?, frames)? },
            }
        } else {
            let var = key_format == 1;
            let (to, tn, ro, rn) = (offs[t * 4], offs[t * 4 + 1], offs[t * 4 + 2], offs[t * 4 + 3]);
            let (so, sn) = if nscale > 0 && strip >= 2 { (scale_offs[t * strip], scale_offs[t * strip + 1]) } else { (0, 0) };
            let at = |o: i32| -> Res<Rd> { if o >= 0 && (o as usize) < bytes.len() { Ok(Rd(bytes, o as usize)) } else { Err("anim: bad track offset".into()) } };
            Track {
                bone,
                trans: if tn > 0 { kl_vec(&mut at(to)?, tn as usize, tf, var, frames)? } else { vec![[0.0; 3]] },
                rot: if rn > 0 { kl_quat(&mut at(ro)?, rn as usize, rf, var, frames)? } else { vec![[0.0, 0.0, 0.0, 1.0]] },
                scale: if sn > 0 && so > 0 { kl_vec(&mut at(so)?, sn as usize, sf, var, frames)? } else { vec![[1.0; 3]] },
            }
        };
        out.push(track);
    }
    Ok(out)
}

fn quat_w(x: f32, y: f32, z: f32) -> [f32; 4] {
    let w2 = 1.0 - x * x - y * y - z * z;
    [x, y, z, if w2 > 0.0 { w2.sqrt() } else { 0.0 }]
}

fn fixed48_quat(r: &mut Rd, mask: u32) -> Res<[f32; 4]> {
    let mut c = [0f32; 3];
    for (k, v) in c.iter_mut().enumerate() {
        if mask & (1 << k) != 0 {
            *v = (r.u16()? as f32 - 32767.0) / 32767.0;
        }
    }
    Ok(quat_w(c[0], c[1], c[2]))
}

/// 11/11/10-bit packed (x in the top bits); `iv` = Some((mins, ranges)) for interval formats.
fn packed32_quat(r: &mut Rd, iv: Option<([f32; 3], [f32; 3])>) -> Res<[f32; 4]> {
    let p = r.u32()?;
    let (ux, uy, uz) = (p >> 21, (p >> 10) & 0x7ff, p & 0x3ff);
    let mut c = [(ux as f32 - 1023.0) / 1023.0, (uy as f32 - 1023.0) / 1023.0, (uz as f32 - 511.0) / 511.0];
    if let Some((m, rg)) = iv {
        for k in 0..3 {
            c[k] = c[k] * rg[k] + m[k];
        }
    }
    Ok(quat_w(c[0], c[1], c[2]))
}

fn float32_quat(r: &mut Rd) -> Res<[f32; 4]> {
    let p = r.u32()?;
    let (ux, uy, uz) = (p >> 21, (p >> 10) & 0x7ff, p & 0x3ff);
    let f11 = |u: u32| f32::from_bits(((((u >> 7) & 7) + 123) << 23) | (((u & 0x7f) | 32 * (u & 0xffff_fc00)) << 16));
    let z = f32::from_bits(((((uz >> 6) & 7) + 123) << 23) | (((uz & 0x3f) | 32 * (uz & 0xffff_fe00)) << 17));
    Ok(quat_w(f11(ux), f11(uy), z))
}

fn interval_vec(r: &mut Rd, m: [f32; 3], rg: [f32; 3]) -> Res<[f32; 3]> {
    let p = r.u32()?;
    let (uz, uy, ux) = (p >> 21, (p >> 10) & 0x7ff, p & 0x3ff);
    let c = [(ux as f32 - 511.0) / 511.0, (uy as f32 - 1023.0) / 1023.0, (uz as f32 - 1023.0) / 1023.0];
    Ok(std::array::from_fn(|k| c[k] * rg[k] + m[k]))
}

/// PerTrack header: u32 (format << 28 | component mask << 24 | key count); mask bit 3 = has time keys.
fn pt_header(r: &mut Rd) -> Res<(u8, u32, usize, Option<([f32; 3], [f32; 3])>)> {
    let h = r.u32()?;
    let (fmt, mask, n) = ((h >> 28) as u8, (h >> 24) & 0xf, (h & 0xff_ffff) as usize);
    if n == 0 || n > 1_000_000 {
        return Err("anim: bad per-track key count".into());
    }
    let iv = if fmt == F_INTERVAL32 {
        let (mut m, mut rg) = ([0f32; 3], [0f32; 3]);
        for k in 0..3 {
            if mask & (1 << k) != 0 {
                m[k] = r.f32()?;
                rg[k] = r.f32()?;
            }
        }
        Some((m, rg))
    } else {
        None
    };
    Ok((fmt, mask, n, iv))
}

fn pt_quat(r: &mut Rd, frames: usize) -> Res<Vec<[f32; 4]>> {
    let (fmt, mask, n, iv) = pt_header(r)?;
    let keys = (0..n)
        .map(|_| match fmt {
            F_NONE | F_FLOAT96 => Ok(quat_w(r.f32()?, r.f32()?, r.f32()?)),
            F_FIXED48 => fixed48_quat(r, mask),
            F_FIXED32 => packed32_quat(r, None),
            F_INTERVAL32 => packed32_quat(r, iv),
            F_FLOAT32 => float32_quat(r),
            F_IDENTITY => Ok([0.0, 0.0, 0.0, 1.0]),
            _ => Err(format!("anim: unknown rotation format {fmt}").into()),
        })
        .collect::<Res<Vec<_>>>()?;
    r.align4();
    let times = if mask & 8 != 0 { Some(times(r, n, frames)?) } else { None };
    Ok(resample(keys, times, frames, true))
}

fn pt_vec(r: &mut Rd, frames: usize) -> Res<Vec<[f32; 3]>> {
    let (fmt, mask, n, iv) = pt_header(r)?;
    let comp = |r: &mut Rd, f: &dyn Fn(&mut Rd) -> Res<f32>| -> Res<[f32; 3]> {
        let mut v = [0f32; 3];
        for (k, x) in v.iter_mut().enumerate() {
            if mask & (1 << k) != 0 {
                *x = f(r)?;
            }
        }
        Ok(v)
    };
    let keys = (0..n)
        .map(|_| match fmt {
            F_NONE | F_FLOAT96 if mask & 7 == 0 => r.v3(),
            F_NONE | F_FLOAT96 => comp(r, &|r| r.f32()),
            F_INTERVAL32 => {
                let (m, rg) = iv.unwrap_or_default();
                interval_vec(r, m, rg)
            }
            // DecodeFixed48_PerTrackComponent<7>: offset 255, factor 1
            F_FIXED48 => comp(r, &|r| Ok(r.u16()? as f32 - 255.0)),
            F_IDENTITY => Ok([0.0; 3]),
            _ => Err(format!("anim: unknown vector format {fmt}").into()),
        })
        .collect::<Res<Vec<_>>>()?;
    r.align4();
    let times = if mask & 8 != 0 { Some(times(r, n, frames)?) } else { None };
    Ok(resample(keys, times, frames, false))
}

/// Key -> frame table: u8 per key if frames < 256, else u16.
fn times(r: &mut Rd, n: usize, frames: usize) -> Res<Vec<f32>> {
    if n <= 1 {
        return Ok(vec![0.0; n]);
    }
    let t = (0..n).map(|_| Ok(if frames < 256 { r.u8()? as f32 } else { r.u16()? as f32 })).collect::<Res<Vec<_>>>()?;
    r.align4();
    Ok(t)
}

fn kl_vec(r: &mut Rd, n: usize, fmt: u8, var: bool, frames: usize) -> Res<Vec<[f32; 3]>> {
    let fmt = if n == 1 { F_NONE } else { fmt };
    let iv = if fmt == F_INTERVAL32 { Some((r.v3()?, r.v3()?)) } else { None };
    let keys = (0..n)
        .map(|_| match fmt {
            F_NONE | F_FLOAT96 => r.v3(),
            F_INTERVAL32 => {
                let (m, rg) = iv.unwrap_or_default();
                interval_vec(r, m, rg)
            }
            F_FIXED48 => {
                let mut v = [0f32; 3];
                for x in &mut v {
                    *x = (r.u16()? as f32 - 32767.0) / 32767.0 * 128.0;
                }
                Ok(v)
            }
            F_IDENTITY => Ok([0.0; 3]),
            _ => Err(format!("anim: unknown vector format {fmt}").into()),
        })
        .collect::<Res<Vec<_>>>()?;
    r.align4();
    let times = if var { Some(times(r, n, frames)?) } else { None };
    Ok(resample(keys, times, frames, false))
}

fn kl_quat(r: &mut Rd, n: usize, fmt: u8, var: bool, frames: usize) -> Res<Vec<[f32; 4]>> {
    let fmt = if n == 1 { F_FLOAT96 } else { fmt };
    let iv = if fmt == F_INTERVAL32 { Some((r.v3()?, r.v3()?)) } else { None };
    let keys = (0..n)
        .map(|_| match fmt {
            F_NONE => Ok([r.f32()?, r.f32()?, r.f32()?, r.f32()?]),
            F_FLOAT96 => Ok(quat_w(r.f32()?, r.f32()?, r.f32()?)),
            F_FIXED48 => fixed48_quat(r, 7),
            F_FIXED32 => packed32_quat(r, None),
            F_INTERVAL32 => packed32_quat(r, iv),
            F_FLOAT32 => float32_quat(r),
            F_IDENTITY => Ok([0.0, 0.0, 0.0, 1.0]),
            _ => Err(format!("anim: unknown rotation format {fmt}").into()),
        })
        .collect::<Res<Vec<_>>>()?;
    r.align4();
    let times = if var { Some(times(r, n, frames)?) } else { None };
    Ok(resample(keys, times, frames, true))
}

/// Keys (at `times` frame numbers, or spread evenly over the clip) -> one value per frame. Quaternions are
/// nlerped along the shortest arc.
fn resample<const N: usize>(keys: Vec<[f32; N]>, times: Option<Vec<f32>>, frames: usize, quat: bool) -> Vec<[f32; N]> {
    let n = keys.len();
    if n <= 1 || frames <= 1 {
        return keys.into_iter().take(1).collect();
    }
    let times = times.unwrap_or_else(|| (0..n).map(|i| i as f32 * (frames - 1) as f32 / (n - 1) as f32).collect());
    (0..frames)
        .map(|f| {
            let f = f as f32;
            let j = times.partition_point(|&t| t <= f).clamp(1, n - 1);
            let (t0, t1) = (times[j - 1], times[j]);
            let a = if t1 > t0 { ((f - t0) / (t1 - t0)).clamp(0.0, 1.0) } else { 0.0 };
            let (k0, mut k1) = (keys[j - 1], keys[j]);
            if quat && (0..N).map(|k| k0[k] * k1[k]).sum::<f32>() < 0.0 {
                k1 = k1.map(|x| -x);
            }
            let mut v: [f32; N] = std::array::from_fn(|k| k0[k] + (k1[k] - k0[k]) * a);
            if quat {
                let l = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if l > 1e-8 {
                    v = v.map(|x| x / l);
                }
            }
            v
        })
        .collect()
}

// --- ACL ---
// Port of the uniformly sampled transform track decompression of ACL 2.1 (compressed_tracks_version16 9..10), from
// Animation Compression Library by Nicholas Frechette (https://github.com/nfrechette/acl, MIT License):
// includes/acl/core/impl/compressed_headers.h, decompression/impl/decompression.transform.h,
// animated_track_cache.transform.h, constant_track_cache.transform.h, math/vector4_packing.h.
// Every stored sample is decoded exactly (no interpolation); stripped keyframes are re-filled by `resample`.
// ponytail: no database (streamed tiers) support; such clips are rejected.
const ACL_TAG: u32 = 0xac11_ac11;

/// MSB-first bit read (ACL animated data is a big-endian bit stream).
fn acl_bits(d: &[u8], at: usize, n: u32) -> Res<u32> {
    let mut v = 0u32;
    for i in 0..n as usize {
        let b = *d.get((at + i) / 8).ok_or("anim: ACL data truncated")?;
        v = (v << 1) | ((b >> (7 - (at + i) % 8)) & 1) as u32;
    }
    Ok(v)
}

/// Decodes an ACL compressed_tracks buffer into tracks + (frames, sample rate).
fn decode_acl(buf: &[u8], map: &[usize]) -> Res<(Vec<Track>, usize, f32)> {
    let u = |o: usize| -> Res<u32> { Ok(u32_at(buf, o).ok_or("anim: ACL data truncated")?) };
    let f = |o: usize| u(o).map(f32::from_bits);
    let (version, algo, ttype) = (u16::from_le_bytes([buf.get(12).copied().unwrap_or(0), buf.get(13).copied().unwrap_or(0)]), buf.get(14), buf.get(15));
    if u(8)? != ACL_TAG || !(9..=10).contains(&version) || algo != Some(&0) || ttype != Some(&12) {
        return Err(format!("anim: unsupported ACL buffer (version {version})").into());
    }
    let (nt, ns, rate, misc) = (u(16)? as usize, u(20)? as usize, f(24)?, u(28)?);
    if nt != map.len() || ns == 0 || ns > 1_000_000 || !(rate > 0.0 && rate < 10_000.0) {
        return Err("anim: bad ACL header".into());
    }
    let has_scale = misc & 1 != 0;
    let default_scale = ((misc >> 1) & 1) as f32;
    let (scale_var, trans_var, rot_fmt) = ((misc >> 2) & 1 != 0, (misc >> 3) & 1 != 0, (misc >> 4) & 15);
    let (has_db, trivial_defaults, stripped, wrap) = (misc & (1 << 8) != 0, misc & (1 << 9) != 0, misc & (1 << 10) != 0, misc & (1 << 30) != 0);
    if has_db || !matches!(rot_fmt, 0 | 2 | 3) {
        return Err("anim: ACL database / unknown rotation format not supported".into());
    }
    let rot_var = rot_fmt == 3;
    // transform_tracks_header, offsets relative to it
    let th = 32;
    let h = |k: usize| u(th + k * 4).map(|v| v as usize);
    let (nseg, nvar, nrot, ntr, nsc) = (h(0)?, h(1)?, h(2)?, h(3)?, h(4)?);
    let (seg_hdrs, types_at, const_at, clip_at) = (th + h(9)?, th + h(10)?, th + h(11)?, th + h(12)?);
    if nseg == 0 || nseg > ns {
        return Err("anim: bad ACL segment count".into());
    }

    // sub-track types: 2 bits each (0 default, 1 constant, 2 animated), 16 per u32, first track in the top bits
    let entries = nt.div_ceil(16);
    let kind = |set: usize, i: usize| u(types_at + (set * entries + i / 16) * 4).map(|e| (e >> (30 - 2 * (i % 16))) & 3);
    let rk: Vec<u32> = (0..nt).map(|i| kind(0, i)).collect::<Res<_>>()?;
    let tk: Vec<u32> = (0..nt).map(|i| kind(1, i)).collect::<Res<_>>()?;
    let sk: Vec<u32> = if has_scale { (0..nt).map(|i| kind(2, i)).collect::<Res<_>>()? } else { vec![0; nt] };
    let count = |k: &[u32], v: u32| k.iter().filter(|&&x| x == v).count();
    if count(&rk, 2) != nrot || count(&tk, 2) != ntr || count(&sk, 2) != nsc {
        return Err("anim: ACL sub-track counts mismatch".into());
    }

    // constant samples: rotations (quat_full: xyzw f32; else SoA groups of 4: xxxx yyyy zzzz), then float3 AoS
    let ncr = count(&rk, 1);
    let rot_const_size = if rot_fmt == 0 { 16 } else { 12 };
    let quat_xyz = |x: f32, y: f32, z: f32| [x, y, z, (1.0 - x * x - y * y - z * z).abs().sqrt()];
    let const_rot = |k: usize| -> Res<[f32; 4]> {
        if rot_fmt == 0 {
            let o = const_at + k * 16;
            return Ok([f(o)?, f(o + 4)?, f(o + 8)?, f(o + 12)?]);
        }
        let (g, i) = (k / 4, k % 4);
        let gs = (ncr - g * 4).min(4);
        let o = const_at + g * 48 + i * 4;
        Ok(quat_xyz(f(o)?, f(o + gs * 4)?, f(o + gs * 8)?))
    };
    let v3 = |o: usize| -> Res<[f32; 3]> { Ok([f(o)?, f(o + 4)?, f(o + 8)?]) };
    let const_tr_at = const_at + rot_const_size * ncr;
    let const_sc_at = const_tr_at + 12 * count(&tk, 1);

    // per-frame values of every animated sub-track: (clip frame, value)
    let mut rot_keys: Vec<Vec<(f32, [f32; 4])>> = vec![Vec::new(); nrot];
    let mut tr_keys: Vec<Vec<(f32, [f32; 3])>> = vec![Vec::new(); ntr];
    let mut sc_keys: Vec<Vec<(f32, [f32; 3])>> = vec![Vec::new(); nsc];
    let raw_bits = 31; // v02_01_99_1+
    let rot_pad = nrot.next_multiple_of(4);
    let multi = nseg > 1;
    for s in 0..nseg {
        let start = if multi { u(th + 52 + s * 4)? as usize } else { 0 };
        let end = if s + 1 < nseg { u(th + 52 + (s + 1) * 4)? as usize } else { ns };
        let hs = seg_hdrs + s * if stripped { 20 } else { 16 };
        let (pose_bits, rot_bits, tr_bits, data) = (u(hs)? as usize, u(hs + 4)? as usize, u(hs + 8)? as usize, th + u(hs + 12)? as usize);
        let frames: Vec<usize> = if stripped {
            let mask = u(hs + 16)?;
            (0..(end.saturating_sub(start)).min(32)).filter(|k| mask & (1 << (31 - k)) != 0).map(|k| start + k).collect()
        } else {
            (start..end).collect()
        };
        let fmt = data;
        let range = (fmt + nvar).next_multiple_of(2);
        let anim = (range + if multi { 6 * nvar } else { 0 }).next_multiple_of(4) * 8; // in bits
        let byte = |o: usize| buf.get(o).copied().map(|b| b as f32).ok_or("anim: ACL data truncated");
        for (key, &frame) in frames.iter().enumerate() {
            let fr = frame as f32;
            let mut bit = anim + key * pose_bits;
            for r in 0..nrot {
                let (g, i) = (r / 4, r % 4);
                let v: [f32; 3] = if rot_fmt == 0 {
                    let q: [f32; 4] = std::array::from_fn(|c| f32::from_bits(acl_bits(buf, bit + c * 32, 32).unwrap_or(0)));
                    bit += 128;
                    rot_keys[r].push((fr, q));
                    continue;
                } else if !rot_var {
                    bit += 96;
                    std::array::from_fn(|c| f32::from_bits(acl_bits(buf, bit - 96 + c * 32, 32).unwrap_or(0)))
                } else {
                    let nb = *buf.get(fmt + r).ok_or("anim: ACL data truncated")? as u32;
                    let seg = range + g * 24 + i;
                    let (mut v, seg_ok, clip_ok) = if nb == 0 {
                        let c = |k: usize| -> Res<f32> { Ok((byte(seg + k * 8)? * 256.0 + byte(seg + k * 8 + 4)?) / 65535.0) };
                        ([c(0)?, c(1)?, c(2)?], false, true)
                    } else if nb == raw_bits {
                        let v = std::array::from_fn(|c| f32::from_bits(acl_bits(buf, bit + c * 32, 32).unwrap_or(0)));
                        bit += 96;
                        (v, false, false)
                    } else if nb <= 23 {
                        let m = ((1u32 << nb) - 1) as f32;
                        let v = [acl_bits(buf, bit, nb)? as f32 / m, acl_bits(buf, bit + nb as usize, nb)? as f32 / m, acl_bits(buf, bit + 2 * nb as usize, nb)? as f32 / m];
                        bit += 3 * nb as usize;
                        (v, true, true)
                    } else {
                        return Err("anim: bad ACL bit rate".into());
                    };
                    if multi && seg_ok {
                        for c in 0..3 {
                            v[c] = v[c] * (byte(seg + 12 + c * 4)? / 255.0) + byte(seg + c * 4)? / 255.0;
                        }
                    }
                    if clip_ok {
                        let gs = (nrot - g * 4).min(4);
                        let cb = clip_at + g * 96 + i * 4;
                        for c in 0..3 {
                            v[c] = v[c] * f(cb + (3 + c) * gs * 4)? + f(cb + c * gs * 4)?;
                        }
                    }
                    v
                };
                rot_keys[r].push((fr, quat_xyz(v[0], v[1], v[2])));
            }
            // translations then scales: AoS ranges (u8x3 min + u8x3 extent per segment, f32x3 min + extent per clip)
            let vectors = |keys: &mut Vec<Vec<(f32, [f32; 3])>>, var: bool, mut bit: usize, fmt0: usize, seg0: usize, clip0: usize| -> Res<()> {
                for (t, k) in keys.iter_mut().enumerate() {
                    let raw = |bit: usize| -> [f32; 3] { std::array::from_fn(|c| f32::from_bits(acl_bits(buf, bit + c * 32, 32).unwrap_or(0))) };
                    if !var {
                        k.push((fr, raw(bit)));
                        bit += 96;
                        continue;
                    }
                    let nb = *buf.get(fmt0 + t).ok_or("anim: ACL data truncated")? as u32;
                    let seg = seg0 + t * 6;
                    let (mut v, seg_ok, clip_ok) = if nb == 0 {
                        let c = |k: usize| -> Res<f32> { Ok((byte(seg + 2 * k)? + byte(seg + 2 * k + 1)? * 256.0) / 65535.0) };
                        ([c(0)?, c(1)?, c(2)?], false, true)
                    } else if nb == raw_bits {
                        bit += 96;
                        (raw(bit - 96), false, false)
                    } else if nb <= 23 {
                        let m = ((1u32 << nb) - 1) as f32;
                        let v = [acl_bits(buf, bit, nb)? as f32 / m, acl_bits(buf, bit + nb as usize, nb)? as f32 / m, acl_bits(buf, bit + 2 * nb as usize, nb)? as f32 / m];
                        bit += 3 * nb as usize;
                        (v, true, true)
                    } else {
                        return Err("anim: bad ACL bit rate".into());
                    };
                    if multi && seg_ok {
                        for c in 0..3 {
                            v[c] = v[c] * (byte(seg + 3 + c)? / 255.0) + byte(seg + c)? / 255.0;
                        }
                    }
                    if clip_ok {
                        let cb = clip0 + t * 24;
                        for c in 0..3 {
                            v[c] = v[c] * f(cb + 12 + c * 4)? + f(cb + c * 4)?;
                        }
                    }
                    k.push((fr, v));
                }
                Ok(())
            };
            let pose = anim + key * pose_bits;
            let (rot_meta, tr_meta) = (if rot_var { rot_pad } else { 0 }, if trans_var { ntr } else { 0 });
            let rot_clip = if rot_var { nrot * 24 } else { 0 };
            let tr_clip = if trans_var { ntr * 24 } else { 0 };
            vectors(&mut tr_keys, trans_var, pose + rot_bits, fmt + rot_meta, range + rot_meta * 6, clip_at + rot_clip)?;
            vectors(&mut sc_keys, scale_var, pose + rot_bits + tr_bits, fmt + rot_meta + tr_meta, range + rot_meta * 6 + tr_meta * 6, clip_at + rot_clip + tr_clip)?;
        }
    }

    // assemble per track; wrap-optimized clips repeat their first sample at the end
    let frames = ns + wrap as usize;
    fn fill<const N: usize>(keys: Vec<(f32, [f32; N])>, frames: usize, wrap: bool, quat: bool) -> Vec<[f32; N]> {
        let mut keys = keys;
        if wrap && let Some(&(_, v)) = keys.first() {
            keys.push(((frames - 1) as f32, v));
        }
        let (t, v): (Vec<f32>, Vec<[f32; N]>) = keys.into_iter().unzip();
        if v.len() == frames { resample(v, None, frames, quat) } else { resample(v, Some(t), frames, quat) }
    }
    let (mut rc, mut tc, mut sc) = (0, 0, 0);
    let mut rot_keys = rot_keys.into_iter();
    let mut tr_keys = tr_keys.into_iter();
    let mut sc_keys = sc_keys.into_iter();
    let mut out = Vec::with_capacity(nt);
    for (i, &bone) in map.iter().enumerate() {
        // default sub-tracks: identity when the clip says so, else the bind pose (empty = leave at rest pose)
        let rot = match rk[i] {
            1 => (vec![const_rot(rc)?], rc += 1).0,
            2 => fill(rot_keys.next().unwrap_or_default(), frames, wrap, true),
            _ if trivial_defaults => vec![[0.0, 0.0, 0.0, 1.0]],
            _ => vec![],
        };
        let trans = match tk[i] {
            1 => (vec![v3(const_tr_at + 12 * tc)?], tc += 1).0,
            2 => fill(tr_keys.next().unwrap_or_default(), frames, wrap, false),
            _ if trivial_defaults => vec![[0.0; 3]],
            _ => vec![],
        };
        let scale = match sk[i] {
            1 => (vec![v3(const_sc_at + 12 * sc)?], sc += 1).0,
            2 => fill(sc_keys.next().unwrap_or_default(), frames, wrap, false),
            _ if trivial_defaults || !has_scale => vec![[default_scale; 3]],
            _ => vec![],
        };
        out.push(Track { bone, rot, trans, scale });
    }
    Ok((out, frames, rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_track_fixed48_quat_with_times_resamples() {
        // format Fixed48 (2), mask x|time (1|8), 2 keys; x = +-0 and 0.6 -> w = 1 and 0.8; key frames 0 and 4
        let mut b = ((2u32 << 28) | (9 << 24) | 2).to_le_bytes().to_vec();
        b.extend(32767u16.to_le_bytes());
        b.extend(((0.6f32 * 32767.0) as u16 + 32767).to_le_bytes());
        b.extend([0, 4, 0, 0]); // times (u8, frames < 256), aligned
        let q = pt_quat(&mut Rd(&b, 0), 5).unwrap();
        assert_eq!(q.len(), 5);
        assert!((q[0][3] - 1.0).abs() < 1e-4 && (q[4][0] - 0.6).abs() < 1e-3 && (q[4][3] - 0.8).abs() < 1e-3);
        let l: f32 = q[2].iter().map(|x| x * x).sum();
        assert!((l - 1.0).abs() < 1e-4 && q[2][0] > 0.2 && q[2][0] < 0.4);
        assert!(pt_quat(&mut Rd(&b[..6], 0), 5).is_err());
    }

    /// Batch decode (no files written). ANIM_GLB=1 also runs the full export (Skeleton + mesh lookup, glTF).
    /// Run: ANIM_STEP=24 ANIM_GLB=1 cargo test --release anim_e2e -- --ignored --nocapture
    #[test]
    #[ignore]
    fn anim_e2e() {
        let utoc_path = crate::test_utoc();
        let utoc = std::path::Path::new(&utoc_path);
        let toc = crate::open_archive(utoc).unwrap();
        let res = crate::global::Resolver::load(&utoc.with_file_name("global.utoc"), &toc).unwrap();
        let mut ucas = std::fs::File::open(utoc.with_extension("ucas")).unwrap();
        let filter = std::env::var("ANIM_FILTER").unwrap_or("/Anim".into());
        let limit: usize = std::env::var("ANIM_LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
        let glb = std::env::var("ANIM_GLB").is_ok();
        let step: usize = std::env::var("ANIM_STEP").ok().and_then(|s| s.parse().ok()).unwrap_or(1).max(1);
        let mut n = 0;
        let (mut ok, mut seen, mut with_mesh) = (0, 0, 0);
        let mut errs = std::collections::BTreeMap::<String, usize>::new();
        let files: Vec<(String, u32)> = toc.files.iter().filter(|(p, _)| p.contains(&filter) && p.ends_with(".uasset")).cloned().collect();
        for (p, e) in &files {
            let Ok(pkg) = crate::zen::parse(crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).unwrap()) else { continue };
            let Some(i) = (0..pkg.exports.len()).find(|&i| res.class_name(&pkg, i) == "AnimSequence") else { continue };
            n += 1;
            if n % step != 0 {
                continue;
            }
            seen += 1;
            if seen > limit {
                break;
            }
            let r = if glb {
                let mut load = |q: &str| -> Option<Vec<u8>> {
                    let (_, e) = toc.files.iter().find(|(f, _)| f == q)?;
                    crate::read_chunk(&toc, &mut ucas, *e, u64::MAX).ok()
                };
                anim_to_glb(&res, &pkg, i, None, &mut load).map(|(g, m)| format!("{} bytes, mesh {m:?}", g.len()))
            } else {
                decode_anim_sequence(&pkg, i).map(|a| format!("{} frames @{} fps, {} tracks, {}", a.frames, a.rate, a.tracks.len(), a.codec))
            };
            match r {
                Ok(s) => {
                    ok += 1;
                    with_mesh += s.contains("mesh Some") as usize;
                    eprintln!("ok {p}: {s}");
                }
                Err(err) => {
                    eprintln!("FAIL {p}: {err}");
                    *errs.entry(err.to_string()).or_default() += 1;
                }
            }
        }
        eprintln!("decoded {ok}/{} (with mesh: {with_mesh}) {errs:#?}", seen.min(limit));
    }
}
