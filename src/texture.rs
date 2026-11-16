//! Texture2D cooked platform data -> RGBA8 -> PNG. OWNER: agent D.
use crate::Res;

pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>, // width*height*4
}

/// `export` = the Texture2D export's serialized data (ZenPackage::export_data).
/// `props_end` = byte offset where properties end, if known (props::read_properties); None => find
/// the native data heuristically (e.g. scan for the "PF_*" pixel format FString).
/// `ubulk` = the matching .ubulk file contents if the package has one (mips stored out of line).
pub fn decode_texture2d(export: &[u8], props_end: Option<usize>, ubulk: Option<&[u8]>) -> Res<Image> {
    let d = export;
    let u32_at = |o: usize| -> Res<u32> {
        d.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).ok_or_else(|| "texture: truncated".into())
    };
    // Layout (UE5 cooked): ... SizeX SizeY PackedData FString(PixelFormat) [OptData] FirstMip NumMips
    // then per mip: i32 bulk-data-map index, [inline payload], SizeX SizeY SizeZ.
    let mut p = find_platform_data(d, props_end.unwrap_or(0)).ok_or("texture: no PF_* platform data found")?;
    let (sx, sy, packed) = (u32_at(p)?, u32_at(p + 4)?, u32_at(p + 8)?);
    let len = u32_at(p + 12)? as usize;
    let fmt = std::str::from_utf8(&d[p + 16..p + 16 + len - 1]).map_err(|_| "texture: bad format string")?.to_string();
    let (bw, bb) = fmt_info(&fmt).ok_or_else(|| format!("texture: unsupported pixel format {fmt}"))?;
    let slices = (packed & 0x1FFF_FFFF).max(1) as usize;
    p += 16 + len;
    if packed & (1 << 30) != 0 {
        p += 8; // FOptTexturePlatformData { ExtData, NumMipsInTail }
    }
    let (first, nmips) = (u32_at(p)?, u32_at(p + 4)?);
    if nmips == 0 || first + nmips > 32 {
        return Err(format!("texture: bad mip count {nmips}").into());
    }
    p += 8;
    let size_of = |w: u32, h: u32| (w as usize).div_ceil(bw) * (h as usize).div_ceil(bw) * bb;
    // (w, h, Ok(inline payload) | Err(external size))
    let mut mips: Vec<(u32, u32, Result<&[u8], usize>)> = Vec::new();
    for j in 0..nmips {
        let (ew, eh) = ((sx >> (first + j)).max(1), (sy >> (first + j)).max(1));
        p += 4; // bulk data map index
        if u32_at(p)? == ew && u32_at(p + 4)? == eh {
            let depth = (u32_at(p + 8)? as usize).max(slices);
            mips.push((ew, eh, Err(size_of(ew, eh) * depth)));
        } else {
            let n = size_of(ew, eh) * slices;
            let payload = d.get(p..p + n).ok_or("texture: truncated inline mip")?;
            p += n;
            if u32_at(p)? != ew || u32_at(p + 4)? != eh {
                return Err(format!("texture: mip {j} layout not understood").into());
            }
            mips.push((ew, eh, Ok(payload)));
        }
        p += 12;
    }
    // ponytail: external mip offsets are derived from sizes, not from the package header's bulk data
    // map (we only get the export bytes). Assumes the .ubulk holds external mips contiguously, largest
    // first; if they don't all fit, the largest ones live in .uptnl/.m.ubulk, so align to the file end.
    // Upgrade: pass FBulkDataMapEntry offsets from zen.rs.
    let ext_total: usize = mips.iter().filter_map(|m| m.2.err()).sum();
    let ub = ubulk.unwrap_or(&[]);
    let mut off = if ext_total <= ub.len() { 0 } else { ub.len() as isize - ext_total as isize };
    for (w, h, src) in mips {
        let data = match src {
            Ok(b) => Some(b),
            Err(n) => {
                let o = off;
                off += n as isize;
                (o >= 0).then(|| ub.get(o as usize..o as usize + n)).flatten()
            }
        };
        if let Some(data) = data {
            return decode(&fmt, &data[..size_of(w, h)], w, h);
        }
    }
    Err("texture: no mip data available (missing .ubulk?)".into())
}

/// Offset of SizeX before the first plausible `FString "PF_*"` at or after `start`.
fn find_platform_data(d: &[u8], start: usize) -> Option<usize> {
    // ponytail: byte scan for the pixel-format FString instead of walking UTexture/UTexture2D native
    // fields; first match wins, so only the first cooked format is decoded. Would misfire if a string
    // property literally held "PF_..." with plausible dims before it.
    let rd = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
    (start.max(12)..d.len().saturating_sub(8)).find(|&i| {
        let len = rd(i) as usize;
        (4..=32).contains(&len)
            && i + 4 + len <= d.len()
            && &d[i + 4..i + 7] == b"PF_"
            && d[i + 3 + len] == 0
            && d[i + 4..i + 3 + len].iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
            && (1..=16384).contains(&rd(i - 12))
            && (1..=16384).contains(&rd(i - 8))
    }).map(|i| i - 12)
}

/// (block width in pixels — 1 for uncompressed, bytes per block/pixel)
fn fmt_info(fmt: &str) -> Option<(usize, usize)> {
    Some(match fmt {
        "PF_DXT1" | "PF_BC4" => (4, 8),
        "PF_DXT3" | "PF_DXT5" | "PF_BC5" | "PF_BC6H" | "PF_BC7" => (4, 16),
        "PF_B8G8R8A8" | "PF_R8G8B8A8" => (1, 4),
        "PF_G8" => (1, 1),
        "PF_FloatRGBA" => (1, 8),
        _ => return None,
    })
}

fn decode(fmt: &str, data: &[u8], w: u32, h: u32) -> Res<Image> {
    let (wu, hu) = (w as usize, h as usize);
    let px = wu * hu;
    let rgba: Vec<u8> = match fmt {
        "PF_B8G8R8A8" => data.chunks_exact(4).flat_map(|c| [c[2], c[1], c[0], c[3]]).collect(),
        "PF_R8G8B8A8" => data.to_vec(),
        "PF_G8" => data.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        "PF_FloatRGBA" => data
            .chunks_exact(2)
            .map(|c| (half_to_f32(u16::from_le_bytes([c[0], c[1]])).clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
            .collect(),
        _ => {
            use texture2ddecoder as t;
            let mut buf = vec![0u32; px];
            match fmt {
                "PF_DXT1" => t::decode_bc1(data, wu, hu, &mut buf),
                "PF_DXT3" => t::decode_bc2(data, wu, hu, &mut buf),
                "PF_DXT5" => t::decode_bc3(data, wu, hu, &mut buf),
                "PF_BC4" => t::decode_bc4(data, wu, hu, &mut buf),
                "PF_BC5" => t::decode_bc5(data, wu, hu, &mut buf),
                "PF_BC6H" => t::decode_bc6_unsigned(data, wu, hu, &mut buf),
                "PF_BC7" => t::decode_bc7(data, wu, hu, &mut buf),
                _ => Err("unsupported"),
            }
            .map_err(|e| format!("texture: {fmt} decode failed: {e}"))?;
            // texture2ddecoder packs as little-endian BGRA
            let bc4 = fmt == "PF_BC4";
            buf.iter()
                .flat_map(|v| {
                    let [b, g, r, a] = v.to_le_bytes();
                    if bc4 { [r, r, r, 255] } else { [r, g, b, a] }
                })
                .collect()
        }
    };
    if rgba.len() != px * 4 {
        return Err(format!("texture: {fmt} {w}x{h}: wrong data size").into());
    }
    Ok(Image { width: w, height: h, rgba })
}

fn half_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let mag = match e {
        0 => m as f32 / 16_777_216.0, // subnormal: m * 2^-24
        31 => if m == 0 { f32::INFINITY } else { 0.0 },
        _ => f32::from_bits(((e + 112) << 23) | (m << 13)),
    };
    if s == 1 { -mag } else { mag }
}

pub fn to_png(img: &Image) -> Res<Vec<u8>> {
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, img.width, img.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().map_err(|e| e.to_string())?.write_image_data(&img.rgba).map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_g8_with_crop_and_scan() {
        // fake export: junk, SizeX=3 SizeY=2 Packed=1, "PF_G8", FirstMip=0, NumMips=1, one inline mip
        let mut d = vec![0xAAu8; 5];
        for v in [3u32, 2, 1, 6] { d.extend(v.to_le_bytes()); }
        d.extend(b"PF_G8\0");
        for v in [0u32, 1, 0] { d.extend(v.to_le_bytes()); }
        d.extend([1, 2, 3, 4, 5, 6]);
        for v in [3u32, 2, 1] { d.extend(v.to_le_bytes()); }
        let img = decode_texture2d(&d, None, None).unwrap();
        assert_eq!((img.width, img.height), (3, 2));
        assert_eq!(&img.rgba[4..8], &[2, 2, 2, 255]);
        assert!(to_png(&img).unwrap().starts_with(b"\x89PNG"));
        assert!((half_to_f32(0x3c00) - 1.0).abs() < 1e-6);
        // BC1 5x5 => 2x2 blocks needed, crop to 5x5
        assert_eq!(decode("PF_DXT1", &[0u8; 32], 5, 5).unwrap().rgba.len(), 100);
    }

    #[test]
    #[ignore]
    fn decode_scratch_files() {
        let dir = std::env::var("TEX_DIR").unwrap();
        for e in std::fs::read_dir(&dir).unwrap() {
            let path = e.unwrap().path();
            if path.extension().map_or(true, |x| x != "uasset") { continue; }
            let d = std::fs::read(&path).unwrap();
            let ub = std::fs::read(path.with_extension("ubulk")).ok();
            match decode_texture2d(&d, None, ub.as_deref()) {
                Ok(img) => {
                    println!("{} {}x{}", path.display(), img.width, img.height);
                    std::fs::write(path.with_extension("png"), to_png(&img).unwrap()).unwrap();
                }
                Err(e) => println!("{} ERR {e}", path.display()),
            }
        }
    }
}
