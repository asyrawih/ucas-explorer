//! Zen (IoStore) package header parser. OWNER: agent A.
//! Input = whole ExportBundleData chunk of a .uasset/.umap (crate::read_chunk with u64::MAX).
#![allow(dead_code)]
use crate::Res;

/// FPackageObjectIndex raw u64. Top 2 bits = type: 0 Export, 1 ScriptImport, 2 PackageImport, 3 Null.
pub type ObjIndex = u64;

pub struct ZenExport {
    pub name: String,
    pub outer_index: ObjIndex,
    pub class_index: ObjIndex,
    pub super_index: ObjIndex,
    pub template_index: ObjIndex,
    pub public_export_hash: u64,
    pub object_flags: u32,
    /// absolute byte range of this export's serialized data inside `ZenPackage::data`
    pub data_offset: usize,
    pub data_size: usize,
}

pub struct ZenPackage {
    pub name: String,                         // e.g. "/Game/Maps/Foo"
    pub flags: u32,                           // PackageFlags (0x2000 = unversioned properties)
    pub names: Vec<String>,                   // name map
    pub imported_packages: Vec<u64>,          // package ids (FPackageId of imported_package_names)
    pub imported_package_names: Vec<String>,  // e.g. "/Script/Engine", "/Game/Foo"
    pub imported_public_export_hashes: Vec<u64>,
    pub imports: Vec<ObjIndex>,               // import map
    pub exports: Vec<ZenExport>,
    pub data: Vec<u8>,
}

impl ZenPackage {
    pub fn export_data(&self, i: usize) -> &[u8] {
        let e = &self.exports[i];
        &self.data[e.data_offset..e.data_offset + e.data_size]
    }
}

struct Cur<'a>(&'a [u8], usize);
impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let s = self.1.checked_add(n).and_then(|e| self.0.get(self.1..e)).ok_or("zen header truncated")?;
        self.1 += n;
        Ok(s)
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn u64s(&mut self, n: usize) -> Res<Vec<u64>> {
        (0..n).map(|_| self.u64()).collect()
    }
}

/// FNameEntrySerialized batch: count, numStringBytes, hashVersion, hashes[count], BE headers[count], strings.
fn load_name_batch(c: &mut Cur) -> Res<Vec<String>> {
    let n = c.u32()? as usize;
    if n == 0 {
        return Ok(vec![]);
    }
    c.take(4 + 8)?; // numStringBytes, hashVersion
    c.take(n.checked_mul(8).ok_or("bad name count")?)?; // hashes
    let headers = c.take(n * 2)?;
    headers
        .chunks(2)
        .map(|h| {
            let len = (((h[0] & 0x7f) as usize) << 8) | h[1] as usize;
            Ok(if h[0] & 0x80 != 0 {
                let w: Vec<u16> = c.take(len * 2)?.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
                String::from_utf16_lossy(&w)
            } else {
                String::from_utf8_lossy(c.take(len)?).into()
            })
        })
        .collect()
}

/// FMappedName (index u32 with 2 type bits on top, number u32) -> "Name" or "Name_{n-1}".
fn mapped_name(names: &[String], idx: u32, num: u32) -> Res<String> {
    let base = names.get((idx & 0x3FFF_FFFF) as usize).ok_or("bad name index")?;
    Ok(if num == 0 { base.clone() } else { format!("{base}_{}", num - 1) })
}

/// Layout seen in Palia (UE5.3+): FZenPackageSummary is 0x3C bytes (two trailing i32s of unknown use
/// after ImportedPackageNamesOffset), name batch follows. Export data starts at HeaderSize and
/// FExportMapEntry::CookedSerialOffset is relative to that.
pub fn parse(data: Vec<u8>) -> Res<ZenPackage> {
    let mut c = Cur(&data, 0);
    if c.u32()? != 0 {
        return Err("zen: versioned packages (bHasVersioningInfo) not supported".into());
    }
    let header_size = c.u32()? as usize;
    let (name_idx, name_num) = (c.u32()?, c.u32()?);
    let flags = c.u32()?;
    let _cooked_header_size = c.u32()?;
    let mut off = || -> Res<usize> { Ok(c.u32()? as usize) };
    let hashes_off = off()?;
    let import_off = off()?;
    let export_off = off()?;
    let bundle_off = off()?;
    let _dep_headers_off = off()?;
    let _dep_entries_off = off()?;
    let pkg_names_off = off()?;
    let _ = (off()?, off()?);
    if header_size > data.len() || !(hashes_off <= import_off && import_off <= export_off && export_off <= bundle_off) {
        return Err("zen: bad summary offsets".into());
    }
    let names = load_name_batch(&mut c)?;
    let name = mapped_name(&names, name_idx, name_num)?;

    c.1 = hashes_off;
    let imported_public_export_hashes = c.u64s((import_off - hashes_off) / 8)?;
    let imports = c.u64s((export_off - import_off) / 8)?;

    c.1 = pkg_names_off;
    let pkg_names = load_name_batch(&mut c)?;
    let imported_package_names = pkg_names
        .iter()
        .map(|n| Ok(match c.u32()? {
            0 => n.clone(),
            num => format!("{n}_{}", num - 1),
        }))
        .collect::<Res<Vec<_>>>()?;
    let imported_packages = imported_package_names.iter().map(|n| crate::global::package_id(n)).collect();

    c.1 = export_off;
    let mut exports = Vec::new();
    for _ in 0..(bundle_off - export_off) / 72 {
        let serial_off = c.u64()? as usize;
        let size = c.u64()? as usize;
        let (ni, nn) = (c.u32()?, c.u32()?);
        let [outer_index, class_index, super_index, template_index, public_export_hash] =
            c.u64s(5)?.try_into().map_err(|_| "unreachable")?;
        let object_flags = c.u32()?;
        c.take(4)?; // FilterFlags + pad
        let data_offset = header_size.checked_add(serial_off).ok_or("bad export offset")?;
        if data_offset.checked_add(size).is_none_or(|e| e > data.len()) {
            return Err("zen: export data out of bounds".into());
        }
        exports.push(ZenExport {
            name: mapped_name(&names, ni, nn)?,
            outer_index,
            class_index,
            super_index,
            template_index,
            public_export_hash,
            object_flags,
            data_offset,
            data_size: size,
        });
    }
    Ok(ZenPackage { name, flags, names, imported_packages, imported_package_names, imported_public_export_hashes, imports, exports, data })
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_synthetic() {
        // summary (0x3C) + name batch ["Foo", "Bar"] at 0x3C..0x67, 1 export at 0x68, empty pkg-name batch, 4 data bytes
        let (exp, pkg, hdr) = (0x68u32, 0x68 + 72, 0x68 + 72 + 4);
        let mut d = vec![];
        for v in [0, hdr, 0, 2, 0x8000_2200, 0, exp, exp, exp, pkg, pkg, pkg, pkg, 0, 0] {
            d.extend(u32::to_le_bytes(v));
        }
        d.extend([2, 0, 0, 0, 6, 0, 0, 0]); // count, numStringBytes
        d.extend([0; 8 + 16]); // hashVersion, hashes
        d.extend([0, 3, 0, 3]);
        d.extend(b"FooBar");
        d.resize(exp as usize, 0);
        d.extend(0u64.to_le_bytes()); // CookedSerialOffset
        d.extend(4u64.to_le_bytes()); // CookedSerialSize
        d.extend([1, 0, 0, 0, 3, 0, 0, 0]); // name "Bar_2"
        d.extend([0xFF; 40]);
        d.extend([0; 8]);
        d.extend([0; 4]); // empty pkg-name batch
        d.extend(b"DATA");
        let p = super::parse(d).unwrap();
        assert_eq!(p.name, "Foo_1");
        assert_eq!(p.names, ["Foo", "Bar"]);
        assert_eq!(p.exports[0].name, "Bar_2");
        assert_eq!(p.export_data(0), b"DATA");
    }
}
