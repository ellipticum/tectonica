use planet_engine::{
    PlanetInputs, SimulationConfig, TectonicInputs,
    run_simulation_native,
};
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
//  Env helpers
// ---------------------------------------------------------------------------

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_str(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

// ---------------------------------------------------------------------------
//  Binary array I/O
// ---------------------------------------------------------------------------

fn write_f32_array(path: &Path, data: &[f32]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    fs::write(path, bytes).expect("failed to write f32 array");
}

fn write_i32_array(path: &Path, data: &[i32]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    fs::write(path, bytes).expect("failed to write i32 array");
}

fn write_i16_array(path: &Path, data: &[i16]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
    };
    fs::write(path, bytes).expect("failed to write i16 array");
}

fn write_i8_array(path: &Path, data: &[i8]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len())
    };
    fs::write(path, bytes).expect("failed to write i8 array");
}

fn read_f32_array(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    assert!(bytes.len() % 4 == 0, "f32 file size not multiple of 4");
    let mut out = vec![0f32; bytes.len() / 4];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    out
}

fn read_u8_array(path: &Path) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn read_i16_array(path: &Path) -> Vec<i16> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    assert!(bytes.len() % 2 == 0);
    let mut out = vec![0i16; bytes.len() / 2];
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        out[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
    }
    out
}

fn read_i8_array(path: &Path) -> Vec<i8> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    bytes.iter().map(|&b| b as i8).collect()
}

// ---------------------------------------------------------------------------
//  Color helpers
// ---------------------------------------------------------------------------

fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

fn clampu8(v: f32) -> u8 {
    clampf(v, 0.0, 255.0).round() as u8
}

struct ColorStop { t: f32, r: f32, g: f32, b: f32 }

fn sample_stops(stops: &[ColorStop], t: f32) -> [f32; 3] {
    let k = clampf(t, 0.0, 1.0);
    for i in 1..stops.len() {
        let a = &stops[i - 1];
        let b = &stops[i];
        if k <= b.t {
            let local = (k - a.t) / (b.t - a.t).max(1e-6);
            return [
                a.r + (b.r - a.r) * local,
                a.g + (b.g - a.g) * local,
                a.b + (b.b - a.b) * local,
            ];
        }
    }
    let last = &stops[stops.len() - 1];
    [last.r, last.g, last.b]
}

fn cs(t: f32, r: f32, g: f32, b: f32) -> ColorStop {
    ColorStop { t, r, g, b }
}

fn ocean_stops() -> Vec<ColorStop> {
    vec![
        cs(0.0,  220.0, 236.0, 250.0),
        cs(0.14, 188.0, 216.0, 243.0),
        cs(0.3,  150.0, 190.0, 232.0),
        cs(0.5,  108.0, 156.0, 218.0),
        cs(0.72, 70.0,  120.0, 195.0),
        cs(0.88, 46.0,  87.0,  169.0),
        cs(1.0,  30.0,  61.0,  143.0),
    ]
}

fn land_stops() -> Vec<ColorStop> {
    vec![
        cs(0.0,    4.0,   104.0, 64.0),
        cs(0.118,  36.0,  129.0, 53.0),
        cs(0.294,  215.0, 179.0, 95.0),
        cs(0.471,  147.0, 51.0,  10.0),
        cs(0.706,  99.0,  96.0,  94.0),
        cs(0.824,  219.0, 218.0, 218.0),
        cs(0.882,  253.0, 253.0, 251.0),
        cs(1.0,    247.0, 246.0, 244.0),
    ]
}

fn sat_ocean_stops() -> Vec<ColorStop> {
    vec![
        cs(0.0,  110.0, 160.0, 190.0),
        cs(0.08, 55.0,  105.0, 155.0),
        cs(0.25, 30.0,  72.0,  130.0),
        cs(0.50, 18.0,  52.0,  108.0),
        cs(0.80, 10.0,  35.0,  82.0),
        cs(1.0,  8.0,   25.0,  65.0),
    ]
}

// ---------------------------------------------------------------------------
//  Hillshade
// ---------------------------------------------------------------------------

fn hillshade(height_map: &[f32], w: usize, h: usize, x: usize, y: usize, wrap_x: bool) -> f32 {
    let (y_up, x_for_up) = if y == 0 {
        (0, if wrap_x { (x + w / 2) % w } else { x })
    } else {
        (y - 1, x)
    };
    let (y_down, x_for_down) = if y == h - 1 {
        (h - 1, if wrap_x { (x + w / 2) % w } else { x })
    } else {
        (y + 1, x)
    };
    let x_left = if wrap_x { (x + w - 1) % w } else { x.saturating_sub(1) };
    let x_right = if wrap_x { (x + 1) % w } else { (x + 1).min(w - 1) };

    let left = height_map[y * w + x_left];
    let right = height_map[y * w + x_right];
    let up = height_map[y_up * w + x_for_up];
    let down = height_map[y_down * w + x_for_down];

    let dzdx = (right - left) * 0.5;
    let dzdy = (down - up) * 0.5;
    let k = 1.0 / 1800.0;
    let mut nx = -dzdx * k;
    let mut ny = -dzdy * k;
    let mut nz = 1.0f32;
    let n_len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-9);
    nx /= n_len;
    ny /= n_len;
    nz /= n_len;

    let (lx, ly, lz) = (-0.45f32, -0.6f32, 0.66f32);
    let l_len = (lx * lx + ly * ly + lz * lz).sqrt();
    let dot = clampf((nx * lx + ny * ly + nz * lz) / l_len, 0.0, 1.0);
    clampf(0.9 + (dot - 0.55) * 0.2, 0.82, 1.04)
}

// ---------------------------------------------------------------------------
//  Land tone range (percentile stretch)
// ---------------------------------------------------------------------------

fn estimate_land_tone_range(height_map: &[f32], max_h: f32) -> (f32, f32) {
    if max_h <= 0.0 { return (0.0, 1.0); }
    let bins = 2048usize;
    let mut hist = vec![0u32; bins];
    let denom = max_h.max(1.0);
    let mut land_count = 0u64;
    for &h in height_map {
        if h <= 0.0 { continue; }
        let t = clampf(h / denom, 0.0, 1.0);
        let bin = ((t * (bins as f32 - 1.0)) as usize).min(bins - 1);
        hist[bin] += 1;
        land_count += 1;
    }
    if land_count < 16 { return (0.0, denom); }
    let q_low = (land_count as f64 * 0.02) as u64;
    let q_high = (land_count as f64 * 0.98) as u64;
    let mut acc = 0u64;
    let mut low_bin = 0;
    for i in 0..bins {
        acc += hist[i] as u64;
        if acc >= q_low { low_bin = i; break; }
    }
    acc = 0;
    let mut high_bin = bins - 1;
    for i in 0..bins {
        acc += hist[i] as u64;
        if acc >= q_high { high_bin = i; break; }
    }
    let min_ref = (low_bin as f32 / (bins as f32 - 1.0)) * denom;
    let max_ref = (high_bin as f32 / (bins as f32 - 1.0)) * denom;
    if max_ref - min_ref < 120.0 { (0.0, denom) } else { (min_ref, max_ref) }
}

// ---------------------------------------------------------------------------
//  Pixel color functions
// ---------------------------------------------------------------------------

fn height_color(h: f32, min_h: f32, max_h: f32, land_min: f32, land_max: f32,
                ocean_st: &[ColorStop], land_st: &[ColorStop]) -> [f32; 3] {
    if h < 0.0 && max_h > 0.0 {
        let t = clampf(-h / (-min_h).max(1.0), 0.0, 1.0).powf(0.68);
        return sample_stops(ocean_st, t);
    }
    if max_h <= 0.0 {
        return sample_stops(ocean_st, 1.0);
    }
    let t_linear = clampf((h - land_min) / (land_max - land_min).max(1.0), 0.0, 1.0);
    sample_stops(land_st, t_linear.powf(0.72))
}

fn biome_color(id: u8) -> [u8; 3] {
    match id {
        0  => [17, 42, 82],
        1  => [204, 213, 238],
        2  => [63, 104, 61],
        3  => [21, 109, 61],
        4  => [196, 168, 84],
        5  => [168, 142, 60],
        6  => [1, 87, 50],
        7  => [132, 173, 93],
        8  => [219, 179, 94],
        9  => [34, 139, 87],
        10 => [130, 106, 74],
        11 => [178, 162, 108],
        _  => [128, 128, 128],
    }
}

fn plate_color(id: i16) -> [u8; 3] {
    let id = id.max(0) as f32;
    let hue = ((id * 37.0) % 360.0) * (std::f32::consts::PI / 180.0);
    [
        (127.0 + 100.0 * (hue).sin()).round() as u8,
        (127.0 + 100.0 * (hue + 2.094).sin()).round() as u8,
        (127.0 + 100.0 * (hue + 4.188).sin()).round() as u8,
    ]
}

fn satellite_color(
    i: usize, x: usize, y: usize,
    height_map: &[f32], biome_map: &[u8], temp_map: &[f32], precip_map: &[f32],
    river_map: &[f32], lake_map: &[u8], slope_map: &[f32],
    w: usize, h: usize, min_h: f32, _max_h: f32, wrap_x: bool,
    sat_ocean_st: &[ColorStop],
) -> [u8; 3] {
    let hv = height_map[i];

    // --- Ocean ---
    if hv < 0.0 {
        let depth_t = clampf(-hv / (-min_h).max(1.0), 0.0, 1.0).powf(0.55);
        let mut base = sample_stops(sat_ocean_st, depth_t);
        if depth_t < 0.08 {
            let sm = 1.0 - depth_t / 0.08;
            base[0] += (80.0 - base[0]) * sm * 0.35;
            base[1] += (155.0 - base[1]) * sm * 0.35;
            base[2] += (150.0 - base[2]) * sm * 0.2;
        }
        let o_shade = hillshade(height_map, w, h, x, y, wrap_x);
        let o_factor = clampf(0.92 + (o_shade - 0.82) * 0.6, 0.88, 1.06);
        return [clampu8(base[0] * o_factor), clampu8(base[1] * o_factor), clampu8(base[2] * o_factor)];
    }

    // --- Land ---
    let precip = precip_map[i];
    let temp = temp_map[i];
    let biome = biome_map[i];

    let veg_t = clampf((temp - (-5.0)) / 30.0, 0.0, 1.0);
    let veg_p = clampf(precip / 900.0, 0.0, 1.0);
    let ndvi = (veg_t * veg_p).powf(0.65);

    let (dry_r, dry_g, dry_b) = (168.0, 152.0, 118.0);
    let (lush_r, lush_g, lush_b) = (15.0, 85.0, 25.0);

    let v = ndvi.powf(0.7);
    let mut r = dry_r + (lush_r - dry_r) * v;
    let mut g = dry_g + (lush_g - dry_g) * v;
    let mut b = dry_b + (lush_b - dry_b) * v;

    // Desert override
    if biome == 8 {
        let dm = 0.6;
        r += (195.0 - r) * dm;
        g += (172.0 - g) * dm;
        b += (130.0 - b) * dm;
    }
    // Tropical rainforest
    if biome == 6 && ndvi > 0.4 {
        let dm = 0.3;
        r += (8.0 - r) * dm;
        g += (55.0 - g) * dm;
        b += (22.0 - b) * dm;
    }

    // Elevation → rock
    let rock_t = clampf((hv - 1200.0) / (4500.0 - 1200.0), 0.0, 1.0);
    if rock_t > 0.0 {
        let slope = slope_map[i];
        let steepness = clampf(slope / 0.8, 0.0, 1.0);
        let rock_r = 140.0 - steepness * 35.0;
        let rock_g = 125.0 - steepness * 30.0;
        let rock_b = 105.0 - steepness * 25.0;
        let treeline = clampf(rock_t * 1.5, 0.0, 1.0);
        let veg_rem = (1.0 - treeline) * ndvi;
        let rock_mix = rock_t * (1.0 - veg_rem * 0.5);
        r += (rock_r - r) * rock_mix;
        g += (rock_g - g) * rock_mix;
        b += (rock_b - b) * rock_mix;
    }

    // Snow/ice
    let lat_deg = 90.0 - (y as f32 + 0.5) * (180.0 / h as f32);
    let abs_lat = lat_deg.abs();
    let snowline = 5200.0 - 62.0 * abs_lat;
    let snow_elev = clampf((hv - snowline) / 1200.0, 0.0, 1.0);
    let polar_snow = if biome == 1 { clampf((abs_lat - 60.0) / 20.0, 0.0, 0.7) } else { 0.0 };
    let cold_peak = clampf((-temp - 10.0) / 20.0, 0.0, 0.4) * clampf(hv / 2000.0, 0.0, 1.0);
    let snow_amt = clampf(snow_elev.max(polar_snow).max(cold_peak), 0.0, 0.95);
    if snow_amt > 0.0 {
        let sr = 235.0 + snow_amt * 10.0;
        let sg = 238.0 + snow_amt * 8.0;
        let sb = 245.0 + snow_amt * 5.0;
        r += (sr - r) * snow_amt;
        g += (sg - g) * snow_amt;
        b += (sb - b) * snow_amt;
    }

    // Hillshade
    let shade = hillshade(height_map, w, h, x, y, wrap_x);
    let sat_shade = clampf(0.62 + (shade - 0.82) * 2.0, 0.55, 1.15);
    r *= sat_shade;
    g *= sat_shade;
    b *= sat_shade;

    // Rivers & lakes (after hillshade)
    let river = river_map[i];
    let lake = lake_map[i];
    if lake > 0 {
        let (lr, lg, lb) = if lake == 2 { (50.0, 75.0, 95.0) } else { (35.0, 65.0, 120.0) };
        r = r * 0.15 + lr * 0.85;
        g = g * 0.15 + lg * 0.85;
        b = b * 0.15 + lb * 0.85;
    } else if river > 0.08 {
        let rs = clampf((river - 0.08) / 0.45, 0.0, 0.95);
        r += (25.0 - r) * rs;
        g += (65.0 - g) * rs;
        b += (140.0 - b) * rs;
    }

    // Atmospheric haze
    let haze = clampf((abs_lat - 60.0) / 25.0, 0.0, 0.08);
    if haze > 0.0 {
        r += (195.0 - r) * haze;
        g += (205.0 - g) * haze;
        b += (225.0 - b) * haze;
    }

    [clampu8(r), clampu8(g), clampu8(b)]
}

// ---------------------------------------------------------------------------
//  Image writers
// ---------------------------------------------------------------------------

fn write_bmp(path: &Path, w: usize, h: usize, pixels: &[[u8; 3]]) {
    let row_stride = (w * 3 + 3) & !3;
    let pixel_data_size = row_stride * h;
    let file_size = 54 + pixel_data_size;
    let mut buf = vec![0u8; file_size];

    // BMP header
    buf[0] = b'B'; buf[1] = b'M';
    buf[2..6].copy_from_slice(&(file_size as u32).to_le_bytes());
    buf[10..14].copy_from_slice(&54u32.to_le_bytes());
    buf[14..18].copy_from_slice(&40u32.to_le_bytes());
    buf[18..22].copy_from_slice(&(w as i32).to_le_bytes());
    buf[22..26].copy_from_slice(&(h as i32).to_le_bytes());
    buf[26..28].copy_from_slice(&1u16.to_le_bytes());
    buf[28..30].copy_from_slice(&24u16.to_le_bytes());
    buf[34..38].copy_from_slice(&(pixel_data_size as u32).to_le_bytes());
    buf[38..42].copy_from_slice(&2835i32.to_le_bytes());
    buf[42..46].copy_from_slice(&2835i32.to_le_bytes());

    // Pixel data (bottom-up)
    for y in 0..h {
        let src_y = h - 1 - y;
        let row_off = 54 + y * row_stride;
        for x in 0..w {
            let [r, g, b] = pixels[src_y * w + x];
            let off = row_off + x * 3;
            buf[off] = b;
            buf[off + 1] = g;
            buf[off + 2] = r;
        }
    }

    fs::write(path, &buf).expect("failed to write BMP");
}

fn write_jpg(path: &Path, w: usize, h: usize, pixels: &[[u8; 3]], quality: u8) {
    use image::codecs::jpeg::JpegEncoder;
    let mut rgb_buf = vec![0u8; w * h * 3];
    for (i, &[r, g, b]) in pixels.iter().enumerate() {
        rgb_buf[i * 3] = r;
        rgb_buf[i * 3 + 1] = g;
        rgb_buf[i * 3 + 2] = b;
    }
    let file = fs::File::create(path).expect("failed to create JPG file");
    let mut bw = BufWriter::new(file);
    let mut encoder = JpegEncoder::new_with_quality(&mut bw, quality);
    encoder
        .encode(&rgb_buf, w as u32, h as u32, image::ExtendedColorType::Rgb8)
        .expect("failed to encode JPG");
    bw.flush().ok();
}

// ---------------------------------------------------------------------------
//  Render previews from data arrays
// ---------------------------------------------------------------------------

fn render_previews(
    run_dir: &Path,
    w: usize, h: usize,
    height_map: &[f32], slope_map: &[f32], river_map: &[f32], lake_map: &[u8],
    temp_map: &[f32], precip_map: &[f32], biome_map: &[u8],
    plates: &[i16], boundary_types: &[i8],
    min_h: f32, max_h: f32,
    wrap_x: bool,
) {
    let t0 = Instant::now();
    let size = w * h;

    let ocean_st = ocean_stops();
    let land_st = land_stops();
    let sat_ocean_st = sat_ocean_stops();
    let (land_min, land_max) = estimate_land_tone_range(height_map, max_h);

    // Height preview
    eprint!("  rendering height_preview...");
    let mut px = vec![[0u8; 3]; size];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let base = height_color(height_map[i], min_h, max_h, land_min, land_max, &ocean_st, &land_st);
            if height_map[i] < 0.0 {
                px[i] = [clampu8(base[0]), clampu8(base[1]), clampu8(base[2])];
            } else {
                let shade = hillshade(height_map, w, h, x, y, wrap_x);
                px[i] = [clampu8(base[0] * shade), clampu8(base[1] * shade), clampu8(base[2] * shade)];
            }
        }
    }
    write_bmp(&run_dir.join("height_preview.bmp"), w, h, &px);
    write_jpg(&run_dir.join("height_preview.jpg"), w, h, &px, 92);
    eprintln!(" done");

    // Plates preview
    eprint!("  rendering plates_preview...");
    for i in 0..size {
        let bt = boundary_types[i];
        px[i] = if bt == 1 { [255, 175, 175] }
                else if bt == 2 { [180, 220, 255] }
                else if bt == 3 { [235, 220, 170] }
                else { plate_color(plates[i]) };
    }
    write_bmp(&run_dir.join("plates_preview.bmp"), w, h, &px);
    write_jpg(&run_dir.join("plates_preview.jpg"), w, h, &px, 92);
    eprintln!(" done");

    // Biomes preview
    eprint!("  rendering biomes_preview...");
    for i in 0..size {
        px[i] = biome_color(biome_map[i]);
    }
    write_bmp(&run_dir.join("biomes_preview.bmp"), w, h, &px);
    write_jpg(&run_dir.join("biomes_preview.jpg"), w, h, &px, 92);
    eprintln!(" done");

    // Satellite preview
    eprint!("  rendering satellite_preview...");
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            px[i] = satellite_color(
                i, x, y,
                height_map, biome_map, temp_map, precip_map,
                river_map, lake_map, slope_map,
                w, h, min_h, max_h, wrap_x,
                &sat_ocean_st,
            );
        }
    }
    write_bmp(&run_dir.join("satellite_preview.bmp"), w, h, &px);
    write_jpg(&run_dir.join("satellite_preview.jpg"), w, h, &px, 92);
    eprintln!(" done");

    eprintln!("  previews rendered in {:.1}s", t0.elapsed().as_secs_f64());
}

// ---------------------------------------------------------------------------
//  Render-only mode: read existing data and produce previews
// ---------------------------------------------------------------------------

fn render_only(run_dir: &Path) {
    eprintln!("=== render-only mode ===");
    eprintln!("  source: {}", run_dir.display());

    // Read meta.json for dimensions
    let meta_path = run_dir.join("meta.json");
    let meta_str = fs::read_to_string(&meta_path)
        .unwrap_or_else(|e| panic!("cannot read meta.json: {e}"));
    let meta: serde_json::Value = serde_json::from_str(&meta_str)
        .unwrap_or_else(|e| panic!("cannot parse meta.json: {e}"));
    let w = meta["width"].as_u64().expect("width not found in meta.json") as usize;
    let h = meta["height"].as_u64().expect("height not found in meta.json") as usize;
    let min_h = meta["stats"]["minHeight"].as_f64().unwrap_or(-11000.0) as f32;
    let max_h = meta["stats"]["maxHeight"].as_f64().unwrap_or(8848.0) as f32;

    eprintln!("  grid:   {w}x{h}");

    let height_map = read_f32_array(&run_dir.join("height_map.f32"));
    let slope_map = read_f32_array(&run_dir.join("slope_map.f32"));
    let river_map = read_f32_array(&run_dir.join("river_map.f32"));
    let lake_map = read_u8_array(&run_dir.join("lake_map.u8"));
    let temp_map = read_f32_array(&run_dir.join("temperature_map.f32"));
    let precip_map = read_f32_array(&run_dir.join("precipitation_map.f32"));
    let biome_map = read_u8_array(&run_dir.join("biome_map.u8"));
    let plates = read_i16_array(&run_dir.join("plate_map.i16"));
    let boundary_types = read_i8_array(&run_dir.join("boundary_types.i8"));

    assert_eq!(height_map.len(), w * h, "height_map size mismatch");

    render_previews(
        run_dir, w, h,
        &height_map, &slope_map, &river_map, &lake_map,
        &temp_map, &precip_map, &biome_map,
        &plates, &boundary_types,
        min_h, max_h, true,
    );

    eprintln!("=== done ===");
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

fn main() {
    // Render-only mode: RENDER=path/to/run/dir
    if let Ok(render_dir) = env::var("RENDER") {
        render_only(Path::new(&render_dir));
        return;
    }

    let seed = env_u32("SEED", 1);
    let resolution = env_usize("RESOLUTION", 4096);
    let width = resolution;
    let height = resolution / 2;
    let radius_km = env_f32("RADIUS_KM", 6371.0);
    let ocean_percent = env_f32("OCEAN_PERCENT", 71.0);
    let gravity = env_f32("GRAVITY", 9.8);
    let rotation_hours = env_f32("ROTATION_HOURS", 24.0);
    let axial_tilt = env_f32("AXIAL_TILT", 23.44);
    let plate_count = env_f32("PLATE_COUNT", 12.0) as i32;
    let plate_speed = env_f32("PLATE_SPEED", 5.0);
    let mantle_heat = env_f32("MANTLE_HEAT", 1.0);
    let preset = env_str("PRESET", "balanced");
    let scope = env_str("SCOPE", "planet");

    eprintln!("=== planet_cli ===");
    eprintln!("  seed:       {seed}");
    eprintln!("  resolution: {width}x{height}");
    eprintln!("  radius_km:  {radius_km}");
    eprintln!("  ocean:      {ocean_percent}%");
    eprintln!("  plates:     {plate_count}");
    eprintln!("  preset:     {preset}");
    eprintln!("  scope:      {scope}");

    let mut cfg = SimulationConfig {
        seed,
        planet: PlanetInputs {
            radius_km,
            gravity,
            density: 5515.0,
            rotation_hours,
            axial_tilt_deg: axial_tilt,
            eccentricity: 0.017,
            atmosphere_bar: 1.0,
            ocean_percent,
        },
        tectonics: TectonicInputs {
            plate_count,
            plate_speed_cm_per_year: plate_speed,
            mantle_heat,
        },
        events: Vec::new(),
        generation_preset: Some(preset),
        scope: Some(scope.clone()),
        island_type: None,
        island_scale_km: None,
        continent_scale_km: None,
        planet_width: Some(width),
        planet_height: Some(height),
    };

    let t0 = Instant::now();
    let last_pct = std::cell::Cell::new(-1i32);
    let result = run_simulation_native(&mut cfg, Some(&|pct: f32| {
        let p = pct as i32;
        if p != last_pct.get() {
            last_pct.set(p);
            eprint!("\r  progress: {p:3}%");
        }
    }));
    eprintln!("\r  progress: 100% — done in {:.1}s", t0.elapsed().as_secs_f64());

    // Create output directory
    let stamp = chrono_stamp();
    let dir_name = format!("{stamp}_seed-{seed}");
    let run_dir = PathBuf::from("generations/runs").join(&dir_name);
    fs::create_dir_all(&run_dir).expect("failed to create output dir");

    // Write meta.json
    let meta = serde_json::json!({
        "seed": result.seed(),
        "width": result.width(),
        "height": result.height(),
        "seaLevel": result.sea_level(),
        "radiusKm": result.radius_km(),
        "oceanPercent": result.ocean_percent(),
        "recomputedLayers": result.recomputed_layers_list(),
        "stats": {
            "minHeight": result.min_height(),
            "maxHeight": result.max_height(),
            "minSlope": result.min_slope(),
            "maxSlope": result.max_slope(),
            "minTemperature": result.min_temperature(),
            "maxTemperature": result.max_temperature(),
            "minPrecipitation": result.min_precipitation(),
            "maxPrecipitation": result.max_precipitation(),
        }
    });
    let meta_str = serde_json::to_string_pretty(&meta).unwrap();
    fs::write(run_dir.join("meta.json"), meta_str).expect("failed to write meta.json");

    // Write binary arrays
    let height_map = result.height_map();
    let slope_map = result.slope_map();
    let river_map = result.river_map();
    let lake_map = result.lake_map();
    let temp_map = result.temperature_map();
    let precip_map = result.precipitation_map();
    let biome_map = result.biome_map();
    let plates = result.plates();
    let boundary_types = result.boundary_types();

    write_f32_array(&run_dir.join("height_map.f32"), &height_map);
    write_f32_array(&run_dir.join("slope_map.f32"), &slope_map);
    write_f32_array(&run_dir.join("river_map.f32"), &river_map);
    fs::write(run_dir.join("lake_map.u8"), &lake_map).unwrap();
    write_i32_array(&run_dir.join("flow_direction.i32"), &result.flow_direction());
    write_f32_array(&run_dir.join("flow_accumulation.f32"), &result.flow_accumulation());
    write_f32_array(&run_dir.join("temperature_map.f32"), &temp_map);
    write_f32_array(&run_dir.join("precipitation_map.f32"), &precip_map);
    fs::write(run_dir.join("biome_map.u8"), &biome_map).unwrap();
    write_f32_array(&run_dir.join("settlement_map.f32"), &result.settlement_map());
    write_i16_array(&run_dir.join("plate_map.i16"), &plates);
    write_i8_array(&run_dir.join("boundary_types.i8"), &boundary_types);

    eprintln!("  output:   {}", run_dir.display());

    // Render preview images
    let wrap_x = scope != "island";
    let boundary_i8: Vec<i8> = boundary_types;
    render_previews(
        &run_dir, width, height,
        &height_map, &slope_map, &river_map, &lake_map,
        &temp_map, &precip_map, &biome_map,
        &plates, &boundary_i8,
        result.min_height(), result.max_height(),
        wrap_x,
    );

    eprintln!("=== done ===");
}

fn chrono_stamp() -> String {
    use std::time::SystemTime;
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    let secs = d.as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    let (y, mo, d) = days_to_ymd(days);
    format!("{y:04}{mo:02}{d:02}_{h:02}{m:02}{s:02}")
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    days += 719468;
    let era = days / 146097;
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
