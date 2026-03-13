//! Draw latitude + longitude grid on an equirectangular map image.
//!
//! Usage:
//!     cargo run --features cli --bin add_latitudes --release -- <image_path>
//!
//! Output: <name>_grid.jpg next to the original.

use image::{GenericImageView, RgbImage, Rgb};
use std::path::PathBuf;

const LATITUDES: &[(f64, &str)] = &[
    (66.5,  "66.5N"),
    (60.0,  "60N"),
    (30.0,  "30N"),
    (23.5,  "23.5N"),
    (0.0,   "0 eq"),
    (-23.5, "23.5S"),
    (-30.0, "30S"),
    (-60.0, "60S"),
    (-66.5, "66.5S"),
];

// Meridians every 30 degrees (0..360 mapped to 0..width)
const MERIDIAN_STEP: f64 = 30.0;

fn lat_to_y(lat: f64, height: u32) -> i32 {
    (height as f64 / 2.0 - (lat / 180.0) * height as f64) as i32
}

fn lon_to_x(lon: f64, width: u32) -> i32 {
    ((lon / 360.0) * width as f64) as i32
}

fn blend_color(base: Rgb<u8>, line: Rgb<u8>, alpha: f32) -> Rgb<u8> {
    let r = base[0] as f32 * (1.0 - alpha) + line[0] as f32 * alpha;
    let g = base[1] as f32 * (1.0 - alpha) + line[1] as f32 * alpha;
    let b = base[2] as f32 * (1.0 - alpha) + line[2] as f32 * alpha;
    Rgb([r as u8, g as u8, b as u8])
}

fn draw_char(img: &mut RgbImage, ch: char, cx: u32, cy: u32, color: Rgb<u8>, scale: u32) {
    let glyph = match ch {
        '0' => [0x7C, 0xC6, 0xCE, 0xD6, 0xE6, 0xC6, 0x7C],
        '1' => [0x30, 0x70, 0x30, 0x30, 0x30, 0x30, 0xFC],
        '2' => [0x78, 0xCC, 0x0C, 0x38, 0x60, 0xCC, 0xFC],
        '3' => [0x78, 0xCC, 0x0C, 0x38, 0x0C, 0xCC, 0x78],
        '4' => [0x1C, 0x3C, 0x6C, 0xCC, 0xFE, 0x0C, 0x0C],
        '5' => [0xFC, 0xC0, 0xF8, 0x0C, 0x0C, 0xCC, 0x78],
        '6' => [0x38, 0x60, 0xC0, 0xF8, 0xCC, 0xCC, 0x78],
        '7' => [0xFC, 0xCC, 0x0C, 0x18, 0x30, 0x30, 0x30],
        '8' => [0x78, 0xCC, 0xCC, 0x78, 0xCC, 0xCC, 0x78],
        '9' => [0x78, 0xCC, 0xCC, 0x7C, 0x0C, 0x18, 0x70],
        'N' => [0xC6, 0xE6, 0xF6, 0xDE, 0xCE, 0xC6, 0xC6],
        'S' => [0x78, 0xCC, 0xC0, 0x78, 0x0C, 0xCC, 0x78],
        'E' => [0xFC, 0xC0, 0xC0, 0xF8, 0xC0, 0xC0, 0xFC],
        'W' => [0xC6, 0xC6, 0xC6, 0xD6, 0xFE, 0xEE, 0xC6],
        'e' => [0x00, 0x00, 0x78, 0xCC, 0xFC, 0xC0, 0x78],
        'q' => [0x00, 0x00, 0x76, 0xCC, 0xCC, 0x7C, 0x0C],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x30],
        ' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        _ => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    };
    let (w, h) = img.dimensions();
    for (row, &bits) in glyph.iter().enumerate() {
        for col in 0..8u32 {
            if bits & (0x80 >> col) != 0 {
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = cx + col * scale + sx;
                        let py = cy + row as u32 * scale + sy;
                        if px < w && py < h {
                            img.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }
    }
}

fn draw_text(img: &mut RgbImage, text: &str, x: u32, y: u32, color: Rgb<u8>, scale: u32) {
    let char_w = 8 * scale + scale;
    for (i, ch) in text.chars().enumerate() {
        draw_char(img, ch, x + i as u32 * char_w, y, color, scale);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: add_latitudes <image_path>");
        std::process::exit(1);
    }

    let src = PathBuf::from(&args[1]);
    let img = image::open(&src).unwrap_or_else(|e| {
        eprintln!("Failed to open {}: {e}", src.display());
        std::process::exit(1);
    });

    let (w, h) = img.dimensions();
    let mut out = img.to_rgb8();

    let font_scale = (h / 1024).max(1);
    let dash_len = (w / 200).max(4);
    let gap_len = dash_len;
    let line_color = Rgb([255, 255, 255]);
    let line_alpha = 0.45_f32;

    // --- Draw latitude lines (horizontal) ---
    for &(lat, label) in LATITUDES {
        let y = lat_to_y(lat, h);
        if y < 0 || y >= h as i32 {
            continue;
        }
        let y = y as u32;

        let mut x = 0u32;
        while x < w {
            let x2 = (x + dash_len).min(w);
            for px in x..x2 {
                let base = Rgb(out.get_pixel(px, y).0);
                out.put_pixel(px, y, blend_color(base, line_color, line_alpha));
            }
            x = x2 + gap_len;
        }

        let ty = if y > 7 * font_scale + 4 {
            y - 7 * font_scale - 4
        } else {
            y + 4
        };
        draw_text(&mut out, label, 13, ty + 1, Rgb([0, 0, 0]), font_scale);
        draw_text(&mut out, label, 12, ty, Rgb([240, 240, 240]), font_scale);
    }

    // --- Draw meridians (vertical) every 30° ---
    let mut lon = MERIDIAN_STEP;
    while lon < 360.0 {
        let x = lon_to_x(lon, w);
        if x <= 0 || x >= w as i32 {
            lon += MERIDIAN_STEP;
            continue;
        }
        let x = x as u32;

        // Dashed vertical line
        let mut y = 0u32;
        while y < h {
            let y2 = (y + dash_len).min(h);
            for py in y..y2 {
                let base = Rgb(out.get_pixel(x, py).0);
                out.put_pixel(x, py, blend_color(base, line_color, line_alpha));
            }
            y = y2 + gap_len;
        }

        // Label near the top
        let deg = lon as i32;
        let label = format!("{}E", deg);
        let lx = x + 4;
        let ly = 12_u32;
        draw_text(&mut out, &label, lx + 1, ly + 1, Rgb([0, 0, 0]), font_scale);
        draw_text(&mut out, &label, lx, ly, Rgb([240, 240, 240]), font_scale);

        lon += MERIDIAN_STEP;
    }

    let stem = src.file_stem().unwrap().to_str().unwrap();
    let out_path = src.parent().unwrap().join(format!("{stem}_grid.jpg"));
    out.save(&out_path).unwrap_or_else(|e| {
        eprintln!("Failed to save: {e}");
        std::process::exit(1);
    });
    eprintln!("Saved: {}", out_path.display());
}
