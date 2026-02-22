use js_sys::Array;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

const WORLD_WIDTH: usize = 2048;
const WORLD_HEIGHT: usize = 1024;
const WORLD_SIZE: usize = WORLD_WIDTH * WORLD_HEIGHT;
const RADIANS: f32 = std::f32::consts::PI / 180.0;
const KILOMETERS_PER_DEGREE: f32 = 111.319_490_793;

#[derive(Clone)]
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state as f32 / 4_294_967_296.0
    }
}

#[inline]
fn clampf(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-6 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = clampf((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn ridge(value: f32) -> f32 {
    (1.0 - value.abs()).max(0.0)
}

#[inline]
fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn hash_u32(mut v: u32) -> u32 {
    v ^= v >> 16;
    v = v.wrapping_mul(0x7FEB_352D);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846C_A68B);
    v ^= v >> 16;
    v
}

#[inline]
fn hash3(seed: u32, x: i32, y: i32, z: i32) -> u32 {
    let hx = (x as u32).wrapping_mul(0x8DA6_B343);
    let hy = (y as u32).wrapping_mul(0xD816_3841);
    let hz = (z as u32).wrapping_mul(0xCB1A_B31F);
    hash_u32(seed ^ hx ^ hy ^ hz)
}

#[inline]
fn hash_to_unit(h: u32) -> f32 {
    (h as f32 / 4_294_967_295.0) * 2.0 - 1.0
}

#[inline]
fn value_noise3(x: f32, y: f32, z: f32, seed: u32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let z0 = z.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let z1 = z0 + 1;

    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let tz = z - z0 as f32;
    let fx = tx * tx * (3.0 - 2.0 * tx);
    let fy = ty * ty * (3.0 - 2.0 * ty);
    let fz = tz * tz * (3.0 - 2.0 * tz);

    let v000 = hash_to_unit(hash3(seed, x0, y0, z0));
    let v100 = hash_to_unit(hash3(seed, x1, y0, z0));
    let v010 = hash_to_unit(hash3(seed, x0, y1, z0));
    let v110 = hash_to_unit(hash3(seed, x1, y1, z0));
    let v001 = hash_to_unit(hash3(seed, x0, y0, z1));
    let v101 = hash_to_unit(hash3(seed, x1, y0, z1));
    let v011 = hash_to_unit(hash3(seed, x0, y1, z1));
    let v111 = hash_to_unit(hash3(seed, x1, y1, z1));

    let x00 = lerpf(v000, v100, fx);
    let x10 = lerpf(v010, v110, fx);
    let x01 = lerpf(v001, v101, fx);
    let x11 = lerpf(v011, v111, fx);
    let y0v = lerpf(x00, x10, fy);
    let y1v = lerpf(x01, x11, fy);
    lerpf(y0v, y1v, fz)
}

#[inline]
fn spherical_fbm(sx: f32, sy: f32, sz: f32, seed_phase: f32) -> f32 {
    let mut freq = 2.25_f32;
    let mut amp = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut norm = 0.0_f32;
    let mut x = sx;
    let mut y = sy;
    let mut z = sz;
    let seed_base = hash_u32(seed_phase.to_bits() ^ 0x9E37_79B9);

    for octave in 0..5 {
        let octave_seed = hash_u32(seed_base.wrapping_add((octave as u32) * 0x85EB_CA6B));
        let px = x * freq + seed_phase * 0.71 + octave as f32 * 17.0;
        let py = y * freq - seed_phase * 0.53 + octave as f32 * 11.0;
        let pz = z * freq + seed_phase * 0.37 + octave as f32 * 7.0;
        let n = value_noise3(px, py, pz, octave_seed);
        sum += n * amp;
        norm += amp;
        let rx = x * 0.82 - y * 0.46 + z * 0.33;
        let ry = x * 0.51 + y * 0.79 - z * 0.28;
        let rz = -x * 0.24 + y * 0.41 + z * 0.88;
        x = rx;
        y = ry;
        z = rz;
        freq *= 2.03;
        amp *= 0.52;
    }

    sum / norm.max(1e-6)
}

#[inline]
fn index(x: usize, y: usize) -> usize {
    y * WORLD_WIDTH + x
}

#[inline]
fn random_range(rng: &mut Rng, min: f32, max: f32) -> f32 {
    min + rng.next_f32() * (max - min)
}

#[inline]
fn spherical_wrap(mut x: i32, mut y: i32) -> (usize, usize) {
    while y < 0 || y >= WORLD_HEIGHT as i32 {
        if y < 0 {
            y = -y - 1;
            x += (WORLD_WIDTH / 2) as i32;
        } else {
            y = 2 * WORLD_HEIGHT as i32 - y - 1;
            x += (WORLD_WIDTH / 2) as i32;
        }
    }

    let mut wrapped_x = x % WORLD_WIDTH as i32;
    if wrapped_x < 0 {
        wrapped_x += WORLD_WIDTH as i32;
    }
    let wrapped_y = y.clamp(0, WORLD_HEIGHT as i32 - 1);
    (wrapped_x as usize, wrapped_y as usize)
}

#[inline]
fn index_spherical(x: i32, y: i32) -> usize {
    let (sx, sy) = spherical_wrap(x, y);
    index(sx, sy)
}

fn quantile_sorted(sorted_values: &[f32], q: f32) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let t = clampf(q, 0.0, 1.0) * (sorted_values.len() as f32 - 1.0);
    let lo = t.floor() as usize;
    let hi = (lo + 1).min(sorted_values.len() - 1);
    let k = t - lo as f32;
    sorted_values[lo] * (1.0 - k) + sorted_values[hi] * k
}

fn min_max(values: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = -f32::INFINITY;
    for &v in values {
        min = min.min(v);
        max = max.max(v);
    }
    (min, max)
}

struct WorldCache {
    lat_by_y: Vec<f32>,
    lon_by_x: Vec<f32>,
    x_by_cell: Vec<f32>,
    y_by_cell: Vec<f32>,
    z_by_cell: Vec<f32>,
}

static WORLD_CACHE: OnceLock<WorldCache> = OnceLock::new();

fn world_cache() -> &'static WorldCache {
    WORLD_CACHE.get_or_init(|| {
        let mut lat_by_y = vec![0.0_f32; WORLD_SIZE];
        let mut lon_by_x = vec![0.0_f32; WORLD_SIZE];
        let mut x_by_cell = vec![0.0_f32; WORLD_SIZE];
        let mut y_by_cell = vec![0.0_f32; WORLD_SIZE];
        let mut z_by_cell = vec![0.0_f32; WORLD_SIZE];

        for y in 0..WORLD_HEIGHT {
            let lat_deg = 90.0 - (y as f32 + 0.5) * (180.0 / WORLD_HEIGHT as f32);
            let lat_rad = lat_deg * RADIANS;
            let cos_lat = lat_rad.cos();
            for x in 0..WORLD_WIDTH {
                let lon_deg = (x as f32 + 0.5) * (360.0 / WORLD_WIDTH as f32) - 180.0;
                let lon_rad = lon_deg * RADIANS;
                let i = index(x, y);
                lat_by_y[i] = lat_deg;
                lon_by_x[i] = lon_deg;
                x_by_cell[i] = cos_lat * lon_rad.cos();
                y_by_cell[i] = cos_lat * lon_rad.sin();
                z_by_cell[i] = lat_rad.sin();
            }
        }

        WorldCache {
            lat_by_y,
            lon_by_x,
            x_by_cell,
            y_by_cell,
            z_by_cell,
        }
    })
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PlanetInputs {
    pub radius_km: f32,
    pub gravity: f32,
    pub density: f32,
    pub rotation_hours: f32,
    pub axial_tilt_deg: f32,
    pub eccentricity: f32,
    pub atmosphere_bar: f32,
    pub ocean_percent: f32,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TectonicInputs {
    pub plate_count: i32,
    pub plate_speed_cm_per_year: f32,
    pub mantle_heat: f32,
}

#[derive(Deserialize, Serialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct WorldEventRecord {
    pub kind: String,
    pub latitude: f32,
    pub longitude: f32,
    #[serde(default)]
    pub diameter_km: f32,
    #[serde(default)]
    pub speed_kms: f32,
    #[serde(default)]
    pub angle_deg: f32,
    #[serde(default)]
    pub density_kg_m3: f32,
    #[serde(default)]
    pub radius_km: f32,
    #[serde(default)]
    pub magnitude: f32,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub energy_joule: Option<f64>,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SimulationConfig {
    pub seed: u32,
    pub planet: PlanetInputs,
    pub tectonics: TectonicInputs,
    pub events: Vec<WorldEventRecord>,
    #[serde(default)]
    pub generation_preset: Option<String>,
}

#[derive(Clone, Copy)]
enum RecomputeReason {
    Global,
    Tectonics,
    Events,
}

#[derive(Clone, Copy)]
enum GenerationPreset {
    Ultra,
    Fast,
    Balanced,
    Detailed,
}

#[derive(Clone, Copy)]
struct DetailProfile {
    buoyancy_smooth_passes: usize,
    erosion_rounds: usize,
    ocean_smooth_passes: usize,
    max_kernel_radius: i32,
    fault_iterations: usize,
    fault_smooth_passes: usize,
}

fn parse_generation_preset(value: Option<&str>) -> GenerationPreset {
    match value.unwrap_or("balanced") {
        "ultra" => GenerationPreset::Ultra,
        "fast" => GenerationPreset::Fast,
        "detailed" => GenerationPreset::Detailed,
        _ => GenerationPreset::Balanced,
    }
}

fn detail_profile(preset: GenerationPreset) -> DetailProfile {
    match preset {
        GenerationPreset::Ultra => DetailProfile {
            buoyancy_smooth_passes: 1,
            erosion_rounds: 0,
            ocean_smooth_passes: 0,
            max_kernel_radius: 3,
            fault_iterations: 720,
            fault_smooth_passes: 1,
        },
        GenerationPreset::Fast => DetailProfile {
            buoyancy_smooth_passes: 6,
            erosion_rounds: 1,
            ocean_smooth_passes: 1,
            max_kernel_radius: 4,
            fault_iterations: 1300,
            fault_smooth_passes: 1,
        },
        GenerationPreset::Detailed => DetailProfile {
            buoyancy_smooth_passes: 18,
            erosion_rounds: 2,
            ocean_smooth_passes: 3,
            max_kernel_radius: 6,
            fault_iterations: 3400,
            fault_smooth_passes: 2,
        },
        GenerationPreset::Balanced => DetailProfile {
            buoyancy_smooth_passes: 14,
            erosion_rounds: 2,
            ocean_smooth_passes: 2,
            max_kernel_radius: 5,
            fault_iterations: 2600,
            fault_smooth_passes: 2,
        },
    }
}

fn parse_reason(reason: &str) -> RecomputeReason {
    match reason {
        "tectonics" => RecomputeReason::Tectonics,
        "events" => RecomputeReason::Events,
        _ => RecomputeReason::Global,
    }
}

fn evaluate_recompute(reason: RecomputeReason) -> Vec<String> {
    let all = [
        "planet",
        "plates",
        "relief",
        "hydrology",
        "climate",
        "biomes",
        "settlement",
    ];

    let active: &[&str] = match reason {
        RecomputeReason::Global => &all,
        RecomputeReason::Tectonics => &all,
        RecomputeReason::Events => &["relief", "hydrology", "climate", "biomes", "settlement"],
    };

    all.iter()
        .filter(|id| active.contains(id))
        .map(|id| (*id).to_string())
        .collect()
}

fn ensure_event_energy(mut event: WorldEventRecord) -> WorldEventRecord {
    if event.kind == "meteorite" && event.energy_joule.is_none() {
        let r = (event.diameter_km * 1000.0) as f64 / 2.0;
        let mass = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3) * event.density_kg_m3 as f64;
        event.energy_joule = Some(0.5 * mass * (event.speed_kms as f64 * 1000.0).powi(2));
    }
    event
}

#[derive(Clone)]
struct PlateSpec {
    lat: f32,
    lon: f32,
    speed: f32,
    dir_x: f32,
    dir_y: f32,
    heat: f32,
    buoyancy: f32,
}

#[derive(Clone)]
struct PlateVector {
    speed: f32,
    dir_x: f32,
    dir_y: f32,
    heat: f32,
    buoyancy: f32,
}

#[derive(Clone)]
struct ComputePlatesResult {
    plate_field: Vec<i16>,
    boundary_types: Vec<i8>,
    boundary_normal_x: Vec<f32>,
    boundary_normal_y: Vec<f32>,
    boundary_strength: Vec<f32>,
    plate_vectors: Vec<PlateVector>,
}

#[derive(Clone)]
struct ReliefResult {
    relief: Vec<f32>,
    sea_level: f32,
}

#[derive(Clone, Copy, Debug)]
struct FrontierNode {
    cost: f32,
    index: usize,
    plate: i16,
}

impl PartialEq for FrontierNode {
    fn eq(&self, other: &Self) -> bool {
        self.cost.to_bits() == other.cost.to_bits()
            && self.index == other.index
            && self.plate == other.plate
    }
}

impl Eq for FrontierNode {}

impl PartialOrd for FrontierNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FrontierNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost
            .total_cmp(&other.cost)
            .then_with(|| self.index.cmp(&other.index))
            .then_with(|| self.plate.cmp(&other.plate))
    }
}

struct MinCostQueue {
    data: BinaryHeap<Reverse<FrontierNode>>,
}

impl MinCostQueue {
    fn new() -> Self {
        Self {
            data: BinaryHeap::new(),
        }
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn push(&mut self, item: FrontierNode) {
        self.data.push(Reverse(item));
    }

    fn pop(&mut self) -> Option<FrontierNode> {
        self.data.pop().map(|Reverse(node)| node)
    }
}

#[derive(Clone, Copy)]
struct GrowthParam {
    drift_x: f32,
    drift_y: f32,
    spread: f32,
    roughness: f32,
    freq_a: f32,
    freq_b: f32,
    freq_c: f32,
    freq_d: f32,
    phase_a: f32,
    phase_b: f32,
}

fn lat_lon_to_index(lat: f32, lon: f32) -> usize {
    let y = clampf(
        ((90.0 - lat) / (180.0 / WORLD_HEIGHT as f32)).round(),
        0.0,
        (WORLD_HEIGHT - 1) as f32,
    ) as usize;
    let x = ((((lon + 180.0) / (360.0 / WORLD_WIDTH as f32)).round() as i32 % WORLD_WIDTH as i32)
        + WORLD_WIDTH as i32)
        % WORLD_WIDTH as i32;
    index(x as usize, y)
}

fn nearest_free_index(start: usize, occupied: &[u8]) -> usize {
    if occupied[start] == 0 {
        return start;
    }

    let sx = start % WORLD_WIDTH;
    let sy = start / WORLD_WIDTH;
    let max_radius = WORLD_WIDTH.max(WORLD_HEIGHT) as i32;

    for radius in 1..max_radius {
        for dy in -radius..=radius {
            let y = sy as i32 + dy;
            if y < 0 || y >= WORLD_HEIGHT as i32 {
                continue;
            }
            let span = radius - dy.abs();
            let candidates = [sx as i32 - span, sx as i32 + span];
            for cx in candidates {
                let mut x = cx % WORLD_WIDTH as i32;
                if x < 0 {
                    x += WORLD_WIDTH as i32;
                }
                let idx = index(x as usize, y as usize);
                if occupied[idx] == 0 {
                    return idx;
                }
            }
        }
    }

    start
}

fn build_irregular_plate_field(plates: &[PlateSpec], seed: u32, cache: &WorldCache) -> Vec<i16> {
    let mut plate_field = vec![-1_i16; WORLD_SIZE];
    let mut open_cost = vec![f32::INFINITY; WORLD_SIZE];
    let mut occupied_seeds = vec![0_u8; WORLD_SIZE];
    let mut growth_rng = Rng::new(seed ^ 0x9e37_79b9);
    let mut queue = MinCostQueue::new();

    let growth_params: Vec<GrowthParam> = plates
        .iter()
        .map(|plate| {
            let len = (plate.dir_x.hypot(plate.dir_y)).max(1.0);
            GrowthParam {
                drift_x: plate.dir_x / len,
                drift_y: plate.dir_y / len,
                spread: random_range(&mut growth_rng, 0.85, 1.25),
                roughness: random_range(&mut growth_rng, 0.28, 1.05),
                freq_a: random_range(&mut growth_rng, 0.045, 0.145),
                freq_b: random_range(&mut growth_rng, 0.055, 0.16),
                freq_c: random_range(&mut growth_rng, 0.08, 0.22),
                freq_d: random_range(&mut growth_rng, 0.07, 0.2),
                phase_a: random_range(&mut growth_rng, -std::f32::consts::PI, std::f32::consts::PI),
                phase_b: random_range(&mut growth_rng, -std::f32::consts::PI, std::f32::consts::PI),
            }
        })
        .collect();

    for (plate_id, plate) in plates.iter().enumerate() {
        let seed_index =
            nearest_free_index(lat_lon_to_index(plate.lat, plate.lon), &occupied_seeds);
        occupied_seeds[seed_index] = 1;
        open_cost[seed_index] = 0.0;
        queue.push(FrontierNode {
            cost: 0.0,
            index: seed_index,
            plate: plate_id as i16,
        });
    }

    const STEPS: [(i32, i32, f32); 8] = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, std::f32::consts::SQRT_2),
        (-1, 1, std::f32::consts::SQRT_2),
        (1, -1, std::f32::consts::SQRT_2),
        (-1, -1, std::f32::consts::SQRT_2),
    ];

    let mut assigned = 0usize;
    while queue.size() > 0 && assigned < WORLD_SIZE {
        let Some(node) = queue.pop() else {
            break;
        };

        if node.cost > open_cost[node.index] + 1e-6 {
            continue;
        }
        if plate_field[node.index] != -1 {
            continue;
        }

        plate_field[node.index] = node.plate;
        assigned += 1;

        let x = node.index % WORLD_WIDTH;
        let y = node.index / WORLD_WIDTH;
        let gp = growth_params[node.plate as usize];

        for (dx, dy, w) in STEPS {
            let j = index_spherical(x as i32 + dx, y as i32 + dy);
            if plate_field[j] != -1 {
                continue;
            }

            let lat = cache.lat_by_y[j];
            let sx = cache.x_by_cell[j];
            let sy = cache.y_by_cell[j];
            let sz = cache.z_by_cell[j];
            let scale_a = 0.8 + gp.freq_a * 24.0;
            let scale_b = 0.8 + gp.freq_b * 24.0;
            let scale_c = 0.8 + gp.freq_c * 24.0;
            let scale_d = 0.8 + gp.freq_d * 24.0;
            let wave_a = (sx * scale_a + sy * scale_b + gp.phase_a).sin();
            let wave_b = (sy * scale_c - sz * scale_d + gp.phase_b).cos();
            let rough_factor = 1.0 + gp.roughness * (0.22 * wave_a + 0.18 * wave_b);
            let drift_align = dx as f32 * gp.drift_x + dy as f32 * gp.drift_y;
            let drift_factor = 1.03 - 0.12 * drift_align;
            let polar_factor = 1.0 + (lat.abs() / 90.0) * 0.1;
            let step_cost = (w * gp.spread * rough_factor * drift_factor * polar_factor).max(0.08);
            let next_cost = node.cost + step_cost;

            if next_cost + 1e-6 < open_cost[j] {
                open_cost[j] = next_cost;
                queue.push(FrontierNode {
                    cost: next_cost,
                    index: j,
                    plate: node.plate,
                });
            }
        }
    }

    if assigned < WORLD_SIZE {
        for i in 0..WORLD_SIZE {
            if plate_field[i] != -1 {
                continue;
            }
            let mut best_plate = 0usize;
            let mut best_cost = f32::INFINITY;
            let cell_x = cache.x_by_cell[i];
            let cell_y = cache.y_by_cell[i];
            let cell_z = cache.z_by_cell[i];

            for (p, plate) in plates.iter().enumerate() {
                let lat_r = plate.lat * RADIANS;
                let lon_r = plate.lon * RADIANS;
                let c = lat_r.cos();
                let px = c * lon_r.cos();
                let py = c * lon_r.sin();
                let pz = lat_r.sin();
                let d = 1.0 - (cell_x * px + cell_y * py + cell_z * pz);
                if d < best_cost {
                    best_cost = d;
                    best_plate = p;
                }
            }

            plate_field[i] = best_plate as i16;
        }
    }

    plate_field
}

fn compute_plates(
    _planet: &PlanetInputs,
    tectonics: &TectonicInputs,
    seed: u32,
    cache: &WorldCache,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> ComputePlatesResult {
    let plate_count = tectonics.plate_count.clamp(2, 20) as usize;
    let mut rng = Rng::new(seed.wrapping_add((plate_count as u32).wrapping_mul(7_919)));
    let mut plates: Vec<PlateSpec> = Vec::with_capacity(plate_count);

    for _ in 0..plate_count {
        let lat = random_range(&mut rng, -90.0, 90.0);
        let lon = random_range(&mut rng, -180.0, 180.0);
        let speed =
            (random_range(&mut rng, 0.5, 1.5) * tectonics.plate_speed_cm_per_year).max(0.001);
        let dir = random_range(&mut rng, 0.0, std::f32::consts::PI * 2.0);
        plates.push(PlateSpec {
            lat,
            lon,
            speed,
            dir_x: dir.cos() * speed,
            dir_y: dir.sin() * speed,
            heat: random_range(
                &mut rng,
                (tectonics.mantle_heat * 0.5).max(1.0),
                tectonics.mantle_heat * 1.5,
            ),
            buoyancy: random_range(&mut rng, -1.0, 1.0),
        });
    }

    let plate_field = build_irregular_plate_field(&plates, seed, cache);

    let plate_vectors: Vec<PlateVector> = plates
        .iter()
        .map(|plate| PlateVector {
            speed: plate.speed,
            dir_x: plate.dir_x,
            dir_y: plate.dir_y,
            heat: plate.heat,
            buoyancy: plate.buoyancy,
        })
        .collect();

    let mut boundary_types = vec![0_i8; WORLD_SIZE];
    let mut boundary_normal_x = vec![0.0_f32; WORLD_SIZE];
    let mut boundary_normal_y = vec![0.0_f32; WORLD_SIZE];
    let mut boundary_strength = vec![0.0_f32; WORLD_SIZE];
    let boundary_scale = (tectonics.plate_speed_cm_per_year * 1.25).max(1.2);

    const NEIGHBORS: [(i32, i32, f32); 8] = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, std::f32::consts::FRAC_1_SQRT_2),
        (-1, -1, std::f32::consts::FRAC_1_SQRT_2),
        (1, 1, std::f32::consts::FRAC_1_SQRT_2),
        (-1, 1, std::f32::consts::FRAC_1_SQRT_2),
    ];

    for y in 0..WORLD_HEIGHT {
        progress.phase(progress_base, progress_span, y as f32 / WORLD_HEIGHT as f32);
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let plate_a = plate_field[i] as usize;
            let a = &plate_vectors[plate_a];
            let mut score = 0.0_f32;
            let mut normal_x = 0.0_f32;
            let mut normal_y = 0.0_f32;
            let mut has_different_neighbor = false;

            for (dx, dy, w) in NEIGHBORS {
                let j = index_spherical(x as i32 + dx, y as i32 + dy);
                let plate_b = plate_field[j] as usize;
                if plate_b == plate_a {
                    continue;
                }
                has_different_neighbor = true;
                let b = &plate_vectors[plate_b];
                let rel_x = b.dir_x - a.dir_x;
                let rel_y = b.dir_y - a.dir_y;
                let edge_score = (rel_x * dx as f32 + rel_y * dy as f32) * w;
                if edge_score.abs() > score.abs() {
                    score = edge_score;
                    normal_x = dx as f32;
                    normal_y = dy as f32;
                }
            }

            boundary_normal_x[i] = normal_x;
            boundary_normal_y[i] = normal_y;
            boundary_strength[i] = clampf(score.abs() / boundary_scale, 0.0, 1.0);

            if score > 0.2 * boundary_scale {
                boundary_types[i] = 1;
            } else if score < -0.2 * boundary_scale {
                boundary_types[i] = 2;
            } else if has_different_neighbor {
                boundary_types[i] = 3;
            } else {
                boundary_types[i] = 0;
            }
        }
    }
    progress.phase(progress_base, progress_span, 1.0);

    ComputePlatesResult {
        plate_field,
        boundary_types,
        boundary_normal_x,
        boundary_normal_y,
        boundary_strength,
        plate_vectors,
    }
}

fn build_fault_backbone(
    seed: u32,
    iterations: usize,
    smooth_passes: usize,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> Vec<f32> {
    let width = WORLD_WIDTH;
    let height = WORLD_HEIGHT;
    let width_i32 = width as i32;
    let height_i32 = height as i32;
    let x_half = width / 2;

    // Column-major storage (x-major) to match classic worldgen fault accumulation.
    let mut fault_col = vec![f32::NAN; WORLD_SIZE];
    for x in 0..width {
        fault_col[x * height] = 0.0;
    }

    let mut sin_iter_phi = vec![0.0_f32; width * 2];
    let step = 2.0 * std::f32::consts::PI / width as f32;
    for i in 0..width {
        let s = (i as f32 * step).sin();
        sin_iter_phi[i] = s;
        sin_iter_phi[i + width] = s;
    }

    let mut rng = Rng::new(seed ^ 0xA341_316C);
    let y_div2 = height as f32 * 0.5;
    let y_div_pi = height as f32 / std::f32::consts::PI;

    let fault_iterations = iterations.max(1);
    for iter in 0..fault_iterations {
        if iter % 24 == 0 {
            let t = iter as f32 / fault_iterations as f32;
            progress.phase(progress_base, progress_span, 0.8 * t);
        }

        let flag_negative = rng.next_f32() < 0.5;
        let alpha = (rng.next_f32() - 0.5) * std::f32::consts::PI;
        let beta = (rng.next_f32() - 0.5) * std::f32::consts::PI;
        let cos_term = clampf(alpha.cos() * beta.cos(), -1.0, 1.0);
        let tan_b = cos_term.acos().tan();
        let xsi = (width as f32 * 0.5 - (width as f32 / std::f32::consts::PI) * beta).round() as i32;

        for phi in 0..x_half {
            let sin_idx = (xsi - phi as i32 + width_i32).clamp(0, (2 * width - 1) as i32) as usize;
            let theta = (y_div_pi * (sin_iter_phi[sin_idx] * tan_b).atan() + y_div2).round() as i32;
            let y = theta.clamp(0, height_i32 - 1) as usize;
            let idx = phi * height + y;
            let delta = if flag_negative { -1.0 } else { 1.0 };
            if fault_col[idx].is_nan() {
                fault_col[idx] = delta;
            } else {
                fault_col[idx] += delta;
            }
        }
    }
    progress.phase(progress_base, progress_span, 0.82);

    // Mirror the second hemisphere from the first (same trick as classic algorithm).
    let offset = x_half * height;
    for x in 0..x_half {
        let row = x * height;
        for y in 1..height {
            fault_col[row + offset + (height - y)] = fault_col[row + y];
        }
    }

    // Integrate step-lines into signed relief profile for each longitude.
    for x in 0..width {
        let row = x * height;
        let mut acc = if fault_col[row].is_nan() {
            0.0
        } else {
            fault_col[row]
        };
        fault_col[row] = acc;
        for y in 1..height {
            let cur = fault_col[row + y];
            if !cur.is_nan() {
                acc += cur;
            }
            fault_col[row + y] = acc;
        }
    }

    let mut min_v = f32::INFINITY;
    let mut max_v = -f32::INFINITY;
    for &v in fault_col.iter() {
        if v.is_nan() {
            continue;
        }
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let mid = (min_v + max_v) * 0.5;
    let half_span = ((max_v - min_v) * 0.5).max(1e-4);

    let mut field = vec![0.0_f32; WORLD_SIZE];
    for y in 0..height {
        for x in 0..width {
            let v = fault_col[x * height + y];
            let n = clampf((v - mid) / half_span, -1.0, 1.0);
            field[index(x, y)] = n;
        }
    }
    progress.phase(progress_base, progress_span, 0.9);

    if smooth_passes == 0 {
        progress.phase(progress_base, progress_span, 1.0);
        return field;
    }

    let mut scratch = field.clone();
    for pass in 0..smooth_passes {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut sum = field[i] * 0.46;
                let mut wsum = 0.46;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.1 } else { 0.065 };
                        sum += field[j] * w;
                        wsum += w;
                    }
                }
                scratch[i] = sum / wsum.max(1e-6);
            }
        }
        field.copy_from_slice(&scratch);
        let t = (pass as f32 + 1.0) / smooth_passes as f32;
        progress.phase(progress_base, progress_span, 0.9 + 0.1 * t);
    }

    field
}

fn build_continent_blob_field(seed: u32, cache: &WorldCache) -> Vec<f32> {
    #[derive(Clone, Copy)]
    struct Blob {
        x: f32,
        y: f32,
        z: f32,
        amp: f32,
        sharpness: f32,
    }

    let mut rng = Rng::new(seed ^ 0x85EB_CA6B);
    let pos_count = 13usize;
    let neg_count = 9usize;
    let mut blobs: Vec<Blob> = Vec::with_capacity(pos_count + neg_count);

    let mut make_blob = |amp_min: f32, amp_max: f32, sh_min: f32, sh_max: f32| -> Blob {
        let z = random_range(&mut rng, -1.0, 1.0);
        let lon = random_range(
            &mut rng,
            -std::f32::consts::PI,
            std::f32::consts::PI,
        );
        let r = (1.0 - z * z).max(0.0).sqrt();
        Blob {
            x: r * lon.cos(),
            y: r * lon.sin(),
            z,
            amp: random_range(&mut rng, amp_min, amp_max),
            sharpness: random_range(&mut rng, sh_min, sh_max),
        }
    };

    for _ in 0..pos_count {
        blobs.push(make_blob(0.65, 1.3, 3.8, 9.4));
    }
    for _ in 0..neg_count {
        blobs.push(make_blob(-1.15, -0.42, 5.5, 13.0));
    }

    let mut out = vec![0.0_f32; WORLD_SIZE];
    let mut min_v = f32::INFINITY;
    let mut max_v = -f32::INFINITY;

    for i in 0..WORLD_SIZE {
        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let mut sum = 0.0_f32;

        for blob in blobs.iter() {
            let dot = clampf(sx * blob.x + sy * blob.y + sz * blob.z, -1.0, 1.0);
            let kernel = ((dot - 1.0) * blob.sharpness).exp();
            sum += blob.amp * kernel;
        }

        // Mild extra isotropic perturbation to avoid perfect radial blobs.
        let jitter = (sx * 2.9 + sy * 3.4 + sz * 2.1 + seed as f32 * 0.0011).sin() * 0.12
            + (sy * 3.1 - sz * 2.6 + sx * 2.4 - seed as f32 * 0.0013).cos() * 0.08;
        let v = sum + jitter;
        min_v = min_v.min(v);
        max_v = max_v.max(v);
        out[i] = v;
    }

    let mid = (min_v + max_v) * 0.5;
    let half = ((max_v - min_v) * 0.5).max(1e-5);
    for v in out.iter_mut() {
        *v = clampf((*v - mid) / half, -1.0, 1.0);
    }

    out
}

fn build_continentality_field(
    seed: u32,
    cache: &WorldCache,
    buoyancy: &[f32],
    blob_backbone: &[f32],
    smooth_passes: usize,
) -> Vec<f32> {
    let mut field = vec![0.0_f32; WORLD_SIZE];
    let seed_phase = seed as f32 * 0.00131;

    for i in 0..WORLD_SIZE {
        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let warp_a = spherical_fbm(
            sx * 1.15 + 3.0,
            sy * 1.15 - 5.0,
            sz * 1.15 + 7.0,
            seed_phase + 11.0,
        );
        let warp_b = spherical_fbm(
            sx * 1.7 - 13.0,
            sy * 1.7 + 17.0,
            sz * 1.7 - 19.0,
            seed_phase + 29.0,
        );
        let warp_c = spherical_fbm(
            sx * 2.2 + 23.0,
            sy * 2.2 - 7.0,
            sz * 2.2 + 31.0,
            seed_phase + 47.0,
        );
        let wx = sx + 0.36 * warp_a + 0.2 * warp_b + 0.12 * warp_c;
        let wy = sy + 0.36 * warp_b + 0.2 * warp_c + 0.12 * warp_a;
        let wz = sz + 0.3 * warp_c + 0.16 * (warp_a - warp_b);

        let low = spherical_fbm(wx * 0.7, wy * 0.7, wz * 0.7, seed_phase + 37.0);
        let mid = spherical_fbm(wx * 1.45, wy * 1.45, wz * 1.45, seed_phase + 53.0);
        let high = spherical_fbm(wx * 2.95, wy * 2.95, wz * 2.95, seed_phase + 71.0);
        let ridgeish = ridge(spherical_fbm(
            wx * 2.1 + 17.0,
            wy * 2.1 - 19.0,
            wz * 2.1 + 11.0,
            seed_phase + 83.0,
        )) * 2.0
            - 1.0;
        let fine = spherical_fbm(
            wx * 5.1 - 41.0,
            wy * 5.1 + 37.0,
            wz * 5.1 - 29.0,
            seed_phase + 97.0,
        );
        field[i] = low * 0.48
            + mid * 0.32
            + high * 0.16
            + ridgeish * 0.04
            + fine * 0.03
            + buoyancy[i] * 0.03
            + blob_backbone[i] * 0.01;
    }

    if smooth_passes > 0 {
        let mut next = field.clone();
        for _ in 0..smooth_passes {
            for y in 0..WORLD_HEIGHT {
                for x in 0..WORLD_WIDTH {
                    let i = index(x, y);
                    let mut sum = field[i] * 0.36;
                    let mut wsum = 0.36;
                    for oy in -1..=1 {
                        for ox in -1..=1 {
                            if ox == 0 && oy == 0 {
                                continue;
                            }
                            let j = index_spherical(x as i32 + ox, y as i32 + oy);
                            let w = if ox == 0 || oy == 0 { 0.11 } else { 0.07 };
                            sum += field[j] * w;
                            wsum += w;
                        }
                    }
                    next[i] = sum / wsum.max(1e-6);
                }
            }
            field.copy_from_slice(&next);
        }
    }

    let (min_v, max_v) = min_max(&field);
    let mid = (min_v + max_v) * 0.5;
    let half = ((max_v - min_v) * 0.5).max(1e-6);
    for v in field.iter_mut() {
        *v = clampf((*v - mid) / half, -1.0, 1.0);
    }

    field
}

fn normalize_height_range(relief: &mut [f32], planet: &PlanetInputs, tectonics: &TectonicInputs) {
    let mut land_heights: Vec<f32> = Vec::new();
    let mut ocean_depths: Vec<f32> = Vec::new();

    for &h in relief.iter() {
        if h > 0.0 {
            land_heights.push(h);
        } else if h < 0.0 {
            ocean_depths.push(-h);
        }
    }

    if land_heights.len() < 8 || ocean_depths.len() < 8 {
        return;
    }

    land_heights.sort_by(|a, b| a.total_cmp(b));
    ocean_depths.sort_by(|a, b| a.total_cmp(b));

    let land_ref = quantile_sorted(&land_heights, 0.995).max(1.0);
    let ocean_ref = quantile_sorted(&ocean_depths, 0.995).max(1.0);

    let speed_factor = clampf(tectonics.plate_speed_cm_per_year / 5.0, 0.35, 2.8);
    let heat_factor = clampf(tectonics.mantle_heat / 55.0, 0.45, 2.2);
    let gravity_factor = clampf(9.81 / planet.gravity.max(1.0), 0.45, 2.4);
    let ocean_factor = clampf(planet.ocean_percent / 67.0, 0.45, 1.9);

    let target_land_max = clampf(
        4300.0 * speed_factor.powf(0.45) * heat_factor.powf(0.32) * gravity_factor.powf(0.62),
        1200.0,
        6600.0,
    );
    let target_ocean_depth = clampf(
        7500.0 * speed_factor.powf(0.5) * heat_factor.powf(0.35) * ocean_factor.powf(0.55),
        1400.0,
        12000.0,
    );
    let land_gamma = clampf(
        1.9 + 0.07 * speed_factor + 0.03 * heat_factor - 0.1 * gravity_factor,
        1.68,
        2.4,
    );
    let ocean_gamma = clampf(0.9 + 0.06 * ocean_factor - 0.05 * speed_factor, 0.74, 1.02);

    for h in relief.iter_mut() {
        if *h > 0.0 {
            let n = (*h / land_ref).max(0.0);
            let core = clampf(n, 0.0, 1.0).powf(land_gamma);
            let tail = (n - 1.0).max(0.0);
            *h = clampf(
                core * target_land_max + tail * target_land_max * 0.14,
                -12000.0,
                9000.0,
            );
        } else if *h < 0.0 {
            let d = (-*h / ocean_ref).max(0.0);
            let core = clampf(d, 0.0, 1.0).powf(ocean_gamma);
            let tail = (d - 1.0).max(0.0);
            *h = clampf(
                -(core * target_ocean_depth + tail * target_ocean_depth * 0.34),
                -12000.0,
                9000.0,
            );
        }
    }
}

fn defuse_orogenic_ribbons(relief: &mut [f32], seed: u32, cache: &WorldCache, detail: DetailProfile) {
    let passes = if detail.erosion_rounds >= 2 { 2 } else { 1 };
    let mut next = relief.to_vec();
    let seed_phase = seed as f32 * 0.00153;

    for pass in 0..passes {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let h = relief[i];
                if h < 700.0 {
                    next[i] = h;
                    continue;
                }

                let sx = cache.x_by_cell[i];
                let sy = cache.y_by_cell[i];
                let sz = cache.z_by_cell[i];
                let cluster_noise = 0.5
                    + 0.5
                        * ((sx * 4.6 + sy * 3.9 + sz * 3.3 + seed_phase + pass as f32 * 1.9).sin()
                            * 0.58
                            + (sy * 5.1 - sz * 3.8 + sx * 2.7 - seed_phase * 1.2).cos() * 0.42);
                let cluster = smoothstep(0.28, 0.88, cluster_noise);

                let mut sum = h * 0.38;
                let mut wsum = 0.38;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        if relief[j] < 0.0 {
                            continue;
                        }
                        let w = if ox == 0 || oy == 0 { 0.1 } else { 0.066 };
                        sum += relief[j] * w;
                        wsum += w;
                    }
                }
                let avg = sum / wsum.max(1e-6);
                let ridge_excess = (h - avg).max(0.0);

                let east = relief[index_spherical(x as i32 + 1, y as i32)].max(0.0);
                let west = relief[index_spherical(x as i32 - 1, y as i32)].max(0.0);
                let north = relief[index_spherical(x as i32, y as i32 - 1)].max(0.0);
                let south = relief[index_spherical(x as i32, y as i32 + 1)].max(0.0);
                let ne = relief[index_spherical(x as i32 + 1, y as i32 - 1)].max(0.0);
                let nw = relief[index_spherical(x as i32 - 1, y as i32 - 1)].max(0.0);
                let se = relief[index_spherical(x as i32 + 1, y as i32 + 1)].max(0.0);
                let sw = relief[index_spherical(x as i32 - 1, y as i32 + 1)].max(0.0);

                let d0 = (east - west).abs();
                let d1 = (north - south).abs();
                let d2 = (ne - sw).abs();
                let d3 = (nw - se).abs();
                let dominant = d0.max(d1).max(d2).max(d3);
                let dir_sum = d0 + d1 + d2 + d3;
                let linearness = if dominant > 1e-4 {
                    clampf((dominant * 4.0 - dir_sum) / dominant, 0.0, 1.0)
                } else {
                    0.0
                };

                let mix = clampf(
                    0.1
                        + 0.34 * (1.0 - cluster)
                        + 0.22 * linearness
                        + 0.18 * clampf(ridge_excess / 900.0, 0.0, 1.0),
                    0.0,
                    0.76,
                );
                let perturb = ((sx * 10.3 + sy * 8.2 + sz * 6.1 + seed_phase).sin() * 0.55
                    + (sy * 9.4 - sz * 7.6 + sx * 5.7 - seed_phase * 1.37).cos() * 0.45)
                    * (18.0 + 58.0 * cluster);
                let target = avg + perturb - linearness * 70.0 * (1.0 - cluster);
                let mut reshaped = h * (1.0 - mix) + target * mix;
                if linearness > 0.55 {
                    reshaped = reshaped.min(avg + 1200.0 + 420.0 * cluster);
                }
                reshaped = reshaped.max(h * 0.4);
                next[i] = reshaped;
            }
        }
        relief.copy_from_slice(&next);
    }
}

fn reshape_ocean_boundaries(relief: &mut [f32], boundary_types: &[i8], boundary_strength: &[f32]) {
    let mut deepest = 0.0_f32;
    for &h in relief.iter() {
        if h < 0.0 {
            deepest = deepest.max(-h);
        }
    }
    let depth_scale = deepest.max(1.0);

    for i in 0..WORLD_SIZE {
        if relief[i] >= 0.0 || relief[i] > -2200.0 {
            continue;
        }

        let x = i % WORLD_WIDTH;
        let y = i / WORLD_WIDTH;
        let mut conv = 0.0_f32;
        let mut div = 0.0_f32;
        let mut transform = 0.0_f32;
        let mut wsum = 0.0_f32;
        for oy in -1..=1 {
            for ox in -1..=1 {
                let j = index_spherical(x as i32 + ox, y as i32 + oy);
                let t = boundary_types[j];
                if t == 0 {
                    continue;
                }
                let s = clampf(boundary_strength[j], 0.0, 1.0);
                let w = if ox == 0 && oy == 0 {
                    1.0
                } else if ox == 0 || oy == 0 {
                    0.62
                } else {
                    0.44
                };
                let value = s * w;
                if t == 1 {
                    conv += value;
                } else if t == 2 {
                    div += value;
                } else if t == 3 {
                    transform += value;
                }
                wsum += w;
            }
        }
        if wsum < 1e-4 {
            continue;
        }
        conv /= wsum;
        div /= wsum;
        transform /= wsum;

        let depth_t = clampf(-relief[i] / depth_scale, 0.0, 1.0);

        if div > 0.08 {
            let uplift = (60.0 + 260.0 * div) * (0.62 + 0.38 * (1.0 - depth_t));
            relief[i] = (relief[i] + uplift).min(-15.0);
        } else if transform > 0.08 {
            let uplift = (14.0 + 65.0 * transform) * (0.6 + 0.4 * (1.0 - depth_t));
            relief[i] += uplift;
        }
        if conv > 0.1 {
            let trench = (45.0 + 220.0 * conv) * (0.54 + 0.46 * depth_t);
            relief[i] -= trench;
        }
    }

    let mut ocean_smoothed = relief.to_vec();
    for _ in 0..2 {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                if relief[i] >= 0.0 {
                    continue;
                }
                let mut sum = relief[i] * 3.9;
                let mut wsum = 3.9;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        if relief[j] >= 0.0 {
                            continue;
                        }
                        let w = if ox == 0 || oy == 0 { 0.72 } else { 0.54 };
                        sum += relief[j] * w;
                        wsum += w;
                    }
                }
                ocean_smoothed[i] = sum / wsum.max(1e-6);
            }
        }
        for i in 0..WORLD_SIZE {
            if relief[i] < 0.0 {
                relief[i] = ocean_smoothed[i];
            }
        }
    }
}

fn apply_coastal_detail(relief: &mut [f32], seed: u32, cache: &WorldCache) {
    for i in 0..WORLD_SIZE {
        let h = relief[i];
        let near_sea = (-h.abs() / 900.0).exp();
        if near_sea < 0.035 {
            continue;
        }

        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let seed_phase = seed as f32 * 0.00091;
        let warp_x =
            sx + 0.22 * (sy * 6.1 + seed_phase).sin() + 0.14 * (sz * 7.4 - seed_phase * 1.7).cos();
        let warp_y = sy
            + 0.2 * (sz * 5.8 - seed_phase * 1.2).sin()
            + 0.12 * (sx * 6.9 + seed_phase * 0.8).cos();
        let warp_z = sz
            + 0.18 * (sx * 5.2 + seed_phase * 1.5).sin()
            + 0.1 * (sy * 7.2 - seed_phase * 0.6).cos();
        let micro = (warp_x * 7.6 + warp_y * 6.4 + warp_z * 5.1 + seed_phase * 1.9).sin() * 0.62
            + (warp_y * 8.1 - warp_z * 5.7 + warp_x * 4.6 - seed_phase * 1.3).cos() * 0.38;
        let macro_n = spherical_fbm(
            warp_x * 0.95 + 2.7,
            warp_y * 0.95 - 3.1,
            warp_z * 0.95 + 1.9,
            seed_phase * 9.2,
        );
        let coast_weight = near_sea.powf(1.08);
        let delta = coast_weight * (macro_n * 26.0 + micro * 14.0);
        if h >= 0.0 {
            relief[i] = (h + delta).max(8.0);
        } else {
            relief[i] = (h + delta).min(-8.0);
        }
    }
}

fn cleanup_coastal_speckles(relief: &mut [f32]) {
    let mut next = relief.to_vec();

    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = relief[i];
            if h.abs() > 260.0 {
                continue;
            }

            let mut land = 0;
            let mut water = 0;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if relief[j] >= 0.0 {
                        land += 1;
                    } else {
                        water += 1;
                    }
                }
            }

            if h >= 0.0 && water >= 7 {
                next[i] = next[i].min(-18.0 - h.abs() * 0.15);
            } else if h < 0.0 && land >= 7 {
                next[i] = next[i].max(18.0 + h.abs() * 0.15);
            }
        }
    }

    relief.copy_from_slice(&next);
}

fn clean_landmask_components(mask: &mut [u8], min_land_cells: usize, min_inland_water_cells: usize) {
    let mut visited = vec![0_u8; WORLD_SIZE];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut component: Vec<usize> = Vec::new();

    // Remove tiny land islands.
    for start in 0..WORLD_SIZE {
        if mask[start] != 1 || visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        queue.push_back(start);
        component.clear();
        while let Some(i) = queue.pop_front() {
            component.push(i);
            let x = i % WORLD_WIDTH;
            let y = i / WORLD_WIDTH;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if visited[j] == 0 && mask[j] == 1 {
                        visited[j] = 1;
                        queue.push_back(j);
                    }
                }
            }
        }
        if component.len() < min_land_cells {
            for &i in component.iter() {
                mask[i] = 0;
            }
        }
    }

    // Find largest water body (global ocean candidate).
    visited.fill(0);
    let mut largest_water = 0_usize;
    for start in 0..WORLD_SIZE {
        if mask[start] != 0 || visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        queue.push_back(start);
        let mut size = 0_usize;
        while let Some(i) = queue.pop_front() {
            size += 1;
            let x = i % WORLD_WIDTH;
            let y = i / WORLD_WIDTH;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if visited[j] == 0 && mask[j] == 0 {
                        visited[j] = 1;
                        queue.push_back(j);
                    }
                }
            }
        }
        largest_water = largest_water.max(size);
    }

    // Fill tiny inland seas/lakes but keep major water bodies.
    visited.fill(0);
    for start in 0..WORLD_SIZE {
        if mask[start] != 0 || visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        queue.push_back(start);
        component.clear();
        while let Some(i) = queue.pop_front() {
            component.push(i);
            let x = i % WORLD_WIDTH;
            let y = i / WORLD_WIDTH;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if visited[j] == 0 && mask[j] == 0 {
                        visited[j] = 1;
                        queue.push_back(j);
                    }
                }
            }
        }
        let size = component.len();
        if size < min_inland_water_cells && size < largest_water / 18 {
            for &i in component.iter() {
                mask[i] = 1;
            }
        }
    }
}

fn smooth_landmask(mask: &mut [u8], passes: usize) {
    if passes == 0 {
        return;
    }
    let mut next = mask.to_vec();
    for _ in 0..passes {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut land_neighbors = 0_i32;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        if mask[j] == 1 {
                            land_neighbors += 1;
                        }
                    }
                }

                let cur = mask[i];
                next[i] = if cur == 1 {
                    if land_neighbors <= 2 {
                        0
                    } else {
                        1
                    }
                } else if land_neighbors >= 6 {
                    1
                } else {
                    0
                };
            }
        }
        mask.copy_from_slice(&next);
    }
}

fn blend_topology_edges(relief: &mut [f32]) {
    let mut next = relief.to_vec();
    for _ in 0..2 {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let sign_i = relief[i] >= 0.0;
                let mut opposite = 0_u8;
                let mut sum = relief[i] * 0.45;
                let mut wsum = 0.45;

                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let sign_j = relief[j] >= 0.0;
                        if sign_i != sign_j {
                            opposite += 1;
                        }
                        let w = if ox == 0 || oy == 0 { 0.14 } else { 0.09 };
                        sum += relief[j] * w;
                        wsum += w;
                    }
                }

                if opposite >= 3 {
                    next[i] = sum / wsum.max(1e-6);
                } else {
                    next[i] = relief[i];
                }
            }
        }
        relief.copy_from_slice(&next);
    }
}

fn stabilize_landmass_topology(relief: &mut [f32], planet: &PlanetInputs, detail: DetailProfile) {
    let mut landmask = vec![0_u8; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        landmask[i] = if relief[i] >= 0.0 { 1 } else { 0 };
    }

    let smooth_passes = if detail.erosion_rounds >= 2 { 3 } else { 2 };
    smooth_landmask(&mut landmask, smooth_passes);

    let min_land_cells = clampf(
        WORLD_SIZE as f32
            * (0.0018 + (planet.ocean_percent / 100.0) * 0.0018),
        1800.0,
        18_000.0,
    ) as usize;
    let min_inland_water_cells = clampf(
        WORLD_SIZE as f32 * (0.0019 + (1.0 - planet.ocean_percent / 100.0) * 0.0014),
        2200.0,
        26_000.0,
    ) as usize;
    clean_landmask_components(&mut landmask, min_land_cells, min_inland_water_cells);

    for i in 0..WORLD_SIZE {
        if landmask[i] == 1 {
            if relief[i] < 0.0 {
                relief[i] = relief[i] * 0.3 + 70.0;
            }
        } else if relief[i] >= 0.0 {
            relief[i] = relief[i] * 0.25 - 75.0;
        }
    }

    blend_topology_edges(relief);
}

fn defuse_plate_linearity(
    relief: &mut [f32],
    plates: &ComputePlatesResult,
    seed: u32,
    cache: &WorldCache,
    detail: DetailProfile,
) {
    let strength_scale = if detail.buoyancy_smooth_passes >= 20 {
        1.0
    } else if detail.buoyancy_smooth_passes >= 12 {
        0.85
    } else {
        0.65
    };

    let mut imprint = vec![0.0_f32; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        if relief[i] < -1600.0 {
            continue;
        }
        let base = clampf(plates.boundary_strength[i], 0.0, 1.0);
        let k = match plates.boundary_types[i] {
            1 => 1.0,
            2 => 0.78,
            3 => 0.58,
            _ => 0.0,
        };
        imprint[i] = base * k;
    }

    let mut imprint_next = imprint.clone();
    for _ in 0..3 {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut sum = imprint[i] * 0.5;
                let mut wsum = 0.5;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.11 } else { 0.07 };
                        sum += imprint[j] * w;
                        wsum += w;
                    }
                }
                imprint_next[i] = sum / wsum.max(1e-6);
            }
        }
        imprint.copy_from_slice(&imprint_next);
    }

    let mut softened = relief.to_vec();
    let seed_phase = seed as f32 * 0.0017;
    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let mask = clampf(imprint[i] * strength_scale, 0.0, 1.0);
            if mask < 0.08 || relief[i] < -2400.0 {
                continue;
            }

            let mut sum = relief[i] * 0.42;
            let mut wsum = 0.42;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let w = if ox == 0 || oy == 0 { 0.1 } else { 0.065 };
                    sum += relief[j] * w;
                    wsum += w;
                }
            }
            let avg = sum / wsum.max(1e-6);
            let sx = cache.x_by_cell[i];
            let sy = cache.y_by_cell[i];
            let sz = cache.z_by_cell[i];
            let depth_softener = smoothstep(-2400.0, 220.0, relief[i]);
            let perturb = ((sx * 12.1 + sy * 9.4 + sz * 8.3 + seed_phase).sin() * 0.58
                + (sy * 13.4 - sz * 7.8 + sx * 6.1 - seed_phase * 1.4).cos() * 0.42)
                * (14.0 + 74.0 * mask)
                * depth_softener;
            let mix = clampf(0.16 + 0.6 * mask, 0.0, 0.84) * depth_softener;
            let blended = relief[i] * (1.0 - mix) + (avg + perturb) * mix;
            softened[i] = if relief[i] >= 0.0 {
                blended.max(10.0)
            } else {
                blended.min(-10.0)
            };
        }
    }

    relief.copy_from_slice(&softened);
}

fn defuse_coastal_linearity(
    relief: &mut [f32],
    plates: &ComputePlatesResult,
    seed: u32,
    cache: &WorldCache,
) {
    let mut imprint = vec![0.0_f32; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        let k = match plates.boundary_types[i] {
            1 => 1.0,
            2 => 0.85,
            3 => 0.7,
            _ => 0.0,
        };
        imprint[i] = clampf(plates.boundary_strength[i], 0.0, 1.0) * k;
    }

    let mut blur = imprint.clone();
    for _ in 0..2 {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut sum = imprint[i] * 0.42;
                let mut wsum = 0.42;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.1 } else { 0.062 };
                        sum += imprint[j] * w;
                        wsum += w;
                    }
                }
                blur[i] = sum / wsum.max(1e-6);
            }
        }
        imprint.copy_from_slice(&blur);
    }

    let mut next = relief.to_vec();
    let seed_phase = seed as f32 * 0.00123;
    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = relief[i];
            if h.abs() > 920.0 {
                continue;
            }

            let mask = imprint[i];
            if mask < 0.1 {
                continue;
            }

            let coast = 1.0 - smoothstep(120.0, 920.0, h.abs());
            if coast <= 0.0 {
                continue;
            }

            let sx = cache.x_by_cell[i];
            let sy = cache.y_by_cell[i];
            let sz = cache.z_by_cell[i];
            let macro_n = spherical_fbm(
                sx * 1.05 + 5.7,
                sy * 1.05 - 4.1,
                sz * 1.05 + 3.3,
                seed_phase + 17.0,
            );
            let micro_n = spherical_fbm(
                sx * 3.4 + 11.0,
                sy * 3.4 - 7.0,
                sz * 3.4 + 9.0,
                seed_phase + 41.0,
            );

            let mut sum = h * 0.34;
            let mut wsum = 0.34;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if h >= 0.0 && relief[j] < -350.0 {
                        continue;
                    }
                    if h < 0.0 && relief[j] > 350.0 {
                        continue;
                    }
                    let w = if ox == 0 || oy == 0 { 0.11 } else { 0.07 };
                    sum += relief[j] * w;
                    wsum += w;
                }
            }
            let avg = sum / wsum.max(1e-6);

            let jitter = (macro_n * 290.0 + micro_n * 120.0) * coast * (0.5 + 0.62 * mask);
            let mix = clampf(0.14 + 0.56 * mask * coast, 0.0, 0.78);
            let mut v = h * (1.0 - mix) + (avg + jitter) * mix;
            if h >= 0.0 {
                v = v.max(-120.0);
            } else {
                v = v.min(120.0);
            }
            next[i] = v;
        }
    }

    relief.copy_from_slice(&next);
}

fn break_straight_coasts(relief: &mut [f32], seed: u32, cache: &WorldCache) {
    let mut next = relief.to_vec();
    let seed_phase = seed as f32 * 0.00097;

    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = relief[i];
            if h.abs() > 620.0 {
                continue;
            }

            let east = relief[index_spherical(x as i32 + 1, y as i32)];
            let west = relief[index_spherical(x as i32 - 1, y as i32)];
            let north = relief[index_spherical(x as i32, y as i32 - 1)];
            let south = relief[index_spherical(x as i32, y as i32 + 1)];
            let gx = east - west;
            let gy = south - north;
            let gl = gx.hypot(gy);
            if gl < 14.0 {
                continue;
            }
            let nx = gx / gl;
            let ny = gy / gl;

            let mut coh_sum = 0.0_f32;
            let mut coh_count = 0.0_f32;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if relief[j].abs() > 420.0 {
                        continue;
                    }
                    let je = relief[index_spherical(x as i32 + ox + 1, y as i32 + oy)];
                    let jw = relief[index_spherical(x as i32 + ox - 1, y as i32 + oy)];
                    let jn = relief[index_spherical(x as i32 + ox, y as i32 + oy - 1)];
                    let js = relief[index_spherical(x as i32 + ox, y as i32 + oy + 1)];
                    let jgx = je - jw;
                    let jgy = js - jn;
                    let jgl = jgx.hypot(jgy);
                    if jgl < 12.0 {
                        continue;
                    }
                    let jnx = jgx / jgl;
                    let jny = jgy / jgl;
                    coh_sum += (nx * jnx + ny * jny).abs();
                    coh_count += 1.0;
                }
            }

            if coh_count < 3.0 {
                continue;
            }
            let coherence = coh_sum / coh_count;
            if coherence < 0.58 {
                continue;
            }

            let coast = 1.0 - smoothstep(40.0, 620.0, h.abs());
            let sx = cache.x_by_cell[i];
            let sy = cache.y_by_cell[i];
            let sz = cache.z_by_cell[i];
            let jitter = spherical_fbm(
                sx * 2.4 + 7.0,
                sy * 2.4 - 9.0,
                sz * 2.4 + 11.0,
                seed_phase + 23.0,
            );
            let gain = ((coherence - 0.58) / 0.42).clamp(0.0, 1.0);
            next[i] = h + jitter * 460.0 * coast * gain;
        }
    }

    relief.copy_from_slice(&next);
}

fn fracture_coastline_band(relief: &mut [f32], seed: u32, cache: &WorldCache) {
    let seed_phase = seed as f32 * 0.00191;
    for i in 0..WORLD_SIZE {
        let h = relief[i];
        if h.abs() > 1200.0 {
            continue;
        }

        let coast = 1.0 - smoothstep(90.0, 1200.0, h.abs());
        if coast <= 0.0 {
            continue;
        }

        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let n1 = spherical_fbm(
            sx * 2.9 + 9.0,
            sy * 2.9 - 5.0,
            sz * 2.9 + 2.0,
            seed_phase + 13.0,
        );
        let n2 = spherical_fbm(
            sx * 6.2 - 17.0,
            sy * 6.2 + 11.0,
            sz * 6.2 - 7.0,
            seed_phase + 31.0,
        );
        let n3 = ridge(spherical_fbm(
            sx * 4.3 + 21.0,
            sy * 4.3 - 19.0,
            sz * 4.3 + 15.0,
            seed_phase + 57.0,
        )) * 2.0
            - 1.0;
        let signal = n1 * 0.52 + n2 * 0.33 + n3 * 0.15;
        relief[i] += signal * 560.0 * coast;
    }
}

fn warp_coastal_band(relief: &mut [f32], seed: u32, cache: &WorldCache) {
    let mut next = relief.to_vec();
    let seed_phase = seed as f32 * 0.00137;

    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = relief[i];
            if h.abs() > 950.0 {
                continue;
            }

            let coast = 1.0 - smoothstep(80.0, 950.0, h.abs());
            if coast <= 0.0 {
                continue;
            }

            let sx = cache.x_by_cell[i];
            let sy = cache.y_by_cell[i];
            let sz = cache.z_by_cell[i];
            let noise_u = spherical_fbm(
                sx * 2.2 + 5.0,
                sy * 2.2 - 7.0,
                sz * 2.2 + 9.0,
                seed_phase + 13.0,
            );
            let noise_v = spherical_fbm(
                sx * 2.5 - 11.0,
                sy * 2.5 + 3.0,
                sz * 2.5 - 17.0,
                seed_phase + 31.0,
            );

            let shift = 0.45 + 2.9 * coast;
            let ox = (noise_u * shift).round() as i32;
            let oy = (noise_v * shift).round() as i32;
            let j = index_spherical(x as i32 + ox, y as i32 + oy);

            let mix = clampf(0.16 + 0.62 * coast, 0.0, 0.78);
            let warped = relief[i] * (1.0 - mix) + relief[j] * mix;
            let fine = spherical_fbm(
                sx * 4.4 + 23.0,
                sy * 4.4 - 19.0,
                sz * 4.4 + 29.0,
                seed_phase + 57.0,
            ) * 130.0 * coast;
            next[i] = warped + fine;
        }
    }

    relief.copy_from_slice(&next);
}

fn apply_ocean_profile(
    relief: &mut [f32],
    boundary_types: &[i8],
    boundary_strength: &[f32],
    planet: &PlanetInputs,
    seed: u32,
    cache: &WorldCache,
    detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) {
    let mut distance_to_coast = vec![-1_i32; WORLD_SIZE];
    let mut queue: VecDeque<usize> = VecDeque::new();

    for y in 0..WORLD_HEIGHT {
        progress.phase(
            progress_base,
            progress_span,
            0.1 * (y as f32 / WORLD_HEIGHT as f32),
        );
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            if relief[i] >= 0.0 {
                continue;
            }

            let mut is_coast_adjacent = false;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if relief[j] >= 0.0 {
                        is_coast_adjacent = true;
                        break;
                    }
                }
                if is_coast_adjacent {
                    break;
                }
            }

            if is_coast_adjacent {
                distance_to_coast[i] = 0;
                queue.push_back(i);
            }
        }
    }

    const FLOW_STEPS: [(i32, i32); 8] = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
    ];

    while let Some(i) = queue.pop_front() {
        let x = i % WORLD_WIDTH;
        let y = i / WORLD_WIDTH;
        let next_dist = distance_to_coast[i] + 1;

        for (dx, dy) in FLOW_STEPS {
            let j = index_spherical(x as i32 + dx, y as i32 + dy);
            if relief[j] >= 0.0 || distance_to_coast[j] >= 0 {
                continue;
            }
            distance_to_coast[j] = next_dist;
            queue.push_back(j);
        }
    }
    progress.phase(progress_base, progress_span, 0.2);

    let mut max_distance = 1_i32;
    for i in 0..WORLD_SIZE {
        if relief[i] < 0.0 && distance_to_coast[i] > max_distance {
            max_distance = distance_to_coast[i];
        }
    }

    let shelf_cells = clampf(4.8 + (planet.radius_km / 6371.0 - 1.0) * 1.2, 3.8, 8.4);
    let slope_cells = clampf(15.0 + (planet.ocean_percent - 67.0) * 0.08, 11.0, 23.0);
    let abyssal_base = clampf(
        4200.0 + (planet.ocean_percent - 67.0) * 26.0,
        3600.0,
        5600.0,
    );

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.2 + 0.55 * (i as f32 / WORLD_SIZE as f32),
            );
        }
        if relief[i] >= 0.0 {
            continue;
        }

        let d = distance_to_coast[i].max(0) as f32;
        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let seed_phase = seed as f32 * 0.0012;
        let margin_signal = 0.5
            + 0.5
                * ((sx * 6.2 + sy * 5.1 + sz * 4.3 + seed_phase).sin() * 0.62
                    + (sy * 7.4 - sz * 4.9 + sx * 3.8 - seed_phase * 1.1).cos() * 0.38);
        let x = i % WORLD_WIDTH;
        let y = i / WORLD_WIDTH;
        let mut conv = 0.0_f32;
        let mut div = 0.0_f32;
        let mut transform = 0.0_f32;
        let mut wsum = 0.0_f32;
        for oy in -1..=1 {
            for ox in -1..=1 {
                let j = index_spherical(x as i32 + ox, y as i32 + oy);
                let t = boundary_types[j];
                if t == 0 {
                    continue;
                }
                let s = clampf(boundary_strength[j], 0.0, 1.0);
                let w = if ox == 0 && oy == 0 {
                    1.0
                } else if ox == 0 || oy == 0 {
                    0.58
                } else {
                    0.4
                };
                let value = s * w;
                if t == 1 {
                    conv += value;
                } else if t == 2 {
                    div += value;
                } else if t == 3 {
                    transform += value;
                }
                wsum += w;
            }
        }

        if wsum > 1e-4 {
            conv /= wsum;
            div /= wsum;
            transform /= wsum;
        }

        let regime_bias = match boundary_types[i] {
            1 => -0.28, // active convergent margins -> narrower shelf
            2 => 0.22,  // passive/divergent margins -> wider shelf
            3 => -0.08,
            _ => 0.0,
        };
        let shelf_texture = 0.5
            + 0.5
                * ((sx * 4.7 + sy * 3.8 + sz * 3.1 + seed_phase * 0.9).sin() * 0.58
                    + (sy * 5.1 - sz * 3.6 + sx * 2.9 - seed_phase * 1.1).cos() * 0.42);
        let tectonic_shelf_factor = clampf(1.0 + 1.35 * div - 1.45 * conv - 0.4 * transform, 0.4, 2.15);
        let tectonic_slope_factor = clampf(1.0 + 1.15 * conv - 0.32 * div, 0.62, 1.95);
        let local_shelf = shelf_cells
            * clampf(
                0.34 + 1.18 * margin_signal + regime_bias + 0.55 * (shelf_texture - 0.5),
                0.22,
                2.7,
            )
            * tectonic_shelf_factor;
        let local_slope = slope_cells
            * clampf(
                0.56 + 1.02 * (1.0 - margin_signal) - regime_bias * 0.35 + 0.22 * conv,
                0.36,
                2.2,
            )
            * tectonic_slope_factor;
        let shore_depth = -12.0
            - 98.0 * clampf(margin_signal + regime_bias * 0.35, 0.0, 1.0)
            - 140.0 * conv
            + 62.0 * div;
        let mut target_depth = if d <= local_shelf {
            let t = clampf(d / local_shelf.max(0.5), 0.0, 1.0);
            shore_depth - (150.0 + 240.0 * margin_signal + 110.0 * conv) * t.powf(1.42)
        } else if d <= local_shelf + local_slope {
            let t = clampf((d - local_shelf) / local_slope.max(0.8), 0.0, 1.0);
            -245.0 - (abyssal_base - 245.0) * t.powf(0.74)
        } else {
            let t = clampf(
                (d - local_shelf - local_slope) / (max_distance as f32 + 1.0),
                0.0,
                1.0,
            );
            -abyssal_base - 1650.0 * t.powf(0.58)
        };
        let basin_weight = clampf((d - local_shelf) / (local_shelf + local_slope + 1.0), 0.0, 1.0);
        let undulation = (sx * 7.2 + sy * 5.3 + sz * 4.1 + seed_phase).sin() * 0.6
            + (sy * 8.6 - sz * 4.9 + sx * 3.7 - seed_phase * 1.35).cos() * 0.4;
        target_depth += undulation * (120.0 + 390.0 * basin_weight);
        target_depth += -760.0 * conv + 320.0 * div - 130.0 * transform;

        let blend = 0.72 + 0.24 * basin_weight;
        relief[i] = clampf(
            relief[i] * (1.0 - blend) + target_depth * blend,
            -12000.0,
            -5.0,
        );
    }

    let mut smoothed = relief.to_vec();
    for pass in 0..detail.ocean_smooth_passes {
        for y in 0..WORLD_HEIGHT {
            let pass_t = (pass as f32 + y as f32 / WORLD_HEIGHT as f32)
                / detail.ocean_smooth_passes.max(1) as f32;
            progress.phase(progress_base, progress_span, 0.75 + 0.25 * pass_t);
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                if relief[i] >= 0.0 {
                    continue;
                }
                let mut sum = relief[i] * 2.8;
                let mut wsum = 2.8;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        if relief[j] >= 0.0 {
                            continue;
                        }
                        let w = if ox == 0 || oy == 0 { 0.5 } else { 0.35 };
                        sum += relief[j] * w;
                        wsum += w;
                    }
                }
                smoothed[i] = sum / wsum.max(1e-6);
            }
        }
        for i in 0..WORLD_SIZE {
            if relief[i] < 0.0 {
                relief[i] = smoothed[i];
            }
        }
    }
    progress.phase(progress_base, progress_span, 1.0);
}

fn compute_relief(
    planet: &PlanetInputs,
    tectonics: &TectonicInputs,
    plates: &ComputePlatesResult,
    seed: u32,
    cache: &WorldCache,
    detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> ReliefResult {
    let mut relief = vec![0.0_f32; WORLD_SIZE];

    let seed_mix = (((planet.radius_km * 1000.0) as i32)
        ^ ((tectonics.plate_speed_cm_per_year * 1000.0) as i32)
        ^ ((tectonics.mantle_heat * 1000.0) as i32)
        ^ (seed.wrapping_mul(2_654_435_761) as i32)) as u32;
    let mut random_seed = Rng::new(seed_mix);
    let k_relief = 560.0_f32;
    let fault_backbone = build_fault_backbone(
        seed_mix,
        detail.fault_iterations,
        detail.fault_smooth_passes,
        progress,
        progress_base,
        progress_span * 0.12,
    );
    let blob_backbone = build_continent_blob_field(seed_mix ^ 0xD1B5_4A35, cache);

    // Smooth per-plate buoyancy so continent/ocean macro-shape is not cut by hard plate polygons.
    let mut buoyancy_field = vec![0.0_f32; WORLD_SIZE];
    for (i, b) in buoyancy_field.iter_mut().enumerate() {
        let pid = plates.plate_field[i] as usize;
        *b = plates.plate_vectors[pid].buoyancy;
    }
    let mut buoyancy_scratch = vec![0.0_f32; WORLD_SIZE];
    for pass in 0..detail.buoyancy_smooth_passes {
        for y in 0..WORLD_HEIGHT {
            let pass_t = (pass as f32 + y as f32 / WORLD_HEIGHT as f32)
                / detail.buoyancy_smooth_passes.max(1) as f32;
            progress.phase(progress_base, progress_span, 0.12 + 0.12 * pass_t);
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut sum = buoyancy_field[i] * 0.34;
                let mut wsum = 0.34;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.125 } else { 0.07 };
                        sum += buoyancy_field[j] * w;
                        wsum += w;
                    }
                }
                buoyancy_scratch[i] = sum / wsum.max(1e-6);
            }
        }
        buoyancy_field.copy_from_slice(&buoyancy_scratch);
    }

    let continentality_field = build_continentality_field(
        seed_mix ^ 0x7FEB_352D,
        cache,
        &buoyancy_field,
        &blob_backbone,
        (detail.buoyancy_smooth_passes / 4).clamp(1, 4),
    );

    let mut heat_norm_field = vec![0.0_f32; WORLD_SIZE];
    let mut weakness_field = vec![0.0_f32; WORLD_SIZE];
    let mut strength_field = vec![0.0_f32; WORLD_SIZE];
    let mut comp_source = vec![0.0_f32; WORLD_SIZE];
    let mut ext_source = vec![0.0_f32; WORLD_SIZE];
    let mut shear_source = vec![0.0_f32; WORLD_SIZE];
    let mut velocity_x = vec![0.0_f32; WORLD_SIZE];
    let mut velocity_y = vec![0.0_f32; WORLD_SIZE];

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.24 + 0.08 * (i as f32 / WORLD_SIZE as f32),
            );
        }
        let plate_id = plates.plate_field[i] as usize;
        let heat = plates.plate_vectors[plate_id].heat;
        let heat_norm = clampf(heat / tectonics.mantle_heat.max(1.0), 0.2, 2.3);
        heat_norm_field[i] = heat_norm;

        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let boundary = clampf(plates.boundary_strength[i], 0.0, 1.0);
        let fault = 0.5 + 0.5 * fault_backbone[i];
        let inherited_noise = 0.5
            + 0.5
                * ((sx * 5.7 + sy * 4.8 + sz * 3.9 + seed_mix as f32 * 0.0011).sin() * 0.61
                    + (sy * 6.3 - sz * 4.4 + sx * 3.2 - seed_mix as f32 * 0.0009).cos() * 0.39);
        let segment_noise = 0.5
            + 0.5
                * ((sx * 12.1 + sy * 8.7 + sz * 6.5 + seed_mix as f32 * 0.0018).sin() * 0.54
                    + (sy * 11.4 - sz * 7.9 + sx * 5.8 - seed_mix as f32 * 0.0014).cos() * 0.46);
        let segment_gate = smoothstep(0.24, 0.88, segment_noise);
        let inherited_gate = smoothstep(0.22, 0.9, inherited_noise * 0.76 + fault * 0.24);
        let weakness = clampf(
            0.14 + 0.58 * inherited_gate + 0.34 * boundary * segment_gate,
            0.0,
            1.0,
        );
        weakness_field[i] = weakness;

        let strength = clampf(
            0.72 + 0.23 * continentality_field[i] + 0.18 * (1.0 - weakness) - 0.24 * heat_norm,
            0.05,
            1.5,
        );
        strength_field[i] = strength;

        let source = boundary * (0.42 + 0.58 * segment_gate) * (0.9 + 0.2 * fault);
        match plates.boundary_types[i] {
            1 => {
                comp_source[i] = source * (0.52 + 0.48 * inherited_gate);
                shear_source[i] = source * 0.24 * (0.4 + 0.6 * segment_gate);
            }
            2 => {
                ext_source[i] = source * (0.54 + 0.46 * segment_gate);
                shear_source[i] = source * 0.2 * (0.46 + 0.54 * (1.0 - segment_gate));
            }
            3 => {
                shear_source[i] = source * (0.5 + 0.5 * (1.0 - segment_gate * 0.35));
            }
            _ => {}
        }

        velocity_x[i] = plates.plate_vectors[plate_id].dir_x;
        velocity_y[i] = plates.plate_vectors[plate_id].dir_y;
    }

    let vel_smooth_passes = (detail.max_kernel_radius as usize + detail.buoyancy_smooth_passes / 6)
        .clamp(3, 8);
    let mut vel_x_next = velocity_x.clone();
    let mut vel_y_next = velocity_y.clone();
    for pass in 0..vel_smooth_passes {
        for y in 0..WORLD_HEIGHT {
            let t = (pass as f32 + y as f32 / WORLD_HEIGHT as f32) / vel_smooth_passes as f32;
            progress.phase(progress_base, progress_span, 0.28 + 0.04 * t);
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut sx = velocity_x[i] * 0.34;
                let mut sy = velocity_y[i] * 0.34;
                let mut wsum = 0.34;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.11 } else { 0.07 };
                        sx += velocity_x[j] * w;
                        sy += velocity_y[j] * w;
                        wsum += w;
                    }
                }
                vel_x_next[i] = sx / wsum.max(1e-6);
                vel_y_next[i] = sy / wsum.max(1e-6);
            }
        }
        velocity_x.copy_from_slice(&vel_x_next);
        velocity_y.copy_from_slice(&vel_y_next);
    }

    let mut comp_grad = vec![0.0_f32; WORLD_SIZE];
    let mut ext_grad = vec![0.0_f32; WORLD_SIZE];
    let mut shear_grad = vec![0.0_f32; WORLD_SIZE];
    let mut comp_peak = 1e-6_f32;
    let mut ext_peak = 1e-6_f32;
    let mut shear_peak = 1e-6_f32;
    for y in 0..WORLD_HEIGHT {
        let t = y as f32 / WORLD_HEIGHT as f32;
        progress.phase(progress_base, progress_span, 0.32 + 0.02 * t);
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let vx_r = velocity_x[index_spherical(x as i32 + 1, y as i32)];
            let vx_l = velocity_x[index_spherical(x as i32 - 1, y as i32)];
            let vx_u = velocity_x[index_spherical(x as i32, y as i32 - 1)];
            let vx_d = velocity_x[index_spherical(x as i32, y as i32 + 1)];
            let vy_r = velocity_y[index_spherical(x as i32 + 1, y as i32)];
            let vy_l = velocity_y[index_spherical(x as i32 - 1, y as i32)];
            let vy_u = velocity_y[index_spherical(x as i32, y as i32 - 1)];
            let vy_d = velocity_y[index_spherical(x as i32, y as i32 + 1)];
            let div = ((vx_r - vx_l) + (vy_d - vy_u)) * 0.5;
            let shear = ((vx_d - vx_u).abs() + (vy_r - vy_l).abs()) * 0.5;
            let cg = (-div).max(0.0);
            let eg = div.max(0.0);
            comp_grad[i] = cg;
            ext_grad[i] = eg;
            shear_grad[i] = shear;
            comp_peak = comp_peak.max(cg);
            ext_peak = ext_peak.max(eg);
            shear_peak = shear_peak.max(shear);
        }
    }

    for i in 0..WORLD_SIZE {
        let weak = weakness_field[i];
        let cg = clampf(comp_grad[i] / comp_peak, 0.0, 1.0).powf(0.9);
        let eg = clampf(ext_grad[i] / ext_peak, 0.0, 1.0).powf(0.9);
        let sg = clampf(shear_grad[i] / shear_peak, 0.0, 1.0).powf(0.92);
        comp_source[i] = clampf(
            comp_source[i] * 0.72 + cg * (0.22 + 0.2 * weak),
            0.0,
            1.5,
        );
        ext_source[i] = clampf(
            ext_source[i] * 0.72 + eg * (0.22 + 0.16 * (1.0 - weak)),
            0.0,
            1.5,
        );
        shear_source[i] = clampf(
            shear_source[i] * 0.68 + sg * (0.2 + 0.22 * weak),
            0.0,
            1.4,
        );
    }

    let mut comp_field = comp_source.clone();
    let mut ext_field = ext_source.clone();
    let mut shear_field = shear_source.clone();
    let mut comp_next = vec![0.0_f32; WORLD_SIZE];
    let mut ext_next = vec![0.0_f32; WORLD_SIZE];
    let mut shear_next = vec![0.0_f32; WORLD_SIZE];

    const STRAIN_NEIGHBORS: [(i32, i32, f32); 8] = [
        (1, 0, 0.17),
        (-1, 0, 0.17),
        (0, 1, 0.17),
        (0, -1, 0.17),
        (1, 1, 0.12),
        (-1, 1, 0.12),
        (1, -1, 0.12),
        (-1, -1, 0.12),
    ];

    let strain_passes = (detail.buoyancy_smooth_passes / 2 + detail.max_kernel_radius as usize + 5)
        .clamp(9, 26);
    for pass in 0..strain_passes {
        let pass_t = pass as f32 / strain_passes.max(1) as f32;
        for y in 0..WORLD_HEIGHT {
            let row_t = (pass as f32 + y as f32 / WORLD_HEIGHT as f32) / strain_passes as f32;
            progress.phase(progress_base, progress_span, 0.32 + 0.18 * row_t);
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut nx = plates.boundary_normal_x[i];
                let mut ny = plates.boundary_normal_y[i];
                let mut nlen = nx.hypot(ny);
                if nlen < 0.15 {
                    let left = continentality_field[index_spherical(x as i32 - 1, y as i32)];
                    let right = continentality_field[index_spherical(x as i32 + 1, y as i32)];
                    let up = continentality_field[index_spherical(x as i32, y as i32 - 1)];
                    let down = continentality_field[index_spherical(x as i32, y as i32 + 1)];
                    nx = right - left;
                    ny = down - up;
                    nlen = nx.hypot(ny);
                }
                if nlen < 1e-5 {
                    nx = 1.0;
                    ny = 0.0;
                    nlen = 1.0;
                }
                let nux = nx / nlen;
                let nuy = ny / nlen;
                let tx = -nuy;
                let ty = nux;
                let weak = weakness_field[i];
                let heat_norm = heat_norm_field[i];
                let strength = strength_field[i];
                let conductivity = clampf(
                    0.12 + 0.54 * weak + 0.2 * heat_norm - 0.2 * strength,
                    0.05,
                    0.96,
                );

                let mut comp_sum = comp_field[i] * 0.42;
                let mut ext_sum = ext_field[i] * 0.42;
                let mut shear_sum = shear_field[i] * 0.42;
                let mut wsum = 0.42;

                for (dx, dy, base_w) in STRAIN_NEIGHBORS {
                    let j = index_spherical(x as i32 + dx, y as i32 + dy);
                    let along = ((dx as f32) * tx + (dy as f32) * ty).abs();
                    let across = ((dx as f32) * nux + (dy as f32) * nuy).abs();
                    let orient = clampf(0.78 + 0.34 * along - 0.12 * across, 0.5, 1.35);
                    let bridge = 0.55 + 0.45 * conductivity * (0.42 + 0.58 * weakness_field[j]);
                    let w = base_w * orient * bridge;
                    comp_sum += comp_field[j] * w;
                    ext_sum += ext_field[j] * w;
                    shear_sum += shear_field[j] * w;
                    wsum += w;
                }

                let comp_avg = comp_sum / wsum.max(1e-6);
                let ext_avg = ext_sum / wsum.max(1e-6);
                let shear_avg = shear_sum / wsum.max(1e-6);

                let comp_source_drive = comp_source[i] * (0.46 + 0.42 * weak);
                let ext_source_drive = ext_source[i] * (0.48 + 0.38 * (1.0 - weak));
                let shear_source_drive = shear_source[i] * (0.4 + 0.45 * weak);

                comp_next[i] = clampf(
                    comp_field[i] * (0.52 + 0.2 * (1.0 - conductivity))
                        + comp_avg * (0.3 + 0.44 * conductivity)
                        + comp_source_drive * 0.3
                        - (0.008 + 0.006 * pass_t),
                    0.0,
                    2.8,
                );
                ext_next[i] = clampf(
                    ext_field[i] * (0.54 + 0.2 * (1.0 - conductivity))
                        + ext_avg * (0.3 + 0.4 * conductivity)
                        + ext_source_drive * 0.31
                        - (0.007 + 0.005 * pass_t),
                    0.0,
                    2.8,
                );
                shear_next[i] = clampf(
                    shear_field[i] * (0.58 + 0.16 * (1.0 - conductivity))
                        + shear_avg * (0.27 + 0.36 * conductivity)
                        + shear_source_drive * 0.36
                        - (0.006 + 0.004 * pass_t),
                    0.0,
                    2.6,
                );
            }
        }

        comp_field.copy_from_slice(&comp_next);
        ext_field.copy_from_slice(&ext_next);
        shear_field.copy_from_slice(&shear_next);
    }

    let mut comp_peak = 1e-6_f32;
    let mut ext_peak = 1e-6_f32;
    let mut shear_peak = 1e-6_f32;
    for i in 0..WORLD_SIZE {
        comp_peak = comp_peak.max(comp_field[i]);
        ext_peak = ext_peak.max(ext_field[i]);
        shear_peak = shear_peak.max(shear_field[i]);
    }
    for i in 0..WORLD_SIZE {
        comp_field[i] = clampf(comp_field[i] / comp_peak, 0.0, 1.0).powf(0.86);
        ext_field[i] = clampf(ext_field[i] / ext_peak, 0.0, 1.0).powf(0.9);
        shear_field[i] = clampf(shear_field[i] / shear_peak, 0.0, 1.0).powf(0.94);
    }

    let corridor_passes = (detail.max_kernel_radius as usize / 2 + 2).clamp(2, 5);
    for pass in 0..corridor_passes {
        for y in 0..WORLD_HEIGHT {
            let t = (pass as f32 + y as f32 / WORLD_HEIGHT as f32) / corridor_passes as f32;
            progress.phase(progress_base, progress_span, 0.5 + 0.02 * t);
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let weak = weakness_field[i];
                let mut comp_sum = comp_field[i] * 0.44;
                let mut ext_sum = ext_field[i] * 0.44;
                let mut shear_sum = shear_field[i] * 0.44;
                let mut wsum = 0.44;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.1 } else { 0.065 };
                        comp_sum += comp_field[j] * w;
                        ext_sum += ext_field[j] * w;
                        shear_sum += shear_field[j] * w;
                        wsum += w;
                    }
                }
                let comp_avg = comp_sum / wsum.max(1e-6);
                let ext_avg = ext_sum / wsum.max(1e-6);
                let shear_avg = shear_sum / wsum.max(1e-6);
                comp_next[i] = clampf(
                    comp_field[i] * (0.6 - 0.12 * weak) + comp_avg * (0.3 + 0.32 * weak),
                    0.0,
                    1.0,
                );
                ext_next[i] = clampf(
                    ext_field[i] * (0.62 - 0.08 * weak) + ext_avg * (0.28 + 0.26 * (1.0 - weak)),
                    0.0,
                    1.0,
                );
                shear_next[i] = clampf(
                    shear_field[i] * (0.64 - 0.06 * weak)
                        + shear_avg * (0.24 + 0.22 * weak),
                    0.0,
                    1.0,
                );
            }
        }
        comp_field.copy_from_slice(&comp_next);
        ext_field.copy_from_slice(&ext_next);
        shear_field.copy_from_slice(&shear_next);
    }

    let mut orogen_drive = vec![0.0_f32; WORLD_SIZE];
    let mut crust_eq = vec![0.0_f32; WORLD_SIZE];
    let mut crust = vec![0.0_f32; WORLD_SIZE];
    let mut crust_next = vec![0.0_f32; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        let continental = continentality_field[i];
        let landness = smoothstep(-0.08, 0.84, continental);
        let eq = lerpf(7.0, 34.8, landness) * (1.02 - 0.14 * heat_norm_field[i]);
        crust_eq[i] = eq;
        let inherited = smoothstep(0.2, 0.9, 0.5 + 0.5 * fault_backbone[i]);
        let drive = smoothstep(
            0.1,
            0.9,
            comp_field[i] * (0.56 + 0.46 * weakness_field[i])
                + shear_field[i] * 0.18
                + inherited * 0.12 * weakness_field[i],
        ) * (0.42 + 0.58 * landness);
        orogen_drive[i] = drive;
        crust[i] = clampf(eq + drive * 2.3 - ext_field[i] * 1.2, 4.0, 60.0);
    }

    let time_steps =
        (detail.erosion_rounds * 9 + detail.max_kernel_radius as usize * 3 + 8).clamp(12, 40);
    for step in 0..time_steps {
        for y in 0..WORLD_HEIGHT {
            let t = (step as f32 + y as f32 / WORLD_HEIGHT as f32) / time_steps as f32;
            progress.phase(progress_base, progress_span, 0.5 + 0.12 * t);
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let landness = smoothstep(-0.08, 0.84, continentality_field[i]);
                let weak = weakness_field[i];
                let strength = strength_field[i];
                let comp_eff = comp_field[i] * (0.54 + 0.62 * weak)
                    + shear_field[i] * 0.24 * (0.32 + 0.68 * landness);
                let ext_eff = ext_field[i] * (0.64 + 0.36 * (1.0 - landness));
                let thickening = (0.08 + 0.7 * comp_eff)
                    * (0.42 + 0.58 * landness)
                    * (1.14 - 0.3 * strength);
                let thinning = (0.04 + 0.48 * ext_eff)
                    * (0.56 + 0.52 * heat_norm_field[i])
                    * (0.48 + 0.52 * (1.0 - landness));
                let relaxation = (crust[i] - crust_eq[i]).max(0.0)
                    * (0.017 + 0.03 * (1.0 - landness) + 0.01 * (step as f32 / time_steps as f32));
                let evolved = crust[i] + thickening - thinning - relaxation;

                let mut sum = evolved * 0.38;
                let mut wsum = 0.38;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.1 } else { 0.065 };
                        sum += crust[j] * w;
                        wsum += w;
                    }
                }
                let avg = sum / wsum.max(1e-6);
                let corridor = comp_field[i]
                    .max(shear_field[i] * 0.85)
                    .max(ext_field[i] * 0.55);
                let lateral = clampf(
                    0.034 + 0.064 * (1.0 - corridor) + 0.03 * (1.0 - weak),
                    0.022,
                    0.16,
                );
                crust_next[i] = clampf(evolved * (1.0 - lateral) + avg * lateral, 4.0, 78.0);
            }
        }
        crust.copy_from_slice(&crust_next);
    }

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.62 + 0.2 * (i as f32 / WORLD_SIZE as f32),
            );
        }
        let plate_id = plates.plate_field[i] as usize;
        let plate_speed = plates.plate_vectors[plate_id].speed;
        let heat = plates.plate_vectors[plate_id].heat;
        let plate_buoyancy = buoyancy_field[i];
        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let continental_raw = continentality_field[i];

        let warp_u = spherical_fbm(
            sx * 1.15 + 3.0,
            sy * 1.15 - 5.0,
            sz * 1.15 + 7.0,
            seed as f32 * 0.0019 + 11.0,
        );
        let warp_v = spherical_fbm(
            sx * 1.67 - 13.0,
            sy * 1.67 + 17.0,
            sz * 1.67 - 19.0,
            seed as f32 * 0.0023 + 29.0,
        );
        let wx = sx + 0.34 * warp_u + 0.2 * warp_v;
        let wy = sy + 0.34 * warp_v + 0.2 * warp_u;
        let wz = sz + 0.27 * (warp_u - warp_v);

        let macro_noise = spherical_fbm(
            wx * 0.82 + 5.0,
            wy * 0.82 - 7.0,
            wz * 0.82 + 9.0,
            seed as f32 * 0.0027 + 17.0,
        );
        let regional_signal = spherical_fbm(
            wx * 1.72 - 11.0,
            wy * 1.72 + 13.0,
            wz * 1.72 - 3.0,
            seed as f32 * 0.0035 + 37.0,
        );
        let micro_signal = spherical_fbm(
            wx * 3.85 + 19.0,
            wy * 3.85 - 17.0,
            wz * 3.85 + 23.0,
            seed as f32 * 0.0049 + 53.0,
        );
        let shelf_noise = spherical_fbm(
            wx * 1.3 + 29.0,
            wy * 1.3 - 31.0,
            wz * 1.3 + 23.0,
            seed as f32 * 0.0039 + 61.0,
        );

        let land_core = continental_raw.max(0.0);
        let ocean_core = (-continental_raw).max(0.0);
        let shelf_span = 0.18 + 0.16 * (0.5 + 0.5 * shelf_noise);
        let coast_weight = 1.0 - smoothstep(0.05, shelf_span.max(0.08), continental_raw.abs());
        let interior_weight = 1.0 - coast_weight;
        let mountain_patch = smoothstep(0.2, 0.88, 0.5 + 0.5 * regional_signal);

        let tectonic_scale = ((k_relief * plate_speed) / planet.gravity.max(1.0)) * 0.0135;
        let land_gate = smoothstep(0.1, 0.95, continental_raw);
        let ocean_gate = smoothstep(0.2, 0.98, -continental_raw);
        let comp = comp_field[i];
        let ext = ext_field[i];
        let shear = shear_field[i];
        let crust_anom = crust[i] - crust_eq[i];
        let orogen = smoothstep(0.08, 0.9, orogen_drive[i]);
        let plateau = smoothstep(3.0, 18.0, crust_anom) * smoothstep(0.24, 0.92, land_gate);
        let segmentation = smoothstep(
            0.16,
            0.88,
            weakness_field[i] * 0.78 + 0.22 * (0.5 + 0.5 * fault_backbone[i]),
        );

        let mountain_uplift = tectonic_scale
            * land_gate
            * (95.0
                + 1480.0 * orogen * (0.54 + 0.46 * segmentation)
                + 280.0 * plateau
                + 230.0 * clampf(crust_anom / 20.0, 0.0, 1.0))
            * (0.62 + 0.38 * mountain_patch)
            * (0.38 + 0.62 * interior_weight);
        let transpress_uplift = tectonic_scale
            * land_gate
            * (45.0 + 260.0 * shear.powf(1.08))
            * smoothstep(0.14, 0.8, shear + comp * 0.34);
        let ridge_ocean_uplift = tectonic_scale
            * ocean_gate
            * (42.0 + 640.0 * ext.powf(0.84))
            * (0.6 + 0.4 * (1.0 - orogen));
        let trench_cut = tectonic_scale
            * ocean_gate
            * (90.0 + 1730.0 * comp.powf(0.92))
            * (0.66 + 0.34 * ocean_core.powf(0.42));
        let transform_term = tectonic_scale
            * 140.0
            * shear.powf(0.9)
            * (regional_signal * 0.6 + micro_signal * 0.4)
            * (land_gate + ocean_gate).min(1.0);
        let base = mountain_uplift + transpress_uplift + ridge_ocean_uplift - trench_cut + transform_term;

        let continental_base = if continental_raw >= 0.0 {
            85.0 + land_core.powf(1.16) * (1480.0 + 2350.0 * smoothstep(0.18, 0.96, land_core))
        } else {
            -(170.0 + ocean_core.powf(1.18) * (1860.0 + 3180.0 * smoothstep(0.22, 0.98, ocean_core)))
        };

        let litho_variation = (regional_signal * 112.0
            + micro_signal * 58.0
            + macro_noise * 44.0
            + plate_buoyancy * 14.0
            + crust_anom * (14.0 + 26.0 * orogen))
            * (0.2 + 0.8 * interior_weight);
        let macro_base = continental_base + litho_variation;

        let noise = (random_seed.next_f32() - 0.5) * 10.0
            + regional_signal * 21.0
            + micro_signal * (6.5 + 9.0 * interior_weight);
        let heat_term = (heat - tectonics.mantle_heat * 0.5) * 0.95;

        relief[i] = base + macro_base + noise + heat_term;
    }

    let mut macro_blend = vec![0.0_f32; WORLD_SIZE];
    let macro_radius = 1_i32;

    for y in 0..WORLD_HEIGHT {
        progress.phase(
            progress_base,
            progress_span,
            0.6 + 0.1 * (y as f32 / WORLD_HEIGHT as f32),
        );
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let mut sum = 0.0_f32;
            let mut wsum = 0.0_f32;
            for oy in -macro_radius..=macro_radius {
                for ox in -macro_radius..=macro_radius {
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let dist2 = (ox * ox + oy * oy) as f32;
                    let w = (-dist2 / 1.9).exp();
                    sum += relief[j] * w;
                    wsum += w;
                }
            }
            macro_blend[i] = sum / wsum.max(1e-6);
        }
    }

    for i in 0..WORLD_SIZE {
        relief[i] = relief[i] * 0.9 + macro_blend[i] * 0.1;
    }

    let erosion_rounds = detail.erosion_rounds;
    let mut smoothed = relief.clone();
    let mut scratch = vec![0.0_f32; WORLD_SIZE];

    for round in 0..erosion_rounds {
        for y in 0..WORLD_HEIGHT {
            let round_t =
                (round as f32 + y as f32 / WORLD_HEIGHT as f32) / erosion_rounds.max(1) as f32;
            progress.phase(progress_base, progress_span, 0.72 + 0.14 * round_t);
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut sum = 0.0_f32;
                let mut count = 0.0_f32;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        sum += smoothed[j];
                        count += 1.0;
                    }
                }
                let avg = sum / count;
                scratch[i] = smoothed[i] * 0.95 + avg * 0.05;
            }
        }

        for i in 0..WORLD_SIZE {
            let drop = (smoothed[i] - scratch[i]).max(0.0);
            let height_loss = (drop * 0.14).min(16.0);
            smoothed[i] = scratch[i] - height_loss;
        }
    }

    relief.copy_from_slice(&smoothed);

    let ocean_cut = (((planet.ocean_percent / 100.0) * WORLD_SIZE as f32).floor() as isize)
        .clamp(0, (WORLD_SIZE - 1) as isize) as usize;

    let mut sorted = relief.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let sea_level = *sorted.get(ocean_cut).unwrap_or(&0.0);
    for h in relief.iter_mut() {
        *h -= sea_level;
    }

    stabilize_landmass_topology(&mut relief, planet, detail);
    apply_coastal_detail(&mut relief, seed, cache);
    defuse_plate_linearity(&mut relief, plates, seed, cache, detail);

    let mut sorted_after_coast = relief.clone();
    sorted_after_coast.sort_by(|a, b| a.total_cmp(b));
    let coast_recenter = *sorted_after_coast.get(ocean_cut).unwrap_or(&0.0);
    for h in relief.iter_mut() {
        *h -= coast_recenter;
    }

    normalize_height_range(&mut relief, planet, tectonics);
    if detail.erosion_rounds == 0 {
        defuse_orogenic_ribbons(&mut relief, seed_mix, cache, detail);
    }
    progress.phase(progress_base, progress_span, 0.86);
    apply_ocean_profile(
        &mut relief,
        &plates.boundary_types,
        &plates.boundary_strength,
        planet,
        seed,
        cache,
        detail,
        progress,
        progress_base + progress_span * 0.86,
        progress_span * 0.1,
    );
    reshape_ocean_boundaries(
        &mut relief,
        &plates.boundary_types,
        &plates.boundary_strength,
    );
    progress.phase(progress_base, progress_span, 0.98);

    let mut sorted_final = relief.clone();
    sorted_final.sort_by(|a, b| a.total_cmp(b));
    let final_recenter = *sorted_final.get(ocean_cut).unwrap_or(&0.0);
    for h in relief.iter_mut() {
        *h -= final_recenter;
    }
    progress.phase(progress_base, progress_span, 1.0);

    ReliefResult { relief, sea_level }
}

fn apply_events(
    planet: &PlanetInputs,
    relief: &[f32],
    events: &[WorldEventRecord],
    cache: &WorldCache,
) -> (Vec<f32>, f32) {
    let mut updated = relief.to_vec();
    let mut aerosol_index = 0.0_f32;

    let nearest_cell = |lat: f32, lon: f32| -> usize {
        let y = clampf(
            ((90.0 - lat) / (180.0 / WORLD_HEIGHT as f32)).round(),
            0.0,
            (WORLD_HEIGHT - 1) as f32,
        ) as usize;
        let x = ((((lon + 180.0) / (360.0 / WORLD_WIDTH as f32)).round() as i32
            % WORLD_WIDTH as i32)
            + WORLD_WIDTH as i32)
            % WORLD_WIDTH as i32;
        index(x as usize, y)
    };

    let km_per_cell_lat = KILOMETERS_PER_DEGREE * planet.radius_km / 6371.0;
    let km_per_cell_lon = |lat: f32, lon: f32| -> f32 {
        (KILOMETERS_PER_DEGREE * (lat * RADIANS).cos() * (lon * RADIANS).cos() * planet.radius_km
            / 6371.0)
            .max(1.0)
    };

    for event in events {
        if event.kind == "meteorite" {
            let radius_m = ((event.diameter_km * 1000.0) / 2.0).max(1.0);
            let mass = (4.0 / 3.0) * std::f32::consts::PI * radius_m.powi(3) * event.density_kg_m3;
            let energy = 0.5 * mass * (event.speed_kms * 1000.0).powi(2);
            let crater_radius_km = 8.0_f32.max(energy.powf(1.0 / 5.0) / 2500.0);
            let crater_depth = 9000.0_f32.min(800.0 + ((energy + 1.0).log10() - 10.0) * 250.0);
            let center_index = nearest_cell(event.latitude, event.longitude);
            let cx = center_index % WORLD_WIDTH;
            let cy = center_index / WORLD_WIDTH;

            let lat_span = crater_radius_km / km_per_cell_lat;
            let lon_span = crater_radius_km / km_per_cell_lon(event.latitude, event.longitude);

            let min_x = (cx as f32 - lon_span - 1.0).floor() as i32;
            let max_x = (cx as f32 + lon_span + 1.0).ceil() as i32;
            let min_y = clampf(
                (cy as f32 - lat_span - 1.0).floor(),
                0.0,
                (WORLD_HEIGHT - 1) as f32,
            ) as i32;
            let max_y = clampf(
                (cy as f32 + lat_span + 1.0).ceil(),
                0.0,
                (WORLD_HEIGHT - 1) as f32,
            ) as i32;

            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let mut wrapped_x = x % WORLD_WIDTH as i32;
                    if wrapped_x < 0 {
                        wrapped_x += WORLD_WIDTH as i32;
                    }
                    let target = index(wrapped_x as usize, y as usize);
                    let d_lat = cache.lat_by_y[target] - event.latitude;
                    let d_lon = (((cache.lon_by_x[target] - event.longitude + 180.0) % 360.0
                        + 360.0)
                        % 360.0)
                        - 180.0;
                    let d_km = ((d_lat * km_per_cell_lat).powi(2)
                        + (d_lon
                            * KILOMETERS_PER_DEGREE
                            * (cache.lat_by_y[target] * RADIANS).cos()
                            * planet.radius_km
                            / 6371.0)
                            .powi(2))
                    .sqrt();

                    if d_km <= crater_radius_km {
                        let falloff = 1.0 - (d_km / crater_radius_km).powi(2);
                        updated[target] -= crater_depth * falloff.max(0.0) * 0.5;
                        updated[target] = updated[target].max(-planet.radius_km * 10.0);
                    }
                }
            }

            aerosol_index += 0.45_f32.min((energy + 1.0).log10() / 18.0);
        } else if event.kind == "oceanShift" {
            let idx = nearest_cell(event.latitude, event.longitude);
            updated[idx] += event.magnitude * 0.5;
            for v in updated.iter_mut() {
                *v += event.magnitude * 0.15;
            }
        } else {
            let center_index = nearest_cell(event.latitude, event.longitude);
            let cx = center_index % WORLD_WIDTH;
            let cy = center_index / WORLD_WIDTH;
            let sign = if event.kind == "rift" { -1.0 } else { 1.0 };
            let magnitude = event.magnitude * 40.0 * sign;
            let radius_cells = ((event.radius_km
                / (KILOMETERS_PER_DEGREE * planet.radius_km / 6371.0))
                .round() as i32)
                .max(1);

            for y in (cy as i32 - radius_cells)..=(cy as i32 + radius_cells) {
                if y < 0 || y >= WORLD_HEIGHT as i32 {
                    continue;
                }
                for x in (cx as i32 - radius_cells)..=(cx as i32 + radius_cells) {
                    let mut wrapped_x = x % WORLD_WIDTH as i32;
                    if wrapped_x < 0 {
                        wrapped_x += WORLD_WIDTH as i32;
                    }
                    let target = index(wrapped_x as usize, y as usize);
                    let dx = x - cx as i32;
                    let dy = y - cy as i32;
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist <= radius_cells as f32 {
                        let falloff = (-dist / (radius_cells as f32).max(1.0)).exp();
                        updated[target] += magnitude * falloff;
                    }
                }
            }
        }
    }

    (updated, clampf(aerosol_index, 0.0, 1.0))
}

fn compute_slope(
    heights: &[f32],
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<f32>, f32, f32) {
    let mut slope = vec![0.0_f32; WORLD_SIZE];
    let mut min_slope = f32::INFINITY;
    let mut max_slope = 0.0_f32;

    for y in 0..WORLD_HEIGHT {
        progress.phase(progress_base, progress_span, y as f32 / WORLD_HEIGHT as f32);
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = heights[i];
            let mut max_drop = 0.0_f32;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let drop = (h - heights[j]).max(0.0);
                    if drop > max_drop {
                        max_drop = drop;
                    }
                }
            }
            slope[i] = max_drop;
            min_slope = min_slope.min(max_drop);
            max_slope = max_slope.max(max_drop);
        }
    }
    progress.phase(progress_base, progress_span, 1.0);

    (slope, min_slope, max_slope)
}

fn compute_hydrology(
    heights: &[f32],
    slope: &[f32],
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<i32>, Vec<f32>, Vec<f32>, Vec<u8>) {
    let mut flow_direction = vec![-1_i32; WORLD_SIZE];
    let mut flow_accumulation = vec![1.0_f32; WORLD_SIZE];

    for y in 0..WORLD_HEIGHT {
        progress.phase(
            progress_base,
            progress_span,
            0.45 * (y as f32 / WORLD_HEIGHT as f32),
        );
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = heights[i];
            let mut best_drop = 0.0_f32;
            let mut best = -1_i32;

            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let drop = h - heights[j];
                    if drop > best_drop {
                        best_drop = drop;
                        best = j as i32;
                    }
                }
            }

            flow_direction[i] = best;
        }
    }

    let mut order: Vec<usize> = (0..WORLD_SIZE).collect();
    order.sort_by(|a, b| heights[*b].total_cmp(&heights[*a]));
    progress.phase(progress_base, progress_span, 0.48);

    for (rank, i) in order.iter().copied().enumerate() {
        if rank % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.48 + 0.35 * (rank as f32 / WORLD_SIZE as f32),
            );
        }
        let to = flow_direction[i];
        if to >= 0 {
            flow_accumulation[to as usize] += flow_accumulation[i] * (1.0 + slope[i] / 1000.0);
        }
    }
    progress.phase(progress_base, progress_span, 0.83);

    let mut sorted = flow_accumulation.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let threshold_index = ((sorted.len() as f32 * 0.985).floor() as usize).min(sorted.len() - 1);
    let threshold = sorted[threshold_index].max(1.0);

    let mut rivers = vec![0.0_f32; WORLD_SIZE];
    let mut lakes = vec![0_u8; WORLD_SIZE];

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.83 + 0.17 * (i as f32 / WORLD_SIZE as f32),
            );
        }
        rivers[i] = flow_accumulation[i] / threshold;
        if flow_direction[i] < 0 && heights[i] < 0.0 && slope[i] < 40.0 {
            lakes[i] = 1;
        }
    }
    progress.phase(progress_base, progress_span, 1.0);

    (flow_direction, flow_accumulation, rivers, lakes)
}

fn compute_climate(
    planet: &PlanetInputs,
    heights: &[f32],
    slope: &[f32],
    rivers: &[f32],
    flow: &[f32],
    aerosol: f32,
    cache: &WorldCache,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<f32>, Vec<f32>, f32, f32, f32, f32) {
    let mut temperature = vec![0.0_f32; WORLD_SIZE];
    let mut precipitation = vec![0.0_f32; WORLD_SIZE];
    let tilt_rad = planet.axial_tilt_deg * RADIANS;
    let rotation_factor = 20.0 / planet.rotation_hours.max(1.0);

    let mut min_temp = f32::INFINITY;
    let mut max_temp = -f32::INFINITY;
    let mut min_prec = f32::INFINITY;
    let mut max_prec = -f32::INFINITY;

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(progress_base, progress_span, i as f32 / WORLD_SIZE as f32);
        }
        let lat = cache.lat_by_y[i] * RADIANS;
        let base_insolation = (lat - tilt_rad).cos().max(0.0) * (1.0 + planet.eccentricity * 0.35);
        let elevation_for_temperature = heights[i].max(0.0);
        let base_temp =
            55.0 * base_insolation - 0.0065 * elevation_for_temperature - lat.abs() * 0.4;
        let pressure_term = 12.0 * planet.atmosphere_bar.max(0.0).ln_1p();
        let ocean_term = if heights[i] < 0.0 { 6.0 } else { 0.0 };
        let seasonal = 1.0 + 0.08 * ((lat + 0.2 * std::f32::consts::PI * tilt_rad) * 2.0).sin();
        let temp =
            base_temp + pressure_term + ocean_term + 0.12 * rotation_factor * 100.0 * seasonal
                - aerosol * 8.0;
        let slope_orographic = 1.0 + (slope[i].abs() / 400.0).min(1.4);
        let precip_raw = 600.0
            + 1800.0 * base_insolation * seasonal
            + 1200.0 * rivers[i].min(1.0) * slope_orographic
            + flow[i] * 0.1;
        let precip = (precip_raw - aerosol * 250.0).max(0.0);

        temperature[i] = clampf(temp, -90.0, 70.0);
        precipitation[i] = precip;

        min_temp = min_temp.min(temperature[i]);
        max_temp = max_temp.max(temperature[i]);
        min_prec = min_prec.min(precip);
        max_prec = max_prec.max(precip);
    }
    progress.phase(progress_base, progress_span, 1.0);

    (
        temperature,
        precipitation,
        min_temp,
        max_temp,
        min_prec,
        max_prec,
    )
}

#[inline]
fn classify_biome(temp: f32, precip: f32, height: f32) -> u8 {
    if height < 0.0 {
        return 0;
    }
    if temp < -5.0 {
        return 1;
    }
    if precip > 1600.0 && temp >= 5.0 && temp < 22.0 {
        return 2;
    }
    if temp > 12.0 && precip > 900.0 {
        return 3;
    }
    if temp > 24.0 && precip > 500.0 && precip <= 1400.0 {
        return 4;
    }
    if precip < 400.0 && temp > 18.0 {
        return 5;
    }
    if precip > 700.0 && temp > 5.0 && temp < 22.0 {
        return 6;
    }
    if height > 1800.0 {
        return 7;
    }
    8
}

fn compute_biomes(temperature: &[f32], precipitation: &[f32], heights: &[f32]) -> Vec<u8> {
    let mut biomes = vec![0_u8; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        biomes[i] = classify_biome(temperature[i], precipitation[i], heights[i]);
    }
    biomes
}

fn compute_settlement(
    biomes: &[u8],
    heights: &[f32],
    temperature: &[f32],
    precipitation: &[f32],
) -> Vec<f32> {
    let mut settlement = vec![0.0_f32; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        if biomes[i] == 0 {
            settlement[i] = 0.0;
            continue;
        }
        let comfort_t = 1.0 - (temperature[i] - 18.0).abs() / 45.0;
        let comfort_p = 1.0 - (precipitation[i] - 1400.0).abs() / 2200.0;
        let elevation_penalty = (heights[i] - 1200.0).max(0.0) / 2600.0;
        settlement[i] = ((comfort_t + comfort_p) / 2.0 - elevation_penalty).clamp(0.0, 1.0);
    }
    settlement
}

#[wasm_bindgen]
pub struct WasmSimulationResult {
    width: u32,
    height: u32,
    seed: u32,
    sea_level: f32,
    radius_km: f32,
    ocean_percent: f32,
    recomputed_layers: Vec<String>,
    plates: Vec<i16>,
    boundary_types: Vec<i8>,
    height_map: Vec<f32>,
    slope_map: Vec<f32>,
    river_map: Vec<f32>,
    lake_map: Vec<u8>,
    flow_direction: Vec<i32>,
    flow_accumulation: Vec<f32>,
    temperature_map: Vec<f32>,
    precipitation_map: Vec<f32>,
    biome_map: Vec<u8>,
    settlement_map: Vec<f32>,
    min_height: f32,
    max_height: f32,
    min_temperature: f32,
    max_temperature: f32,
    min_precipitation: f32,
    max_precipitation: f32,
    min_slope: f32,
    max_slope: f32,
}

#[wasm_bindgen]
impl WasmSimulationResult {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> u32 {
        self.seed
    }

    #[wasm_bindgen(js_name = seaLevel)]
    pub fn sea_level(&self) -> f32 {
        self.sea_level
    }

    #[wasm_bindgen(js_name = radiusKm)]
    pub fn radius_km(&self) -> f32 {
        self.radius_km
    }

    #[wasm_bindgen(js_name = oceanPercent)]
    pub fn ocean_percent(&self) -> f32 {
        self.ocean_percent
    }

    #[wasm_bindgen(js_name = recomputedLayers)]
    pub fn recomputed_layers(&self) -> Array {
        self.recomputed_layers.iter().map(JsValue::from).collect()
    }

    pub fn plates(&self) -> Vec<i16> {
        self.plates.clone()
    }

    #[wasm_bindgen(js_name = boundaryTypes)]
    pub fn boundary_types(&self) -> Vec<i8> {
        self.boundary_types.clone()
    }

    #[wasm_bindgen(js_name = heightMap)]
    pub fn height_map(&self) -> Vec<f32> {
        self.height_map.clone()
    }

    #[wasm_bindgen(js_name = slopeMap)]
    pub fn slope_map(&self) -> Vec<f32> {
        self.slope_map.clone()
    }

    #[wasm_bindgen(js_name = riverMap)]
    pub fn river_map(&self) -> Vec<f32> {
        self.river_map.clone()
    }

    #[wasm_bindgen(js_name = lakeMap)]
    pub fn lake_map(&self) -> Vec<u8> {
        self.lake_map.clone()
    }

    #[wasm_bindgen(js_name = flowDirection)]
    pub fn flow_direction(&self) -> Vec<i32> {
        self.flow_direction.clone()
    }

    #[wasm_bindgen(js_name = flowAccumulation)]
    pub fn flow_accumulation(&self) -> Vec<f32> {
        self.flow_accumulation.clone()
    }

    #[wasm_bindgen(js_name = temperatureMap)]
    pub fn temperature_map(&self) -> Vec<f32> {
        self.temperature_map.clone()
    }

    #[wasm_bindgen(js_name = precipitationMap)]
    pub fn precipitation_map(&self) -> Vec<f32> {
        self.precipitation_map.clone()
    }

    #[wasm_bindgen(js_name = biomeMap)]
    pub fn biome_map(&self) -> Vec<u8> {
        self.biome_map.clone()
    }

    #[wasm_bindgen(js_name = settlementMap)]
    pub fn settlement_map(&self) -> Vec<f32> {
        self.settlement_map.clone()
    }

    #[wasm_bindgen(js_name = minHeight)]
    pub fn min_height(&self) -> f32 {
        self.min_height
    }

    #[wasm_bindgen(js_name = maxHeight)]
    pub fn max_height(&self) -> f32 {
        self.max_height
    }

    #[wasm_bindgen(js_name = minTemperature)]
    pub fn min_temperature(&self) -> f32 {
        self.min_temperature
    }

    #[wasm_bindgen(js_name = maxTemperature)]
    pub fn max_temperature(&self) -> f32 {
        self.max_temperature
    }

    #[wasm_bindgen(js_name = minPrecipitation)]
    pub fn min_precipitation(&self) -> f32 {
        self.min_precipitation
    }

    #[wasm_bindgen(js_name = maxPrecipitation)]
    pub fn max_precipitation(&self) -> f32 {
        self.max_precipitation
    }

    #[wasm_bindgen(js_name = minSlope)]
    pub fn min_slope(&self) -> f32 {
        self.min_slope
    }

    #[wasm_bindgen(js_name = maxSlope)]
    pub fn max_slope(&self) -> f32 {
        self.max_slope
    }
}

fn report_progress(progress_callback: Option<&js_sys::Function>, value: f32) {
    if let Some(cb) = progress_callback {
        let _ = cb.call1(
            &JsValue::NULL,
            &JsValue::from_f64(clampf(value, 0.0, 100.0) as f64),
        );
    }
}

struct ProgressTap<'a> {
    callback: Option<&'a js_sys::Function>,
    last: f32,
}

impl<'a> ProgressTap<'a> {
    fn new(callback: Option<&'a js_sys::Function>) -> Self {
        Self {
            callback,
            last: -1.0,
        }
    }

    fn emit(&mut self, value: f32) {
        let v = clampf(value, 0.0, 100.0);
        if v >= 100.0 || self.last < 0.0 || v - self.last >= 0.08 {
            report_progress(self.callback, v);
            self.last = v;
        }
    }

    fn phase(&mut self, base: f32, span: f32, t: f32) {
        self.emit(base + span * clampf(t, 0.0, 1.0));
    }
}

fn run_simulation_internal(
    config: JsValue,
    reason: String,
    progress_callback: Option<&js_sys::Function>,
) -> Result<WasmSimulationResult, JsValue> {
    let mut progress = ProgressTap::new(progress_callback);
    progress.emit(0.0);
    let mut cfg: SimulationConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("config deserialize error: {e}")))?;
    progress.emit(1.0);

    cfg.events = cfg.events.into_iter().map(ensure_event_energy).collect();

    let cache = world_cache();
    let recomputed_layers = evaluate_recompute(parse_reason(&reason));
    let preset = parse_generation_preset(cfg.generation_preset.as_deref());
    let detail = detail_profile(preset);

    let plates_layer = compute_plates(
        &cfg.planet,
        &cfg.tectonics,
        cfg.seed,
        cache,
        &mut progress,
        2.0,
        22.0,
    );
    let relief_raw = compute_relief(
        &cfg.planet,
        &cfg.tectonics,
        &plates_layer,
        cfg.seed,
        cache,
        detail,
        &mut progress,
        24.0,
        50.0,
    );
    progress.emit(74.0);
    let (event_relief, aerosol) = apply_events(&cfg.planet, &relief_raw.relief, &cfg.events, cache);
    progress.emit(78.0);

    let (slope_map, min_slope, max_slope) = compute_slope(&event_relief, &mut progress, 78.0, 8.0);
    let (flow_direction, flow_accumulation, river_map, lake_map) =
        compute_hydrology(&event_relief, &slope_map, &mut progress, 86.0, 7.0);

    let (
        temperature_map,
        precipitation_map,
        min_temperature,
        max_temperature,
        min_precipitation,
        max_precipitation,
    ) = compute_climate(
        &cfg.planet,
        &event_relief,
        &slope_map,
        &river_map,
        &flow_accumulation,
        aerosol,
        cache,
        &mut progress,
        93.0,
        5.0,
    );
    progress.emit(98.0);

    let biome_map = compute_biomes(&temperature_map, &precipitation_map, &event_relief);
    progress.emit(99.0);
    let settlement_map = compute_settlement(
        &biome_map,
        &event_relief,
        &temperature_map,
        &precipitation_map,
    );
    progress.emit(99.7);
    let (min_height, max_height) = min_max(&event_relief);
    let result = WasmSimulationResult {
        width: WORLD_WIDTH as u32,
        height: WORLD_HEIGHT as u32,
        seed: cfg.seed,
        sea_level: relief_raw.sea_level,
        radius_km: cfg.planet.radius_km,
        ocean_percent: cfg.planet.ocean_percent,
        recomputed_layers,
        plates: plates_layer.plate_field,
        boundary_types: plates_layer.boundary_types,
        height_map: event_relief,
        slope_map,
        river_map,
        lake_map,
        flow_direction,
        flow_accumulation,
        temperature_map,
        precipitation_map,
        biome_map,
        settlement_map,
        min_height,
        max_height,
        min_temperature,
        max_temperature,
        min_precipitation,
        max_precipitation,
        min_slope,
        max_slope,
    };

    // Keep progress <100 until JS wrapper and worker finish marshalling + posting result.
    progress.emit(99.9);
    Ok(result)
}

#[wasm_bindgen]
pub fn run_simulation(config: JsValue, reason: String) -> Result<WasmSimulationResult, JsValue> {
    run_simulation_internal(config, reason, None)
}

#[wasm_bindgen(js_name = run_simulation_with_progress)]
pub fn run_simulation_with_progress(
    config: JsValue,
    reason: String,
    progress_callback: js_sys::Function,
) -> Result<WasmSimulationResult, JsValue> {
    run_simulation_internal(config, reason, Some(&progress_callback))
}
