use bytemuck;
use flate2::Compression;
use flate2::write::GzEncoder;
use glam::Vec3;
use rayon::prelude::*;
use std::io::Write;
use std::sync::{LazyLock, RwLock, atomic::{AtomicUsize, Ordering}};
use std::time::Instant;

mod util;
use util::{Settings, init_particles, load_settings};

static SETTINGS: LazyLock<Settings> = LazyLock::new(|| load_settings());
static PARTICLES: LazyLock<RwLock<Vec<Particle>>> = LazyLock::new(|| {
    let particles = init_particles();
    println!("Done with particle init");
    RwLock::new(particles)
});
static PAIRS: LazyLock<Vec<(usize, usize)>> = LazyLock::new(|| {
    let mut pairs: Vec<(usize, usize)> = vec![];
    // only need to allocate half of them (not including diagonal)
    for i in 0..SETTINGS.num_particles {
        for j in (i + 1)..SETTINGS.num_particles {
            pairs.push((i, j));
        }
    }
    pairs
});
static FORCE_LUT_FLAT: LazyLock<Vec<Vec3>> = LazyLock::new(|| {
    vec![Vec3::ZERO; SETTINGS.num_particles * SETTINGS.num_particles]
});

#[inline]
fn get_force_index(col: usize, row: usize) -> usize {
    col * SETTINGS.num_particles + row
}

/// how many pairs each thread should work on at a time
pub const LUT_CHUNK_SIZE: usize = 1000;
/// Provides the next list of chunks to process.
/// 
/// Would ideally not have to allocate
pub struct ChunkManager {
    next_chunk_idx: AtomicUsize,
    total_chunks: usize,
}

impl ChunkManager {
    pub fn new() -> Self {
        let total_pairs = PAIRS.len();
        Self {
            next_chunk_idx: AtomicUsize::new(0),
            total_chunks: (total_pairs + LUT_CHUNK_SIZE - 1) / LUT_CHUNK_SIZE,
        }
    }
    
    /// Returns (start_idx, end_idx) into PAIRS array
    /// No allocations, just index math
    pub fn next_chunk(&self) -> Option<(usize, usize)> {
        let chunk_idx = self.next_chunk_idx.fetch_add(1, Ordering::Relaxed);
        if chunk_idx >= self.total_chunks {
            return None;
        }
        
        let start = chunk_idx * LUT_CHUNK_SIZE;
        let end = ((chunk_idx + 1) * LUT_CHUNK_SIZE).min(PAIRS.len());
        Some((start, end))
    }
}

/// process a single files worth of frames.
fn process_frame_group(frame_list: &mut Vec<Vec<Vec3>>, batch_num: usize) {

    for frame in frame_list.iter_mut() {
        let chunk_manager = ChunkManager::new();
        // Fill each force pair in the look up table
        // needs to replace this with a more effecient version with 1Mil+ num-particles
        // doesn't work effeciently.

        (0..rayon::current_num_threads()).into_par_iter().for_each(|_| {
            let particles = PARTICLES.read().unwrap();
            
            while let Some((start, end)) = chunk_manager.next_chunk() {
                for idx in start..end {
                    let (col_idx, row_idx) = PAIRS[idx];
                    let result = particles[col_idx].get_influence(&particles[row_idx]);
                    
                    unsafe {
                        let ptr = FORCE_LUT_FLAT.as_ptr() as *mut Vec3;
                        *ptr.add(get_force_index(col_idx, row_idx)) = result;
                        *ptr.add(get_force_index(row_idx, col_idx)) = -result;
                    }
                }
            }
        });

        // propagate force and store result
        {
            let mut particles = PARTICLES.write().unwrap();
            for (idx, out_pos) in frame.iter_mut().enumerate() {
                let part = particles.get_mut(idx).unwrap();
                let start = idx * SETTINGS.num_particles;
                let force = FORCE_LUT_FLAT[start..start + SETTINGS.num_particles].iter().copied().sum();
                part.tick(&force);
                *out_pos = part.pos;
            }
        }
    }

    let start = Instant::now();
    write_frame_group(frame_list, &batch_num);
    println!("Took to save: {}", start.elapsed().as_secs_f32());
}

// Write batch of frames
fn write_frame_group(frame_list: &mut Vec<Vec<Vec3>>, batch_num: &usize) {
    let filename = SETTINGS
        .out_path
        .join(format!("batch_{:04}.bin.gz", batch_num));
    let file = std::fs::File::create(filename).unwrap();
    let mut encoder = GzEncoder::new(file, Compression::fast());

    // header - convert to u32 for consistent 4-byte format
    encoder
        .write_all(&(SETTINGS.frames_per_file as u32).to_le_bytes())
        .unwrap();
    encoder
        .write_all(&(SETTINGS.num_particles as u32).to_le_bytes())
        .unwrap();

    for frame in frame_list.iter() {
        for pos in frame.iter() {
            encoder.write_all(bytemuck::bytes_of(pos)).unwrap();
        }
    }
    encoder.finish().unwrap();
}

fn main() {
    let mut frame_list: Vec<Vec<Vec3>> =
        vec![vec![Vec3::ZERO; SETTINGS.num_particles]; SETTINGS.frames_per_file];

    let num_batches = SETTINGS.frames_total / SETTINGS.frames_per_file;
    for batch in 0..num_batches {
        let time_start = Instant::now();
        process_frame_group(&mut frame_list, batch.clone());
        println!(
            "Done with batch: {}, frames: {}-{}, Seconds: {} per frame: {}",
            batch,
            batch * SETTINGS.frames_per_file,
            (batch + 1) * SETTINGS.frames_per_file - 1,
            time_start.elapsed().as_secs_f32(),
            time_start.elapsed().as_secs_f32() / SETTINGS.frames_per_file as f32
        );
    }

    println!("Finished!");
}

pub struct Particle {
    mass: f32,
    pos: Vec3,
    vel: Vec3,
    acc: Vec3,
}

impl Particle {
    pub fn new(mass: f32, pos: Vec3, vel: Vec3, acc: Vec3) -> Particle {
        Particle {
            mass: mass,
            pos: pos,
            vel: vel,
            acc: acc,
        }
    }

    /// New with default values at zero
    pub fn new_zero() -> Particle {
        Particle {
            mass: 1.0,
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
            acc: Vec3::ZERO,
        }
    }

    /// Returns force that `self` experiences from other.
    ///
    /// Returns the force vector of influence
    pub fn get_influence(&self, other: &Particle) -> Vec3 {
        const EPSILON_SQ: f32 = 1e-8; // Pre-squared softening

        let r_vec = other.pos - self.pos;
        let r_sq = (r_vec).dot(r_vec).max(EPSILON_SQ);

        // Combined magnitude and direction calculation
        let force_over_r3 = SETTINGS.g_const * self.mass * other.mass / (r_sq * r_sq.sqrt());

        r_vec * force_over_r3
    }

    /// Propogate force accumulated over a tick into movement.
    pub fn tick(&mut self, force: &Vec3) {
        // Simple Euler integration (more stable for this system)
        self.acc = force / self.mass;
        self.vel += self.acc * SETTINGS.dt;
        self.pos += self.vel * SETTINGS.dt;
    }
}
