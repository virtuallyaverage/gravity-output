use std::ops::Neg;
use std::sync::atomic::Ordering;
use glam::Vec3;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::iter::IntoParallelRefIterator;
use std::io::Write;
use std::sync::{LazyLock, RwLock};
use std::time::Instant;
use atomic_float::AtomicF32;

mod util;
mod gpu;
use util::{Settings, init_particles, load_settings};

static SETTINGS: LazyLock<Settings> = LazyLock::new(|| load_settings());
static PARTICLES: LazyLock<RwLock<Vec<Particle>>> = LazyLock::new(|| {
    println!("Particle init");
    let particles = init_particles();
    println!("Done with particle init");
    RwLock::new(particles)
});

/// the summed forces for each particle value.
pub static FORCES: LazyLock<Vec<(AtomicF32, AtomicF32, AtomicF32)>> = LazyLock::new(|| {
    (0..SETTINGS.num_particles).map(|_| (AtomicF32::new(0.0), AtomicF32::new(0.0), AtomicF32::new(0.0))).collect()
});

pub fn process_forces(particles: &[Particle]) {
    particles.par_iter().enumerate().for_each(|(p1_idx, p1)| {
        particles[p1_idx+1..particles.len()].par_iter().enumerate().for_each(|(p2_idx, p2)| {
            let force = p1.get_influence(p2);
            add_force(p1_idx, (force.x, force.y, force.z));
            add_force(p2_idx, (force.x.neg(), force.y.neg(), force.z.neg()));
        });
    });
}

fn add_force(idx: usize, force: (f32, f32, f32)) {
    let current = &FORCES[idx];
    current.0.fetch_add(force.0, Ordering::Relaxed);
    current.1.fetch_add(force.1, Ordering::Relaxed);
    current.2.fetch_add(force.2, Ordering::Relaxed);
}

/// GPU Force calculation
fn process_frame_group(frame_list: &mut Vec<Vec<Vec3>>, batch_num: usize) {
    for frame in frame_list.iter_mut() {
        let particles: Vec<Particle> = PARTICLES.read().unwrap().clone();

        // GPU compute
        process_forces(&particles);

        // Apply forces on CPU
        {
            let mut particles_mut = PARTICLES.write().unwrap();
            let positions: Vec<Vec3> = particles_mut
                .par_iter_mut()
                .zip(FORCES.par_iter())
                .map(|(particle, force_atomic)| {
                    let force = Vec3::new(
                        force_atomic.0.load(std::sync::atomic::Ordering::Relaxed),
                        force_atomic.1.load(std::sync::atomic::Ordering::Relaxed),
                        force_atomic.2.load(std::sync::atomic::Ordering::Relaxed),
                    );
                    particle.tick(&force);
                    particle.pos
                })
                .collect();

            // Copy positions to frame
            frame.copy_from_slice(&positions);
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
        .join(format!("batch_{:04}.bin", batch_num));
    let mut file = std::fs::File::create(filename).unwrap();

    // header - convert to u32 for consistent 4-byte format
    file.write_all(&(SETTINGS.frames_per_file as u32).to_le_bytes())
        .unwrap();
    file.write_all(&(SETTINGS.num_particles as u32).to_le_bytes())
        .unwrap();

    for frame in frame_list.iter() {
        for pos in frame.iter() {
            file.write_all(bytemuck::bytes_of(pos)).unwrap();
        }
    }
}


fn main() {
    let mut frame_list: Vec<Vec<Vec3>> =
        vec![vec![Vec3::ZERO; SETTINGS.num_particles]; SETTINGS.frames_per_file];
    println!("Frame_lists");

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

#[derive(Clone)]
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
