use bytemuck::{Pod, Zeroable};
use flate2::Compression;
use flate2::write::GzEncoder;
use futures;
use glam::Vec3;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::io::Write;
use std::sync::{LazyLock, RwLock};
use std::time::Instant;

mod util;
use util::{Settings, init_particles, load_settings};

static SETTINGS: LazyLock<Settings> = LazyLock::new(|| load_settings());
static PARTICLES: LazyLock<RwLock<Vec<Particle>>> = LazyLock::new(|| {
    let particles = init_particles();
    println!("Done with particle init");
    RwLock::new(particles)
});
static GPU_COMPUTE: LazyLock<GpuCompute> =
    LazyLock::new(|| pollster::block_on(GpuCompute::new(SETTINGS.num_particles)));

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParticle {
    pos: [f32; 3],
    mass: f32,
    vel: [f32; 3],
    _padding: f32,
}

struct GpuCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    particle_buffer: wgpu::Buffer,
    force_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl GpuCompute {
    async fn new(num_particles: usize) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        // Compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("N-Body Compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("nbody.wgsl").into()),
        });

        // Buffers
        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particles"),
            size: (num_particles * std::mem::size_of::<GpuParticle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let force_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Forces"),
            size: (num_particles * 16) as u64, // vec3 + padding
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Bind group layout and pipeline
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("N-Body Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: force_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            compute_pipeline,
            particle_buffer,
            force_buffer,
            bind_group,
        }
    }

    async fn compute_forces(&self, particles: &[Particle]) -> Vec<Vec3> {
        let num_particles = particles.len();

        // Convert to GPU format and upload
        let gpu_particles: Vec<GpuParticle> = particles
            .iter()
            .map(|p| GpuParticle {
                pos: [p.pos.x, p.pos.y, p.pos.z],
                mass: p.mass,
                vel: [p.vel.x, p.vel.y, p.vel.z],
                _padding: 0.0,
            })
            .collect();

        self.queue.write_buffer(
            &self.particle_buffer,
            0,
            bytemuck::cast_slice(&gpu_particles),
        );

        // Run compute shader
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                timestamp_writes: None,
                label: Some("N-Body Pass"),
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);

            // Launch with 64 threads per workgroup
            let workgroups = ((num_particles + 63) / 64) as u32;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Read back results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: (num_particles * 16) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.force_buffer,
            0,
            &staging_buffer,
            0,
            (num_particles * 16) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).unwrap();
        });

        let _ = self.device.poll(wgpu::wgt::PollType::Wait);
        receiver.await.unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let forces: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();

        forces.iter().map(|f| Vec3::new(f[0], f[1], f[2])).collect()
    }
}

/// GPU Force calculation
fn process_frame_group(frame_list: &mut Vec<Vec<Vec3>>, batch_num: usize) {
    for frame in frame_list.iter_mut() {
        let particles: Vec<Particle> = PARTICLES.read().unwrap().clone();

        // GPU compute
        let forces = pollster::block_on(GPU_COMPUTE.compute_forces(&particles));

        // Apply forces on CPU
        {
            let mut particles_mut = PARTICLES.write().unwrap();
            let positions: Vec<Vec3> = particles_mut
                .par_iter_mut()
                .enumerate()
                .map(|(idx, particle)| {
                    let force = &forces[idx];
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
