use bytemuck::{Pod, Zeroable};
use futures;
use glam::Vec3;
use crate::{SETTINGS, Particle};


//pub static GPU_COMPUTE: LazyLock<GpuCompute> =
    //LazyLock::new(|| pollster::block_on(GpuCompute::new(SETTINGS.num_particles)));

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuParticle {
    pos: [f32; 3],
    mass: f32,
    vel: [f32; 3],
    _padding: f32,
}

pub struct GpuCompute {
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

    pub async fn compute_forces(&self, particles: &[Particle]) -> Vec<Vec3> {
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