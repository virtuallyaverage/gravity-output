# About
I recently stumbled across a bunch of videos showing cool 2D gravity simulations. This intrigued me so I decided to try a project centered around it in rust, as well as 3D. 

This is a simple project exploring high-fidelity 3d gravity simulations. 

In the interest of learning about parallelizing in rust and gpu compute calculations, no algorithm that leveraged any sort of approximation was allowed. Various algorithms were explored resulting in a method using a wgsl compute shader and wgpu to brute force earlier algorithm enhancements. 

# Data
The focus of this project is performance in rust, so all data is exported to zipped binary files. The simulation files can be viewed from the included Unity program that reads in the files and plays them back. Although definately not ideal, and very storage intensive even with the zipped binary files, it does the job and allows the simulation program to be entirely decoupled from the UI.

The compression isn't really necessary at small particle values, but at larger ones it makes a difference.

# Algorithms
A couple algorithms were tried to refine approaches to work with large data numbers. They will be demonstrated at a high level in the following sections.

Rough benchmarks were completed by having each algorithm calculate 500 frames of simulation data and then finding the average time per frame from there.

Given the relatively cheap compute cost of the gravity-like calculations and the massive scale at which the interim operations are used, this project quickly became a game of optimizing for cache misses, function calls, and similar semantics. This resulted in some not very pretty code structure, which has been simplified away for the below examples.

## Naive approach
Since each particle affects each other particle without any sort of concession, the base algorigthm results in O(N^2) time. Making particle counts above 1k impracticly slow in most computers.
```rs
for particle in particles {
    let force = Vec3::ZERO;
    for sub_particle in particles {
        force += particle.interaction(sub_particle);
    }
    particle.apply(force);
}
```

## Split Force Accumulation
Recognizing that there are two steps to this algorithm, force accumulation and applying this force, allows for these two to be split out into their own steps. This split allows for parallelization on both ends and flattens the equation into O(N^2/Threads)+O(N/Threads). This massively benefits performance and helps get real-time simulation numbers for a few thousand particles.

I chose to use the `rayon` crate for it's simplicity. This resulted in a ~12x speedup in rough benchmarks over batches of 500 frames of simulation.

```rs
let forces = vec![Vec3::ZERO; PARTICLE_COUNT];
// collect forces in parallel
forces.par_iter_mut().enumerate().for_each(|(index, force)| {
    for particle2 in particles {
        force += particles[index].interaction(particles[index]);
    }
});

// apply forces in parallel
particles.par_iter_mut().enumerate().for_each(|idx, particle| {
    particle.apply(force)
});
```
*Note*:  This split method conveniently isolates each step and index so it can be safely written to without the type system complaining.

## Force Look Up Table
Since gravity is always in tension, the force vectors on two points always point towards each other. This means the force on particle `N1` is the inverse of `N2`, and we get the force value for `N1 -> N2` by taking `-(N1 <- N2)`. 

Algorithms up to this point have been naively calcuating both directions of vectors, resulting in an N^2 number of force calculations. Since memory size hasn't been an issue at all with these implementations, creating a Look Up Table (LUT) to trade memory for compute seems like a worthwhile venture.

### Processing index pairs
To facilitate the multi-threading a strategy needs to be taken so that we can safely manipulate data within the look up table. In other words, calculations need to be "divvy'd" up between threads such that they don't overlap. A simple and, definately not ideal approach, was to create a list of index pairs that can be ran over in parallel to keep the theme of massive threading going.

The pairs can be statically allocated since the number of particles won't be increased or decreased.

```rs
static PAIRS: LazyLock<Vec<(usize, usize)>> = LazyLock::new(|| {
    let mut pairs: Vec<(usize, usize)> = vec![];
    // only need to allocate half of them (not including diagonal)
    for i in 0..PARTICLE_COUNT {
        for j in (i + 1)..PARTICLE_COUNT {
            pairs.push((i, j));
        }
    }
    pairs
});
```

### Actual Simulation
We can know that no other threads will be accessing N1 -> N2 thanks to our `PAIRS` array, and ensure the mirror won't be accesed via a `ChunkManager` that uses cheap pointer arithmetic to allocate a unique chunk of `PAIRS` indices to each unique thread. With this knowledge we can safely use an unsafe block to manually set values without needing to worry about any mutex overheads, or locking delays.

I am aware this method is unoptimized and can see several opportunities for some pretty decent gains but the complexity of this method make it inferior to the previously mentioned method for particle numbers on the scale of a few thousand.

This version also impelements a flat look up table which helped ~10% overall peformance with cache optimization as the most likely culprit.

```rs
let chunk_manager = ChunkManager::new();

// fill look up table
(0..rayon::current_num_threads()).into_par_iter().for_each(|_| {
    let particles = PARTICLES.read().unwrap();
    
    while let Some((start, end)) = chunk_manager.next_chunk() {
        for idx in start..end {
            let (col_idx, row_idx) = PAIRS[idx];
            let result = particles[col_idx].get_influence(&particles[row_idx]);
            
            unsafe {
                let ptr = FORCE_LUT_FLAT.as_ptr() as *mut Vec<Vec3>;
                *ptr.add(get_force_index(col_idx, row_idx)) = result;
                *ptr.add(get_force_index(row_idx, col_idx)) = -result;
            }
        }
    }
});

// sum forces and apply result
particles.par_iter_mut().enumerate().zip().for_each(|idx, particle| {
    let start = idx * PARTICLE_COUNT;
    let force = FORCE_LUT_FLAT[start..start + PARTICLE_COUNT].iter().copied().sum();
    particle.apply(&force);
});
```

### LUT Result
The Look up table algorithm produced measurable but not ground-breaking results. (only a few tens of percent) Although I am reducing the number of calculations by millions at these scales, it is simply alot of overhead and memory accesses when compared to the dumber algorithms. In applications where the desired calculation doesn't fit into cache, or perhaps on CPU's that don't have the raw cache size of AMD's X3D chips, this algorithm likely holds a more advantageous position.

## The GPU
A GPU is specialized in doing massively multithreaded work loads with trivial input functions, which just happens to fit our workload profile perfectly. The only issue is the enourmous increase in complexity. 

Rather than try to harness vendor specific compute modules that would provide more specialized features, I decided to go with wgpu. This requires a `nbody.wgsl` compute shader, as well as a gpu interface in rust to handle communicating with that shader. While more complex and alot more boilerplate it ends up with a simpler surface level interface.

### GPU Interface 
The semantics of dealing with gpu's is a little bit complicated with setting up buffers, initializing pipelines, and copying data. Overall though this can be obscured away into it's own interface so the bulk of it doesn't need to be addressed at once:
```rs
// GPU compute
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
```

### Computing a frame
Forces are still applied on the CPU for simplicity's sake, and with the assumption the sequential accesses make that half of the algorithm trivial to burn through in a few milliseconds at most.

```rs
// accumulate forces
let forces = pollster::block_on(GPU_COMPUTE.compute_forces(&particles));

// Apply forces on CPU
{
    let mut particles_mut = PARTICLES.write().unwrap();
    particles_mut
        .par_iter_mut()
        .enumerate()
        .map(|(idx, particle)| {
            let force = &forces[idx];
            particle.tick(&force);
        })
        .collect();
}
```

# Conclusion
Overall the naive GPU algorithm results in a 3x improvement over the look up table, which is still the best CPU native option. This gives an algorithm that can run 100k particles at full fidelity, with only a 5x simulation to playback ratio. While I can garuntee there is ALOT more performance on the table I believe the majority of what I can effectively learn from this project is finished, So I will leave it here.