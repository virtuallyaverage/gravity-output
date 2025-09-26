struct Particle {
    pos: vec3<f32>,
    mass: f32,
    vel: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> forces: array<vec4<f32>>;

const WORKGROUP_SIZE: u32 = 64u;
const G_CONST: f32 = 0.01;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let num_particles = arrayLength(&particles);
    
    if (idx < num_particles) {
        var force = vec3<f32>(0.0);
        let pos_i = particles[idx].pos;
        let mass_i = particles[idx].mass;
        
        // Use shared memory for better performance
        var shared_particles: array<vec4<f32>, WORKGROUP_SIZE>;
        
        // Process in tiles
        for (var tile = 0u; tile < (num_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE; tile++) {
            let tile_idx = tile * WORKGROUP_SIZE + local_id.x;
            
            // Load tile to shared memory
            if (tile_idx < num_particles) {
                shared_particles[local_id.x] = vec4<f32>(
                    particles[tile_idx].pos, 
                    particles[tile_idx].mass
                );
            }
            workgroupBarrier();
            
            // Compute forces from this tile
            for (var j = 0u; j < WORKGROUP_SIZE; j++) {
                let global_j = tile * WORKGROUP_SIZE + j;
                if (global_j < num_particles && global_j != idx) {
                    let pos_j = shared_particles[j].xyz;
                    let mass_j = shared_particles[j].w;
                    
                    let diff = pos_j - pos_i;
                    let dist_sq = dot(diff, diff) + 0.001; // softening
                    let dist = sqrt(dist_sq);
                    let force_mag = G_CONST * mass_i * mass_j / dist_sq;
                    
                    force += (diff / dist) * force_mag;
                }
            }
            workgroupBarrier();
        }
        
        forces[idx] = vec4<f32>(force, 0.0);
    }
}