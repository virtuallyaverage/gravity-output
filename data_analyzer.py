"""
Simple data reader to test and understand the gravity simulation output format.
This can be used to verify data before importing into Blender.
"""

import gzip
import struct
import os

def read_gravity_batch(filepath):
    """Read a single batch file and return frame data"""
    frames = []
    
    print(f"Reading: {filepath}")
    
    with gzip.open(filepath, 'rb') as f:
        # Read header - checking what we actually get
        header_data = f.read(8)
        if len(header_data) < 8:
            print(f"  Error: Could not read header, got {len(header_data)} bytes")
            return frames
        
        frames_per_file = struct.unpack('<I', header_data[:4])[0]
        num_particles = struct.unpack('<I', header_data[4:8])[0]
        
        print(f"  Frames per file: {frames_per_file}")
        print(f"  Particles per frame: {num_particles}")
        
        # Debug: Check if we have the expected amount of data
        remaining_data = f.read()
        expected_bytes = frames_per_file * num_particles * 12  # 3 f32 values per particle
        print(f"  Expected data bytes: {expected_bytes}")
        print(f"  Actual data bytes: {len(remaining_data)}")
        
        if len(remaining_data) != expected_bytes:
            print(f"  Warning: Data size mismatch!")
            # Try to determine actual particle count
            if frames_per_file > 0:
                actual_particles = len(remaining_data) // (frames_per_file * 12)
                print(f"  Calculated particles per frame: {actual_particles}")
                num_particles = actual_particles
        
        # Reset file position and skip header again
        f.seek(8)
        
        # Read frame data
        for frame_idx in range(frames_per_file):
            frame_positions = []
            for particle_idx in range(num_particles):
                data = f.read(12)
                if len(data) < 12:
                    print(f"  Warning: Incomplete data at frame {frame_idx}, particle {particle_idx}")
                    break
                # Read Vec3 (3 f32 values in little-endian)
                x, y, z = struct.unpack('<fff', data)
                frame_positions.append((x, y, z))
            frames.append(frame_positions)
            
            if frame_idx == 0 and len(frame_positions) > 0:  # Show sample data from first frame
                print(f"  Sample positions from frame 0:")
                for i in range(min(5, len(frame_positions))):
                    x, y, z = frame_positions[i]
                    print(f"    Particle {i}: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    return frames

def export_to_obj_sequence(data_folder, output_folder, scale_factor=0.001):
    """Export each frame as an OBJ file for external use"""
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_counter = 0
    
    for batch_idx in range(12):  # 0-11 batches
        batch_file = os.path.join(data_folder, f"batch_{batch_idx:04d}.bin.gz")
        if not os.path.exists(batch_file):
            continue
        
        frames = read_gravity_batch(batch_file)
        
        for frame_data in frames:
            frame_counter += 1
            obj_filename = os.path.join(output_folder, f"frame_{frame_counter:04d}.obj")
            
            with open(obj_filename, 'w') as f:
                f.write("# Gravity simulation frame\n")
                f.write(f"# Frame {frame_counter}\n")
                f.write(f"# Particles: {len(frame_data)}\n\n")
                
                # Write vertices
                for i, (x, y, z) in enumerate(frame_data):
                    # Scale down the coordinates
                    scaled_x = x * scale_factor
                    scaled_y = y * scale_factor  
                    scaled_z = z * scale_factor
                    f.write(f"v {scaled_x:.6f} {scaled_y:.6f} {scaled_z:.6f}\n")
            
            if frame_counter % 50 == 0:
                print(f"Exported frame {frame_counter}")

def analyze_data_range(data_folder):
    """Analyze the coordinate ranges to determine appropriate scaling"""
    
    min_coords = [float('inf')] * 3
    max_coords = [float('-inf')] * 3
    
    for batch_idx in range(12):
        batch_file = os.path.join(data_folder, f"batch_{batch_idx:04d}.bin.gz")
        if not os.path.exists(batch_file):
            continue
        
        print(f"Analyzing batch {batch_idx}...")
        frames = read_gravity_batch(batch_file)
        
        for frame_data in frames:
            for x, y, z in frame_data:
                min_coords[0] = min(min_coords[0], x)
                min_coords[1] = min(min_coords[1], y)
                min_coords[2] = min(min_coords[2], z)
                max_coords[0] = max(max_coords[0], x)
                max_coords[1] = max(max_coords[1], y)
                max_coords[2] = max(max_coords[2], z)
    
    print("\nCoordinate ranges:")
    print(f"X: {min_coords[0]:.2f} to {max_coords[0]:.2f}")
    print(f"Y: {min_coords[1]:.2f} to {max_coords[1]:.2f}")
    print(f"Z: {min_coords[2]:.2f} to {max_coords[2]:.2f}")
    
    # Suggest scaling
    max_range = max(max_coords[i] - min_coords[i] for i in range(3))
    suggested_scale = 10.0 / max_range  # Scale to fit in 10 unit box
    print(f"\nSuggested scale factor: {suggested_scale:.6f}")
    
    return min_coords, max_coords, suggested_scale

if __name__ == "__main__":
    data_folder = r"output"  # Relative to this script
    
    print("=== Gravity Simulation Data Analysis ===\n")
    
    # First, analyze the data range
    min_coords, max_coords, suggested_scale = analyze_data_range(data_folder)
    
    # Test reading one batch
    print("\n=== Testing Batch Read ===")
    batch_0_file = os.path.join(data_folder, "batch_0000.bin.gz")
    if os.path.exists(batch_0_file):
        frames = read_gravity_batch(batch_0_file)
        print(f"Successfully read {len(frames)} frames")
    
    # Optionally export to OBJ sequence (uncomment to use)
    # print("\n=== Exporting OBJ Sequence ===")
    # export_to_obj_sequence(data_folder, "obj_output", suggested_scale)