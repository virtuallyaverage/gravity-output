using System;
using System.IO;
using System.Numerics;

class Program
{
    static void Main(string[] args)
    {
        // Example usage of the GravityFrameReader
        string filePath = "output/batch_0000.bin.gz";
        
        if (!File.Exists(filePath))
        {
            Console.WriteLine($"File not found: {filePath}");
            Console.WriteLine("Make sure to run the Rust program first to generate the data files.");
            return;
        }

        try
        {
            using (var reader = new GravityFrameReader(filePath))
            {
                Console.WriteLine($"File loaded successfully!");
                Console.WriteLine($"Total frames in file: {reader.FramesPerFile}");
                Console.WriteLine($"Particles per frame: {reader.NumParticles}");
                Console.WriteLine();

                // Read and process each frame
                int frameCount = 0;
                while (reader.HasMoreFrames)
                {
                    Vector3[] frame = reader.GetNextFrame();
                    
                    if (frame == null)
                        break;

                    frameCount++;
                    
                    // Process the frame data
                    ProcessFrame(frame, frameCount);
                    
                    // Show progress every 10 frames
                    if (frameCount % 10 == 0)
                    {
                        var info = reader.GetInfo();
                        Console.WriteLine($"Progress: {info}");
                    }
                }

                Console.WriteLine($"\nProcessed {frameCount} frames successfully!");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    static void ProcessFrame(Vector3[] particles, int frameNumber)
    {
        // Example processing: calculate center of mass and print some statistics
        if (particles.Length == 0) return;

        Vector3 centerOfMass = Vector3.Zero;
        float minDistance = float.MaxValue;
        float maxDistance = 0f;

        // Calculate center of mass
        foreach (var particle in particles)
        {
            centerOfMass += particle;
        }
        centerOfMass /= particles.Length;

        // Calculate min/max distances from center
        foreach (var particle in particles)
        {
            float distance = Vector3.Distance(particle, centerOfMass);
            minDistance = Math.Min(minDistance, distance);
            maxDistance = Math.Max(maxDistance, distance);
        }

        // Print statistics for first few frames and every 10th frame
        if (frameNumber <= 3 || frameNumber % 10 == 0)
        {
            Console.WriteLine($"Frame {frameNumber:D3}: " +
                            $"CoM=({centerOfMass.X:F1}, {centerOfMass.Y:F1}, {centerOfMass.Z:F1}) " +
                            $"Spread: {minDistance:F1} - {maxDistance:F1}");
        }
    }

    // Example of reading specific frames
    static void ReadSpecificFrames()
    {
        string filePath = "output/batch_0000.bin.gz";
        
        using (var reader = new GravityFrameReader(filePath))
        {
            // Read first frame
            Vector3[] firstFrame = reader.GetNextFrame();
            if (firstFrame != null)
            {
                Console.WriteLine($"First frame has {firstFrame.Length} particles");
                Console.WriteLine($"First particle position: {firstFrame[0]}");
            }

            // Skip to frame 10 (if it exists)
            if (reader.SeekToFrame(10))
            {
                Vector3[] tenthFrame = reader.GetNextFrame();
                if (tenthFrame != null)
                {
                    Console.WriteLine($"Tenth frame first particle: {tenthFrame[0]}");
                }
            }
        }
    }
}