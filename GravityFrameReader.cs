using System;
using System.IO;
using System.IO.Compression;
using System.Numerics;

public class GravityFrameReader : IDisposable
{
    private readonly GZipStream _gzipStream;
    private readonly BinaryReader _reader;
    private readonly int _framesPerFile;
    private readonly int _numParticles;
    private int _currentFrame;
    private bool _disposed = false;

    public int FramesPerFile => _framesPerFile;
    public int NumParticles => _numParticles;
    public int CurrentFrame => _currentFrame;
    public bool HasMoreFrames => _currentFrame < _framesPerFile;

    public GravityFrameReader(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"File not found: {filePath}");

        try
        {
            var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            _gzipStream = new GZipStream(fileStream, CompressionMode.Decompress);
            _reader = new BinaryReader(_gzipStream);

            // Read header
            _framesPerFile = _reader.ReadInt32(); // little-endian by default
            _numParticles = _reader.ReadInt32();
            _currentFrame = 0;

            Console.WriteLine($"Loaded file: {Path.GetFileName(filePath)}");
            Console.WriteLine($"Frames per file: {_framesPerFile}");
            Console.WriteLine($"Particles per frame: {_numParticles}");
        }
        catch (Exception ex)
        {
            Dispose();
            throw new InvalidOperationException($"Failed to initialize GravityFrameReader: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Loads the next frame of particle position data into memory and returns it.
    /// </summary>
    /// <returns>Array of Vector3 positions for all particles in the frame, or null if no more frames</returns>
    public Vector3[] GetNextFrame()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GravityFrameReader));

        if (!HasMoreFrames)
            return null;

        try
        {
            var positions = new Vector3[_numParticles];

            for (int i = 0; i < _numParticles; i++)
            {
                // Read Vec3 as 3 consecutive float32 values (little-endian)
                float x = _reader.ReadSingle();
                float y = _reader.ReadSingle();
                float z = _reader.ReadSingle();
                
                positions[i] = new Vector3(x, y, z);
            }

            _currentFrame++;
            return positions;
        }
        catch (EndOfStreamException)
        {
            Console.WriteLine($"Reached end of file at frame {_currentFrame}");
            return null;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Error reading frame {_currentFrame}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Attempts to skip to a specific frame number (0-indexed).
    /// Note: This requires reading and discarding data, so it's not very efficient for large skips.
    /// </summary>
    /// <param name="frameNumber">Target frame number (0-indexed)</param>
    /// <returns>True if successful, false if frame number is invalid</returns>
    public bool SeekToFrame(int frameNumber)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GravityFrameReader));

        if (frameNumber < 0 || frameNumber >= _framesPerFile)
            return false;

        if (frameNumber < _currentFrame)
        {
            throw new InvalidOperationException("Cannot seek backwards. Create a new reader instance to restart from the beginning.");
        }

        // Skip frames until we reach the target
        while (_currentFrame < frameNumber)
        {
            if (GetNextFrame() == null)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Gets information about the current reader state
    /// </summary>
    public GravityReaderInfo GetInfo()
    {
        return new GravityReaderInfo
        {
            FramesPerFile = _framesPerFile,
            NumParticles = _numParticles,
            CurrentFrame = _currentFrame,
            HasMoreFrames = HasMoreFrames,
            ProgressPercentage = (_currentFrame / (float)_framesPerFile) * 100f
        };
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _reader?.Dispose();
            _gzipStream?.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Information about the gravity frame reader state
/// </summary>
public struct GravityReaderInfo
{
    public int FramesPerFile { get; set; }
    public int NumParticles { get; set; }
    public int CurrentFrame { get; set; }
    public bool HasMoreFrames { get; set; }
    public float ProgressPercentage { get; set; }

    public override string ToString()
    {
        return $"Frame {CurrentFrame}/{FramesPerFile} ({ProgressPercentage:F1}%) - {NumParticles} particles";
    }
}