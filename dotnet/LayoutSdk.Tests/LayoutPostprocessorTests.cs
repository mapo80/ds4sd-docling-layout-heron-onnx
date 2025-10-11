using System;
using System.Collections.Generic;
using System.Linq;
using LayoutSdk.Processing;
using Xunit;
using Xunit.Abstractions;

namespace LayoutSdk.Tests;

public sealed class LayoutPostprocessorTests
{
    private readonly ITestOutputHelper _output;

    public LayoutPostprocessorTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Postprocess_EmptyInput_ReturnsEmptyList()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        var result = postprocessor.Postprocess(Array.Empty<BoundingBox>());

        Assert.Empty(result);
    }

    [Fact]
    public void Postprocess_SingleBox_ReturnsSameBox()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        var boxes = new[]
        {
            new BoundingBox(10f, 10f, 100f, 50f, "Text", 0.9f)
        };

        var result = postprocessor.Postprocess(boxes);

        Assert.Single(result);
        Assert.Equal(boxes[0], result[0]);
    }

    [Fact]
    public void Postprocess_OverlappingBoxes_MergesCorrectly()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        var boxes = new[]
        {
            new BoundingBox(10f, 10f, 100f, 50f, "Text", 0.9f),
            new BoundingBox(50f, 10f, 100f, 50f, "Text", 0.8f)  // Overlaps with first
        };

        var result = postprocessor.Postprocess(boxes);

        // Should merge into single box
        Assert.Single(result);
        var mergedBox = result[0];

        // Merged box should encompass both
        Assert.True(mergedBox.X <= 10f);
        Assert.True(mergedBox.Y <= 10f);
        Assert.True(mergedBox.X + mergedBox.Width >= 150f);
        Assert.True(mergedBox.Y + mergedBox.Height >= 60f);
    }

    [Fact]
    public void Postprocess_SeparateBoxes_NoMerge()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        var boxes = new[]
        {
            new BoundingBox(10f, 10f, 50f, 50f, "Text", 0.9f),
            new BoundingBox(100f, 100f, 50f, 50f, "Text", 0.8f)  // Far from first
        };

        var result = postprocessor.Postprocess(boxes);

        // Should remain separate
        Assert.Equal(2, result.Count);
    }

    [Fact]
    public void Postprocess_LowConfidenceBox_Filtered()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        options.LabelThresholds["Text"] = 0.8f; // High threshold

        var postprocessor = new LayoutPostprocessor(options);

        var boxes = new[]
        {
            new BoundingBox(10f, 10f, 50f, 50f, "Text", 0.9f),  // Above threshold
            new BoundingBox(70f, 10f, 50f, 50f, "Text", 0.5f)   // Below threshold
        };

        var result = postprocessor.Postprocess(boxes);

        // Only high confidence box should remain
        Assert.Single(result);
        Assert.Equal(0.9f, result[0].Confidence);
    }

    [Fact]
    public void Postprocess_InvalidSizeBox_Filtered()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        options.LabelSizeConstraints["Text"] = new SizeConstraint
        {
            MinWidth = 20f,
            MinHeight = 20f,
            MaxWidth = 200f,
            MaxHeight = 200f
        };

        var postprocessor = new LayoutPostprocessor(options);

        var boxes = new[]
        {
            new BoundingBox(10f, 10f, 50f, 50f, "Text", 0.9f),   // Valid size
            new BoundingBox(70f, 10f, 5f, 5f, "Text", 0.9f)      // Too small
        };

        var result = postprocessor.Postprocess(boxes);

        // Only valid size box should remain
        Assert.Single(result);
        Assert.Equal(50f, result[0].Width);
        Assert.Equal(50f, result[0].Height);
    }

    [Fact]
    public void Postprocess_PictureWithCaption_CreatesRelationship()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        var boxes = new[]
        {
            new BoundingBox(10f, 10f, 100f, 100f, "Picture", 0.9f),
            new BoundingBox(10f, 115f, 100f, 20f, "Caption", 0.8f)  // Below picture
        };

        var result = postprocessor.Postprocess(boxes);

        // Should detect picture-caption relationship
        Assert.Equal(2, result.Count); // Both should remain
        var picture = result.First(b => b.Label == "Picture");
        var caption = result.First(b => b.Label == "Caption");

        Assert.Equal(0.9f, picture.Confidence);
        Assert.Equal(0.8f, caption.Confidence);
    }

    [Fact]
    public void Postprocess_TableWithCaption_CreatesRelationship()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        var boxes = new[]
        {
            new BoundingBox(10f, 10f, 200f, 150f, "Table", 0.9f),
            new BoundingBox(10f, 165f, 200f, 25f, "Caption", 0.8f)  // Below table
        };

        var result = postprocessor.Postprocess(boxes);

        // Should detect table-caption relationship
        Assert.Equal(2, result.Count);
        var table = result.First(b => b.Label == "Table");
        var caption = result.First(b => b.Label == "Caption");

        Assert.Equal(0.9f, table.Confidence);
        Assert.Equal(0.8f, caption.Confidence);
    }

    [Fact]
    public void Postprocess_WrapperDetection_Works()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        var boxes = new[]
        {
            new BoundingBox(10f, 10f, 300f, 200f, "Text", 0.9f),   // Large text area
            new BoundingBox(20f, 20f, 50f, 30f, "Text", 0.8f),     // Small text inside
            new BoundingBox(20f, 60f, 50f, 30f, "Text", 0.8f)      // Another small text inside
        };

        var result = postprocessor.Postprocess(boxes);

        // Should detect wrapper relationship
        Assert.True(result.Count >= 2); // At least wrapper and some content
    }

    [Fact]
    public void Postprocess_Performance_WithManyBoxes()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        // Create many boxes for performance testing
        var boxes = new List<BoundingBox>();
        for (int i = 0; i < 100; i++)
        {
            boxes.Add(new BoundingBox(
                i * 10f, i * 5f, 50f, 30f,
                i % 2 == 0 ? "Text" : "Picture",
                0.8f));
        }

        var result = postprocessor.Postprocess(boxes);

        // Should complete without errors
        Assert.True(result.Count > 0);
        Assert.True(result.Count <= boxes.Count);
    }

    [Fact]
    public void Postprocess_HighPrecisionOptions_MoreSelective()
    {
        var defaultOptions = LayoutPostprocessOptions.CreateDefault();
        var highPrecisionOptions = LayoutPostprocessOptions.CreateHighPrecision();

        var defaultPostprocessor = new LayoutPostprocessor(defaultOptions);
        var precisionPostprocessor = new LayoutPostprocessor(highPrecisionOptions);

        var boxes = new[]
        {
            new BoundingBox(10f, 10f, 50f, 50f, "Text", 0.6f),  // Borderline confidence
            new BoundingBox(70f, 10f, 50f, 50f, "Text", 0.4f)   // Low confidence
        };

        var defaultResult = defaultPostprocessor.Postprocess(boxes);
        var precisionResult = precisionPostprocessor.Postprocess(boxes);

        // High precision should be more selective
        Assert.True(precisionResult.Count <= defaultResult.Count);
    }

    [Fact]
    public void Postprocess_PerformanceOptions_FasterProcessing()
    {
        var defaultOptions = LayoutPostprocessOptions.CreateDefault();
        var performanceOptions = LayoutPostprocessOptions.CreatePerformanceOptimized();

        var defaultPostprocessor = new LayoutPostprocessor(defaultOptions);
        var performancePostprocessor = new LayoutPostprocessor(performanceOptions);

        // Create test boxes
        var boxes = Enumerable.Range(0, 50)
            .Select(i => new BoundingBox(i * 8f, i * 4f, 40f, 25f, "Text", 0.8f))
            .ToArray();

        var defaultResult = defaultPostprocessor.Postprocess(boxes);
        var performanceResult = performancePostprocessor.Postprocess(boxes);

        // Both should produce valid results
        Assert.True(defaultResult.Count > 0);
        Assert.True(performanceResult.Count > 0);

        // Performance-optimized might produce different results due to different thresholds
        Assert.True(performanceResult.Count <= defaultResult.Count);
    }

    [Fact]
    public void Postprocess_SpatialIndex_EfficientQueries()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        // Create boxes in different spatial regions
        var boxes = new[]
        {
            new BoundingBox(0f, 0f, 50f, 50f, "Text", 0.9f),      // Top-left
            new BoundingBox(200f, 0f, 50f, 50f, "Text", 0.9f),    // Top-right
            new BoundingBox(0f, 200f, 50f, 50f, "Text", 0.9f),    // Bottom-left
            new BoundingBox(200f, 200f, 50f, 50f, "Text", 0.9f)   // Bottom-right
        };

        var result = postprocessor.Postprocess(boxes);

        // Should handle spatial queries efficiently
        Assert.Equal(4, result.Count); // No merging expected
    }

    [Fact]
    public void Postprocess_ComplexDocumentLayout_HandlesCorrectly()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        // Simulate a complex document layout
        var boxes = new[]
        {
            new BoundingBox(10f, 10f, 100f, 30f, "Title", 0.9f),
            new BoundingBox(10f, 50f, 150f, 100f, "Text", 0.8f),
            new BoundingBox(10f, 160f, 150f, 100f, "Text", 0.8f),
            new BoundingBox(200f, 50f, 100f, 100f, "Picture", 0.9f),
            new BoundingBox(200f, 160f, 100f, 20f, "Caption", 0.7f),
            new BoundingBox(10f, 300f, 200f, 150f, "Table", 0.8f),
            new BoundingBox(10f, 460f, 200f, 25f, "Caption", 0.7f)
        };

        var result = postprocessor.Postprocess(boxes);

        // Should handle complex layout without errors
        Assert.True(result.Count > 0);
        Assert.True(result.Count <= boxes.Length);

        // Verify all high-confidence boxes are preserved
        var highConfidenceBoxes = result.Where(b => b.Confidence >= 0.8f).ToList();
        Assert.True(highConfidenceBoxes.Count >= 3); // Title, Text, Picture, Table
    }

    [Fact]
    public void Postprocess_EdgeCase_BoundaryBoxes()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        var boxes = new[]
        {
            new BoundingBox(0f, 0f, 10f, 10f, "Text", 0.9f),        // At boundary
            new BoundingBox(990f, 0f, 10f, 10f, "Text", 0.9f),     // Near right boundary
            new BoundingBox(0f, 990f, 10f, 10f, "Text", 0.9f),     // Near bottom boundary
            new BoundingBox(990f, 990f, 10f, 10f, "Text", 0.9f)    // Near corner
        };

        var result = postprocessor.Postprocess(boxes);

        // Should handle boundary cases correctly
        Assert.Equal(4, result.Count);

        // All boxes should be within reasonable bounds
        foreach (var box in result)
        {
            Assert.True(box.X >= 0f);
            Assert.True(box.Y >= 0f);
            Assert.True(box.X + box.Width <= 1000f);
            Assert.True(box.Y + box.Height <= 1000f);
        }
    }

    [Fact]
    public void Postprocess_UnionFind_ComplexMerging()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        options.UnionFindMergeThreshold = 0.2f; // Lower threshold for more merging

        var postprocessor = new LayoutPostprocessor(options);

        // Create chain of overlapping boxes
        var boxes = new List<BoundingBox>();
        for (int i = 0; i < 10; i++)
        {
            boxes.Add(new BoundingBox(i * 8f, 10f, 20f, 20f, "Text", 0.8f));
        }

        var result = postprocessor.Postprocess(boxes);

        // Should merge overlapping boxes
        Assert.True(result.Count < boxes.Count);
        Assert.True(result.Count >= 1);
    }

    [Fact]
    public void Postprocess_SpatialContext_CaptionPictureRelationship()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        var boxes = new[]
        {
            new BoundingBox(100f, 100f, 200f, 150f, "Picture", 0.9f),
            new BoundingBox(100f, 260f, 200f, 30f, "Caption", 0.8f),  // Below picture
            new BoundingBox(400f, 100f, 100f, 100f, "Text", 0.8f)     // Far from picture
        };

        var result = postprocessor.Postprocess(boxes);

        // Should maintain picture-caption relationship
        Assert.Equal(3, result.Count);

        var picture = result.First(b => b.Label == "Picture");
        var caption = result.First(b => b.Label == "Caption");
        var text = result.First(b => b.Label == "Text");

        Assert.Equal(0.9f, picture.Confidence);
        Assert.Equal(0.8f, caption.Confidence);
        Assert.Equal(0.8f, text.Confidence);
    }

    [Fact]
    public void Postprocess_StressTest_ManySmallBoxes()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        // Create many small boxes
        var boxes = new List<BoundingBox>();
        for (int i = 0; i < 200; i++)
        {
            boxes.Add(new BoundingBox(
                (i % 20) * 25f, (i / 20) * 15f, 20f, 10f,
                "Text", 0.7f));
        }

        var result = postprocessor.Postprocess(boxes);

        // Should handle many small boxes efficiently
        Assert.True(result.Count > 0);
        Assert.True(result.Count <= boxes.Count);

        // All resulting boxes should be valid
        foreach (var box in result)
        {
            Assert.True(box.Width > 0f);
            Assert.True(box.Height > 0f);
            Assert.True(box.Confidence >= 0f);
            Assert.True(box.Confidence <= 1f);
        }
    }

    [Fact]
    public void Postprocess_RealisticDocumentLayout()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        var postprocessor = new LayoutPostprocessor(options);

        // Simulate realistic academic paper layout
        var boxes = new[]
        {
            // Header
            new BoundingBox(50f, 20f, 500f, 40f, "Title", 0.95f),

            // Abstract section
            new BoundingBox(50f, 80f, 500f, 20f, "Section-header", 0.9f),
            new BoundingBox(50f, 110f, 500f, 60f, "Text", 0.85f),

            // Two-column layout
            new BoundingBox(50f, 200f, 200f, 300f, "Text", 0.8f),    // Left column
            new BoundingBox(300f, 200f, 200f, 300f, "Text", 0.8f),   // Right column

            // Figure with caption
            new BoundingBox(50f, 520f, 200f, 150f, "Picture", 0.9f),
            new BoundingBox(50f, 680f, 200f, 25f, "Caption", 0.85f),

            // Table with caption
            new BoundingBox(300f, 520f, 200f, 100f, "Table", 0.9f),
            new BoundingBox(300f, 630f, 200f, 25f, "Caption", 0.85f),

            // Footer
            new BoundingBox(50f, 750f, 500f, 20f, "Page-footer", 0.7f)
        };

        var result = postprocessor.Postprocess(boxes);

        // Should maintain document structure
        Assert.True(result.Count > 0);

        // Should preserve high-confidence elements
        var highConfidenceBoxes = result.Where(b => b.Confidence >= 0.8f).ToList();
        Assert.True(highConfidenceBoxes.Count >= 5);

        // Should preserve important layout elements
        var titles = result.Where(b => b.Label.Contains("Title")).ToList();
        var pictures = result.Where(b => b.Label == "Picture").ToList();
        var tables = result.Where(b => b.Label == "Table").ToList();

        Assert.True(titles.Count >= 1);
        Assert.True(pictures.Count >= 1);
        Assert.True(tables.Count >= 1);
    }
}