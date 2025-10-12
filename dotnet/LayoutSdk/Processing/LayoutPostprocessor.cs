using System;
using System.Collections.Generic;
using LayoutSdk.Inference;

namespace LayoutSdk.Processing;

/// <summary>
/// Mirrors the HuggingFace <c>RTDetrImageProcessor.post_process_object_detection</c> implementation.
/// </summary>
public sealed class LayoutPostprocessor
{
    private readonly LayoutPostprocessOptions _options;

    public LayoutPostprocessor(LayoutPostprocessOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        if (_options.Labels == null || _options.Labels.Count == 0)
        {
            throw new ArgumentException("At least one label must be configured.", nameof(options));
        }
    }

    /// <summary>
    /// Converts raw model outputs into absolute bounding boxes aligned with HuggingFace's Python pipeline.
    /// </summary>
    /// <param name="backendResult">Raw logits and box predictions from the inference backend.</param>
    /// <param name="targetHeight">Original image height (in pixels).</param>
    /// <param name="targetWidth">Original image width (in pixels).</param>
    public IReadOnlyList<BoundingBox> Postprocess(LayoutBackendResult backendResult, int targetHeight, int targetWidth)
    {
        ArgumentNullException.ThrowIfNull(backendResult);

        if (targetHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetHeight));
        }

        if (targetWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetWidth));
        }

        var logitsShape = backendResult.LogitsShape;
        var boxesShape = backendResult.PredictedBoxesShape;

        if (logitsShape[0] != boxesShape[0])
        {
            throw new InvalidOperationException("Logits and boxes batch dimensions must match.");
        }

        var batchSize = logitsShape[0];
        if (batchSize != 1)
        {
            throw new NotSupportedException("Only batch size 1 is supported in the current pipeline.");
        }

        var numQueries = logitsShape[1];
        var numClasses = logitsShape[2];
        if (boxesShape[1] != numQueries || boxesShape[2] != 4)
        {
            throw new InvalidOperationException("Box tensor shape must be [batch, queries, 4].");
        }

        var normalizedBoxes = ConvertToCornerCoordinates(
            backendResult.PredictedBoxes,
            boxesShape,
            batchIndex: 0);

        var candidates = _options.UseFocalLoss
            ? SelectWithFocalLoss(backendResult.Logits, logitsShape, batchIndex: 0)
            : SelectWithSoftmax(backendResult.Logits, logitsShape, batchIndex: 0);

        if (candidates.Count == 0)
        {
            return Array.Empty<BoundingBox>();
        }

        var results = new List<BoundingBox>(candidates.Count);
        foreach (var candidate in candidates)
        {
            if (candidate.QueryIndex < 0 || candidate.QueryIndex >= numQueries)
            {
                continue;
            }

            if (candidate.ClassIndex < 0)
            {
                continue;
            }

            var label = ResolveLabel(candidate.ClassIndex);
            var threshold = _options.GetThreshold(label);
            if (!(candidate.Score > threshold))
            {
                continue;
            }

            var baseIndex = candidate.QueryIndex * 4;
            var x0 = Math.Clamp(normalizedBoxes[baseIndex + 0] * targetWidth, 0f, targetWidth);
            var y0 = Math.Clamp(normalizedBoxes[baseIndex + 1] * targetHeight, 0f, targetHeight);
            var x1 = Math.Clamp(normalizedBoxes[baseIndex + 2] * targetWidth, 0f, targetWidth);
            var y1 = Math.Clamp(normalizedBoxes[baseIndex + 3] * targetHeight, 0f, targetHeight);

            var width = x1 - x0;
            var height = y1 - y0;
            if (width <= 0f || height <= 0f)
            {
                continue;
            }

            results.Add(new BoundingBox(
                (float)x0,
                (float)y0,
                (float)width,
                (float)height,
                label,
                candidate.Score));
        }

        return results;
    }

    private IReadOnlyList<DetectionCandidate> SelectWithFocalLoss(
        float[] logits,
        int[] logitsShape,
        int batchIndex)
    {
        var numQueries = logitsShape[1];
        var numClasses = logitsShape[2];
        var totalEntries = numQueries * numClasses;

        var entries = new DetectionCandidate[totalEntries];
        var batchOffset = batchIndex * totalEntries;

        for (var query = 0; query < numQueries; query++)
        {
            for (var cls = 0; cls < numClasses; cls++)
            {
                var flatIndex = query * numClasses + cls;
                var logitIndex = batchOffset + flatIndex;
                var score = Sigmoid(logits[logitIndex]);
                entries[flatIndex] = new DetectionCandidate(score, query, cls, flatIndex);
            }
        }

        Array.Sort(entries, static (a, b) =>
        {
            var byScore = b.Score.CompareTo(a.Score);
            return byScore != 0 ? byScore : a.FlattenedIndex.CompareTo(b.FlattenedIndex);
        });

        var take = Math.Min(numQueries, entries.Length);
        if (take == entries.Length)
        {
            return entries;
        }

        var subset = new DetectionCandidate[take];
        Array.Copy(entries, subset, take);
        return subset;
    }

    private IReadOnlyList<DetectionCandidate> SelectWithSoftmax(
        float[] logits,
        int[] logitsShape,
        int batchIndex)
    {
        var numQueries = logitsShape[1];
        var numClasses = logitsShape[2];
        var entries = new DetectionCandidate[numQueries];
        var batchOffset = batchIndex * numQueries * numClasses;

        for (var query = 0; query < numQueries; query++)
        {
            var rowOffset = batchOffset + query * numClasses;
            var probabilities = ComputeSoftmax(logits, rowOffset, numClasses);

            var bestClass = 0;
            var bestScore = probabilities[0];
            // In non-focal mode the last class is the "no-object" class.
            var limit = numClasses - 1;
            for (var cls = 1; cls < limit; cls++)
            {
                var score = probabilities[cls];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = cls;
                }
            }

            entries[query] = new DetectionCandidate(
                bestScore,
                query,
                bestClass,
                query * limit + bestClass);
        }

        return entries;
    }

    private static float[] ConvertToCornerCoordinates(float[] boxes, int[] shape, int batchIndex)
    {
        var numQueries = shape[1];
        var stride = shape[2];

        var corners = new float[numQueries * stride];
        var batchOffset = batchIndex * numQueries * stride;

        for (var query = 0; query < numQueries; query++)
        {
            var sourceIndex = batchOffset + query * stride;
            var destinationIndex = query * stride;

            var cx = boxes[sourceIndex + 0];
            var cy = boxes[sourceIndex + 1];
            var width = boxes[sourceIndex + 2];
            var height = boxes[sourceIndex + 3];

            corners[destinationIndex + 0] = cx - (width / 2f);
            corners[destinationIndex + 1] = cy - (height / 2f);
            corners[destinationIndex + 2] = cx + (width / 2f);
            corners[destinationIndex + 3] = cy + (height / 2f);
        }

        return corners;
    }

    private string ResolveLabel(int classIndex)
    {
        if (classIndex >= 0 && classIndex < _options.Labels.Count)
        {
            return _options.Labels[classIndex];
        }

        return $"Label-{classIndex}";
    }

    private static float Sigmoid(float value)
    {
        var neg = -value;
        return 1f / (1f + MathF.Exp(neg));
    }

    private static float[] ComputeSoftmax(float[] logits, int offset, int length)
    {
        var slice = new float[length];
        var max = float.NegativeInfinity;
        for (var i = 0; i < length; i++)
        {
            var value = logits[offset + i];
            slice[i] = value;
            if (value > max)
            {
                max = value;
            }
        }

        var sum = 0f;
        for (var i = 0; i < length; i++)
        {
            slice[i] = MathF.Exp(slice[i] - max);
            sum += slice[i];
        }

        if (sum == 0f)
        {
            return slice;
        }

        for (var i = 0; i < length; i++)
        {
            slice[i] /= sum;
        }

        return slice;
    }

    private readonly struct DetectionCandidate
    {
        public DetectionCandidate(float score, int queryIndex, int classIndex, int flattenedIndex)
        {
            Score = score;
            QueryIndex = queryIndex;
            ClassIndex = classIndex;
            FlattenedIndex = flattenedIndex;
        }

        public float Score { get; }

        public int QueryIndex { get; }

        public int ClassIndex { get; }

        public int FlattenedIndex { get; }
    }
}
