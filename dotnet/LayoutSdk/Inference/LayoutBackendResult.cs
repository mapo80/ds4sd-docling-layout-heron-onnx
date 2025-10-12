using System;
using System.Linq;

namespace LayoutSdk.Inference;

public sealed class LayoutBackendResult
{
    public LayoutBackendResult(
        float[] logits,
        int[] logitsShape,
        float[] predictedBoxes,
        int[] predictedBoxesShape)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(logitsShape);
        ArgumentNullException.ThrowIfNull(predictedBoxes);
        ArgumentNullException.ThrowIfNull(predictedBoxesShape);

        if (logitsShape.Length != 3)
        {
            throw new ArgumentException("Logits tensor must have rank 3 [batch, queries, classes].", nameof(logitsShape));
        }

        if (predictedBoxesShape.Length != 3 || predictedBoxesShape[^1] != 4)
        {
            throw new ArgumentException("Boxes tensor must have rank 3 [batch, queries, 4].", nameof(predictedBoxesShape));
        }

        if (logitsShape[0] != predictedBoxesShape[0] || logitsShape[1] != predictedBoxesShape[1])
        {
            throw new ArgumentException("Logits and boxes tensors must share batch and query dimensions.");
        }

        Logits = logits;
        LogitsShape = logitsShape.ToArray();
        PredictedBoxes = predictedBoxes;
        PredictedBoxesShape = predictedBoxesShape.ToArray();
    }

    /// <summary>
    /// Flattened logits in [batch, queries, classes] order.
    /// </summary>
    public float[] Logits { get; }

    public int[] LogitsShape { get; }

    /// <summary>
    /// Flattened predicted boxes (cx, cy, w, h) in [batch, queries, 4] order.
    /// </summary>
    public float[] PredictedBoxes { get; }

    public int[] PredictedBoxesShape { get; }
}
