using System.Collections.Generic;

namespace LayoutSdk.Inference;

public sealed record LayoutBackendResult(IReadOnlyList<BoundingBox> Boxes);
