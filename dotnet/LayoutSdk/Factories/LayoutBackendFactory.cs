using System;
using LayoutSdk.Configuration;

namespace LayoutSdk.Factories;

public sealed class LayoutBackendFactory : ILayoutBackendFactory
{
    private readonly LayoutSdkOptions _options;

    public LayoutBackendFactory(LayoutSdkOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.EnsureModelPaths();
    }

    public ILayoutBackend Create(LayoutRuntime runtime) => runtime switch
    {
        LayoutRuntime.OnnxRuntime => new OnnxRuntimeBackend(_options.OnnxModelPath),
        LayoutRuntime.OpenVino => new OpenVinoBackend(
            _options.OpenVino.ModelXmlPath,
            _options.OpenVino.WeightsBinPath),
        _ => throw new ArgumentOutOfRangeException(nameof(runtime), runtime, null)
    };
}
