using LayoutSdk;
using LayoutSdk.Processing;
using System;
using Xunit;

namespace LayoutSdk.Tests;

public class OnnxRuntimeBackendTests
{
   [Fact]
   public void Infer_ThrowsNotSupportedException()
   {
       var backend = new OnnxRuntimeBackend("missing.onnx");
       using var tensor = ImageTensor.Rent(2, 2, 4);
       Assert.Throws<Microsoft.ML.OnnxRuntime.OnnxRuntimeException>(() => backend.Infer(tensor));
   }
}
