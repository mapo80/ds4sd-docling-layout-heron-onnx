using System;
using LayoutSdk.Processing;
using Xunit;

namespace LayoutSdk.Tests;

public class OpenVinoBackendTests
{
   [Fact]
   public void Infer_ThrowsNotSupportedException()
   {
       var backend = new OpenVinoBackend("missing.xml", "missing.bin");
       using var tensor = ImageTensor.Rent(1, 1, 1);
       Assert.Throws<NotSupportedException>(() => backend.Infer(tensor));
   }
}
