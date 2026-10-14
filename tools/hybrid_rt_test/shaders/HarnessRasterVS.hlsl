//
// Hybrid ray tracing test harness.
//
// The positions are already transformed on the CPU with the same chain of transformations RT64 uses
// (RSPProcessCS followed by RasterVS), so the vertex shader only passes them through.
//

void VSMain(in float4 iPosition : POSITION, in float4 iColor : COLOR, out float4 oPosition : SV_Position, out float4 oColor : COLOR0) {
    oPosition = iPosition;
    oColor = iColor;
}
