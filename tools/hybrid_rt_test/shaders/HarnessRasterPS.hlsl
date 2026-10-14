//
// Hybrid ray tracing test harness.
//

float4 PSMain(in float4 iPosition : SV_Position, in float4 iColor : COLOR0) : SV_TARGET {
    return iColor;
}
