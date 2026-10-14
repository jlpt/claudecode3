//
// Hybrid ray tracing test harness.
//
// Packs a color texture into RGBA8 values in a buffer that can be read back by the CPU.
//

struct ReadbackCB {
    uint2 size;
};

[[vk::push_constant]] ConstantBuffer<ReadbackCB> gConstants : register(b0);
Texture2D<float4> gInput : register(t1);
RWStructuredBuffer<uint> gOutput : register(u2);

[numthreads(8, 8, 1)]
void CSMain(uint2 coord : SV_DispatchThreadID) {
    if (any(coord >= gConstants.size)) {
        return;
    }

    const uint4 v = uint4(round(saturate(gInput.Load(int3(coord, 0))) * 255.0f));
    gOutput[coord.y * gConstants.size.x + coord.x] = v.r | (v.g << 8) | (v.b << 16) | (v.a << 24);
}
