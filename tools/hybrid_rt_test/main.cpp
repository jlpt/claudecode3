//
// Hybrid ray tracing test harness.
//
// Renders a synthetic N64-style scene with the exact chain of camera and viewport transformations RT64 uses for
// perspective geometry (RSPProcessCS followed by RasterVS), then runs RT64's HybridRaytracer on top of it and writes
// the results as PNG images. It works on any Vulkan device with ray tracing pipeline support, including Mesa's
// lavapipe software driver, which makes it possible to validate the pass without a GPU or the game.
//
// Usage: hybrid_rt_test [output directory] [hybrid ray tracing configuration JSON]
//

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "common/rt64_plume.h"
#include "render/rt64_hybrid_raytracing.h"

#include "shaders/HarnessRasterVS.hlsl.spirv.h"
#include "shaders/HarnessRasterPS.hlsl.spirv.h"
#include "shaders/HarnessReadbackCS.hlsl.spirv.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace plume {
    extern std::unique_ptr<RenderInterface> CreateVulkanInterface();
};

using namespace RT64;

namespace {
    const float Pi = 3.14159265358979f;
    const RenderFormat ColorFormat = RenderFormat::R8G8B8A8_UNORM;
    const RenderFormat DepthFormat = RenderFormat::D32_FLOAT;

    // N64 native framebuffer and the standard viewport used by most games (G_MAXZ / 2 for the depth scale and translation).
    const float NativeWidth = 320.0f;
    const float NativeHeight = 240.0f;
    const float DepthRange = 1024.0f;
    const float ViewportDepth = 511.0f / DepthRange;

    struct RasterVertex {
        float position[4];
        float color[4];
    };

    struct SceneObject {
        std::vector<hlslpp::float3> positions;
        std::vector<uint32_t> indices;
        std::vector<hlslpp::float3> faceColors;
        hlslpp::float3 origin;
        hlslpp::float4x4 model;
    };

    struct Scene {
        std::vector<SceneObject> objects;
    };

    struct ReadbackDescriptorSet : RenderDescriptorSetBase {
        uint32_t gInput;
        uint32_t gOutput;

        ReadbackDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gInput = builder.addTexture(1);
            gOutput = builder.addReadWriteStructuredBuffer(2);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    struct Variant {
        std::string name;
        uint32_t width;
        uint32_t height;
        RenderSampleCounts sampleCount;
        bool widescreen;
        bool roof;
    };

    hlslpp::float4x4 translationMatrix(hlslpp::float3 t) {
        return hlslpp::float4x4(
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            float(t.x), float(t.y), float(t.z), 1.0f);
    }

    // Row vector convention (v * M), the same one used by libultra and RT64.
    hlslpp::float4x4 lookAtRotation(hlslpp::float3 eye, hlslpp::float3 target) {
        const hlslpp::float3 forward = hlslpp::normalize(target - eye);
        const hlslpp::float3 right = hlslpp::normalize(hlslpp::cross(forward, hlslpp::float3(0.0f, 1.0f, 0.0f)));
        const hlslpp::float3 up = hlslpp::cross(right, forward);
        return hlslpp::float4x4(
            float(right.x), float(up.x), -float(forward.x), 0.0f,
            float(right.y), float(up.y), -float(forward.y), 0.0f,
            float(right.z), float(up.z), -float(forward.z), 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);
    }

    // Equivalent to guPerspective in row vector convention.
    interop::float4x4 perspective(float fovYDegrees, float aspect, float nearPlane, float farPlane) {
        const float f = 1.0f / std::tan(fovYDegrees * Pi / 360.0f);
        return interop::float4x4(
            f / aspect, 0.0f, 0.0f, 0.0f,
            0.0f, f, 0.0f, 0.0f,
            0.0f, 0.0f, (nearPlane + farPlane) / (nearPlane - farPlane), -1.0f,
            0.0f, 0.0f, (2.0f * nearPlane * farPlane) / (nearPlane - farPlane), 0.0f);
    }

    void addQuad(SceneObject &object, uint32_t a, uint32_t b, uint32_t c, uint32_t d, hlslpp::float3 color) {
        object.indices.insert(object.indices.end(), { a, b, c, a, c, d });
        object.faceColors.push_back(color);
        object.faceColors.push_back(color);
    }

    SceneObject makeBox(hlslpp::float3 position, hlslpp::float3 size, hlslpp::float3 color) {
        SceneObject object;
        const float hx = float(size.x) * 0.5f;
        const float sy = float(size.y);
        const float hz = float(size.z) * 0.5f;
        for (int i = 0; i < 8; i++) {
            object.positions.push_back(hlslpp::float3((i & 1) ? hx : -hx, (i & 2) ? sy : 0.0f, (i & 4) ? hz : -hz));
        }

        addQuad(object, 0, 1, 3, 2, color); // Front (-Z).
        addQuad(object, 5, 4, 6, 7, color); // Back (+Z).
        addQuad(object, 4, 0, 2, 6, color); // Left (-X).
        addQuad(object, 1, 5, 7, 3, color); // Right (+X).
        addQuad(object, 2, 3, 7, 6, color); // Top.
        addQuad(object, 4, 5, 1, 0, color); // Bottom.
        object.origin = position;
        return object;
    }

    SceneObject makeGround(float extent, int cells, hlslpp::float3 colorA, hlslpp::float3 colorB) {
        SceneObject object;
        for (int z = 0; z <= cells; z++) {
            for (int x = 0; x <= cells; x++) {
                object.positions.push_back(hlslpp::float3(-extent + 2.0f * extent * x / cells, 0.0f, -extent + 2.0f * extent * z / cells));
            }
        }

        for (int z = 0; z < cells; z++) {
            for (int x = 0; x < cells; x++) {
                const uint32_t i0 = z * (cells + 1) + x;
                const uint32_t i1 = i0 + 1;
                const uint32_t i2 = i0 + (cells + 1);
                const uint32_t i3 = i2 + 1;
                addQuad(object, i0, i2, i3, i1, ((x + z) % 2) ? colorA : colorB);
            }
        }

        object.origin = hlslpp::float3(0.0f, 0.0f, 0.0f);
        return object;
    }

    SceneObject makeSphere(hlslpp::float3 position, float radius, hlslpp::float3 color, int rings, int segments) {
        SceneObject object;
        for (int r = 0; r <= rings; r++) {
            const float phi = Pi * r / rings;
            for (int s = 0; s <= segments; s++) {
                const float theta = 2.0f * Pi * s / segments;
                object.positions.push_back(hlslpp::float3(radius * std::sin(phi) * std::cos(theta), radius * std::cos(phi), radius * std::sin(phi) * std::sin(theta)));
            }
        }

        for (int r = 0; r < rings; r++) {
            for (int s = 0; s < segments; s++) {
                const uint32_t i0 = r * (segments + 1) + s;
                const uint32_t i1 = i0 + 1;
                const uint32_t i2 = i0 + (segments + 1);
                const uint32_t i3 = i2 + 1;
                addQuad(object, i0, i1, i3, i2, color);
            }
        }

        object.origin = position;
        return object;
    }

    SceneObject makeCone(hlslpp::float3 position, float radius, float height, hlslpp::float3 color, int segments) {
        SceneObject object;
        object.positions.push_back(hlslpp::float3(0.0f, height, 0.0f));
        object.positions.push_back(hlslpp::float3(0.0f, 0.0f, 0.0f));
        for (int s = 0; s <= segments; s++) {
            const float theta = 2.0f * Pi * s / segments;
            object.positions.push_back(hlslpp::float3(radius * std::cos(theta), 0.0f, radius * std::sin(theta)));
        }

        for (int s = 0; s < segments; s++) {
            object.indices.insert(object.indices.end(), { 0, uint32_t(2 + s + 1), uint32_t(2 + s) });
            object.faceColors.push_back(color);
            object.indices.insert(object.indices.end(), { 1, uint32_t(2 + s), uint32_t(2 + s + 1) });
            object.faceColors.push_back(color);
        }

        object.origin = position;
        return object;
    }

    Scene buildScene(hlslpp::float3 cameraPosition, bool roof) {
        Scene scene;
        const hlslpp::float3 grassA(0.36f, 0.62f, 0.25f);
        const hlslpp::float3 grassB(0.32f, 0.56f, 0.22f);
        const hlslpp::float3 stone(0.68f, 0.64f, 0.58f);
        const hlslpp::float3 wood(0.62f, 0.42f, 0.24f);
        const hlslpp::float3 leaves(0.20f, 0.48f, 0.20f);
        const hlslpp::float3 fur(0.66f, 0.40f, 0.18f);
        const hlslpp::float3 backpack(0.30f, 0.38f, 0.78f);
        scene.objects.push_back(makeGround(1500.0f, 30, grassA, grassB));

        // A character stand-in in the middle of the scene.
        scene.objects.push_back(makeSphere(hlslpp::float3(0.0f, 50.0f, 0.0f), 50.0f, fur, 12, 18));
        scene.objects.push_back(makeSphere(hlslpp::float3(0.0f, 125.0f, 8.0f), 32.0f, fur, 10, 16));
        scene.objects.push_back(makeBox(hlslpp::float3(0.0f, 45.0f, -48.0f), hlslpp::float3(70.0f, 70.0f, 40.0f), backpack));

        // Pillars, crates, an arch and a tree.
        scene.objects.push_back(makeBox(hlslpp::float3(-320.0f, 0.0f, -260.0f), hlslpp::float3(70.0f, 320.0f, 70.0f), stone));
        scene.objects.push_back(makeBox(hlslpp::float3(330.0f, 0.0f, -420.0f), hlslpp::float3(70.0f, 380.0f, 70.0f), stone));
        scene.objects.push_back(makeBox(hlslpp::float3(-190.0f, 0.0f, 80.0f), hlslpp::float3(90.0f, 90.0f, 90.0f), wood));
        scene.objects.push_back(makeBox(hlslpp::float3(-150.0f, 0.0f, 160.0f), hlslpp::float3(60.0f, 60.0f, 60.0f), wood));
        scene.objects.push_back(makeBox(hlslpp::float3(-110.0f, 0.0f, -560.0f), hlslpp::float3(60.0f, 200.0f, 60.0f), stone));
        scene.objects.push_back(makeBox(hlslpp::float3(110.0f, 0.0f, -560.0f), hlslpp::float3(60.0f, 200.0f, 60.0f), stone));
        scene.objects.push_back(makeBox(hlslpp::float3(0.0f, 200.0f, -560.0f), hlslpp::float3(300.0f, 40.0f, 80.0f), stone));
        scene.objects.push_back(makeBox(hlslpp::float3(250.0f, 0.0f, -60.0f), hlslpp::float3(30.0f, 120.0f, 30.0f), wood));
        scene.objects.push_back(makeCone(hlslpp::float3(250.0f, 100.0f, -60.0f), 110.0f, 260.0f, leaves, 16));

        // A large roof over the whole scene simulates an enclosed area where the sun is blocked everywhere.
        if (roof) {
            scene.objects.push_back(makeBox(hlslpp::float3(0.0f, 650.0f, -400.0f), hlslpp::float3(5000.0f, 40.0f, 5000.0f), stone));
        }

        // Like Banjo-Kazooie, the world matrices are relative to the camera and the view matrix only rotates.
        for (SceneObject &object : scene.objects) {
            object.model = translationMatrix(object.origin - cameraPosition);
        }

        return scene;
    }

    hlslpp::float3 transformPoint(const hlslpp::float3 &p, const hlslpp::float4x4 &m) {
        const hlslpp::float4 r = hlslpp::mul(hlslpp::float4(p, 1.0f), m);
        const float w = float(r.w);
        return hlslpp::float3(float(r.x) / w, float(r.y) / w, float(r.z) / w);
    }

    struct CameraSetup {
        hlslpp::float4x4 viewProj;
        interop::RSPViewport viewport;
        hlslpp::float2 screenScale;
        hlslpp::float2 screenOffset;
        float targetWidth;
        float targetHeight;
    };

    // Replicates RSPProcessCS followed by RasterVS for perspective geometry.
    hlslpp::float4 rasterPosition(const CameraSetup &camera, const hlslpp::float3 &worldPosition) {
        const hlslpp::float4 clip = hlslpp::mul(hlslpp::float4(worldPosition, 1.0f), camera.viewProj);
        const float w = float(clip.w);
        const float ndcX = float(clip.x) / w;
        const float ndcY = float(clip.y) / -w;
        const float ndcZ = float(clip.z) / w;
        const float screenX = ndcX * camera.viewport.scale.x + camera.viewport.translate.x;
        const float screenY = ndcY * camera.viewport.scale.y + camera.viewport.translate.y;
        const float screenZ = ndcZ * camera.viewport.scale.z + camera.viewport.translate.z;
        float x = (screenX - NativeWidth / 2.0f) / (NativeWidth / 2.0f);
        float y = (screenY - NativeHeight / 2.0f) / (NativeHeight / -2.0f);
        x = x * float(camera.screenScale.x) + float(camera.screenOffset.x);
        y = y * float(camera.screenScale.y) + float(camera.screenOffset.y);
        return hlslpp::float4(x * w, y * w, screenZ * w, w);
    }

    HybridRaytracingCamera toHybridCamera(const CameraSetup &camera) {
        HybridRaytracingCamera hybridCamera;
        hybridCamera.viewProj = camera.viewProj;
        hybridCamera.viewport = camera.viewport;
        hybridCamera.fbResolution = hlslpp::float2(NativeWidth, NativeHeight);
        hybridCamera.screenScale = camera.screenScale;
        hybridCamera.screenOffset = camera.screenOffset;
        hybridCamera.targetWidth = camera.targetWidth;
        hybridCamera.targetHeight = camera.targetHeight;
        return hybridCamera;
    }

    // Verifies on the CPU that the pixel to world matrix inverts the raster transformations.
    float checkReconstruction(const CameraSetup &camera, const Scene &scene) {
        const hlslpp::float4x4 pixelToWorld = computeHybridPixelToWorld(toHybridCamera(camera));
        float maxRelativeError = 0.0f;
        for (const SceneObject &object : scene.objects) {
            for (const hlslpp::float3 &position : object.positions) {
                const hlslpp::float3 world = transformPoint(position, object.model);
                const hlslpp::float4 raster = rasterPosition(camera, world);
                const float rasterW = float(raster.w);
                if (rasterW <= 0.0f) {
                    continue;
                }

                const float ndcX = float(raster.x) / rasterW;
                const float ndcY = float(raster.y) / rasterW;
                const float depth = float(raster.z) / rasterW;
                if ((std::abs(ndcX) > 1.0f) || (std::abs(ndcY) > 1.0f) || (depth < 0.0f) || (depth > 1.0f)) {
                    continue;
                }

                const float pixelX = (ndcX * 0.5f + 0.5f) * camera.targetWidth;
                const float pixelY = (0.5f - ndcY * 0.5f) * camera.targetHeight;
                const hlslpp::float3 reconstructed = transformPoint(hlslpp::float3(pixelX, pixelY, depth), pixelToWorld);
                const float error = float(hlslpp::length(reconstructed - world)) / std::max(float(hlslpp::length(world)), 1.0f);
                maxRelativeError = std::max(maxRelativeError, error);
            }
        }

        return maxRelativeError;
    }

    bool writePNG(const std::filesystem::path &path, const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height) {
        return stbi_write_png(path.string().c_str(), int(width), int(height), 4, pixels.data(), int(width * 4)) != 0;
    }
};

int main(int argc, char *argv[]) {
    const std::filesystem::path outputDirectory = (argc > 1) ? std::filesystem::path(argv[1]) : std::filesystem::path("hybrid_rt_output");
    std::filesystem::create_directories(outputDirectory);

    HybridRaytracingConfiguration baseConfig;
    if (argc > 2) {
        if (!baseConfig.load(argv[2])) {
            fprintf(stderr, "Unable to load the configuration from %s.\n", argv[2]);
            return 1;
        }
    }

    std::unique_ptr<RenderInterface> renderInterface = plume::CreateVulkanInterface();
    if (renderInterface == nullptr) {
        fprintf(stderr, "Unable to create the Vulkan interface.\n");
        return 1;
    }

    std::unique_ptr<RenderDevice> device = renderInterface->createDevice();
    if (device == nullptr) {
        fprintf(stderr, "Unable to create the Vulkan device.\n");
        return 1;
    }

    fprintf(stdout, "Device: %s\n", device->getDescription().name.c_str());
    if (!device->getCapabilities().raytracing) {
        fprintf(stderr, "The device doesn't support ray tracing pipelines.\n");
        return 1;
    }

    // Camera placed behind and above the character, like the game's third person camera.
    const hlslpp::float3 cameraPosition(60.0f, 300.0f, 620.0f);
    const hlslpp::float3 cameraTarget(-20.0f, 60.0f, -120.0f);
    const float nearPlane = 20.0f;
    const float farPlane = 20000.0f;
    auto createUploadBuffer = [&](const void *data, size_t size, RenderBufferFlags flags) {
        std::unique_ptr<RenderBuffer> buffer = device->createBuffer(RenderBufferDesc::UploadBuffer(size, flags));
        void *bytes = buffer->map();
        memcpy(bytes, data, size);
        buffer->unmap();
        return buffer;
    };

    // Raster pipeline.
    RenderPipelineLayoutBuilder layoutBuilder;
    layoutBuilder.begin(false, true);
    layoutBuilder.end();
    std::unique_ptr<RenderPipelineLayout> rasterLayout = layoutBuilder.create(device.get());
    std::unique_ptr<RenderShader> rasterVS = device->createShader(HarnessRasterVSBlobSPIRV, sizeof(HarnessRasterVSBlobSPIRV), "VSMain", RenderShaderFormat::SPIRV);
    std::unique_ptr<RenderShader> rasterPS = device->createShader(HarnessRasterPSBlobSPIRV, sizeof(HarnessRasterPSBlobSPIRV), "PSMain", RenderShaderFormat::SPIRV);
    const RenderInputSlot inputSlot(0, sizeof(RasterVertex));
    const RenderInputElement inputElements[] = {
        RenderInputElement("POSITION", 0, 0, RenderFormat::R32G32B32A32_FLOAT, 0, 0),
        RenderInputElement("COLOR", 0, 1, RenderFormat::R32G32B32A32_FLOAT, 0, sizeof(float) * 4)
    };

    // Readback pipeline. Plume can't copy textures into buffers, so a compute shader packs the image instead.
    ReadbackDescriptorSet readbackSet(device.get());
    layoutBuilder.begin();
    layoutBuilder.addPushConstant(0, 0, sizeof(uint32_t) * 2, RenderShaderStageFlag::COMPUTE);
    layoutBuilder.addDescriptorSet(readbackSet);
    layoutBuilder.end();
    std::unique_ptr<RenderPipelineLayout> readbackLayout = layoutBuilder.create(device.get());
    std::unique_ptr<RenderShader> readbackShader = device->createShader(HarnessReadbackCSBlobSPIRV, sizeof(HarnessReadbackCSBlobSPIRV), "CSMain", RenderShaderFormat::SPIRV);
    std::unique_ptr<RenderPipeline> readbackPipeline = device->createComputePipeline(RenderComputePipelineDesc(readbackLayout.get(), readbackShader.get(), 8, 8, 1));

    std::unique_ptr<RenderCommandQueue> commandQueue = device->createCommandQueue(RenderCommandListType::DIRECT);
    std::unique_ptr<RenderCommandList> commandList = commandQueue->createCommandList();
    std::unique_ptr<RenderCommandFence> commandFence = device->createCommandFence();
    HybridRaytracer hybridRaytracer(device.get(), RenderShaderFormat::SPIRV);
    if (!hybridRaytracer.isSupported()) {
        fprintf(stderr, "The hybrid ray tracer is not supported on this device.\n");
        return 1;
    }

    const Variant variants[] = {
        { "4x3", 1280, 960, RenderSampleCount::COUNT_1, false, false },
        { "4x3_msaa4x", 1280, 960, RenderSampleCount::COUNT_4, false, false },
        { "16x9", 1664, 936, RenderSampleCount::COUNT_1, true, false },
        { "4x3_interior", 1280, 960, RenderSampleCount::COUNT_1, false, true },
    };

    const HybridRaytracingConfiguration::DebugView debugViews[] = {
        HybridRaytracingConfiguration::DebugView::None,
        HybridRaytracingConfiguration::DebugView::Shadows,
        HybridRaytracingConfiguration::DebugView::AmbientOcclusion,
        HybridRaytracingConfiguration::DebugView::Normals
    };

    const char *debugViewNames[] = { "hybrid", "debug_shadows", "debug_ao", "debug_normals" };
    int failures = 0;
    for (const Variant &variant : variants) {
        hybridRaytracer.resetTemporalState();
        Scene scene = buildScene(cameraPosition, variant.roof);

        // Flatten the scene into the buffers RT64 would have: model space positions, per vertex world transform indices,
        // world transforms and the global triangle indices.
        std::vector<float> positionFloats;
        std::vector<float> velocityFloats;
        std::vector<uint16_t> worldIndices;
        std::vector<interop::float4x4> worldTransforms;
        std::vector<uint32_t> globalIndices;
        for (uint32_t o = 0; o < uint32_t(scene.objects.size()); o++) {
            const SceneObject &object = scene.objects[o];
            const uint32_t vertexBase = uint32_t(worldIndices.size());
            for (const hlslpp::float3 &position : object.positions) {
                positionFloats.insert(positionFloats.end(), { float(position.x), float(position.y), float(position.z) });
                velocityFloats.insert(velocityFloats.end(), { 0.0f, 0.0f, 0.0f });
                worldIndices.push_back(uint16_t(o));
            }

            for (uint32_t index : object.indices) {
                globalIndices.push_back(vertexBase + index);
            }

            worldTransforms.push_back(object.model);
        }

        std::unique_ptr<RenderBuffer> positionBuffer = createUploadBuffer(positionFloats.data(), positionFloats.size() * sizeof(float), RenderBufferFlag::FORMATTED);
        std::unique_ptr<RenderBuffer> velocityBuffer = createUploadBuffer(velocityFloats.data(), velocityFloats.size() * sizeof(float), RenderBufferFlag::FORMATTED);
        std::unique_ptr<RenderBuffer> worldIndicesBuffer = createUploadBuffer(worldIndices.data(), worldIndices.size() * sizeof(uint16_t), RenderBufferFlag::FORMATTED);
        std::unique_ptr<RenderBuffer> worldTransformsBuffer = createUploadBuffer(worldTransforms.data(), worldTransforms.size() * sizeof(interop::float4x4), RenderBufferFlag::STORAGE);
        std::unique_ptr<RenderBufferFormattedView> positionView = positionBuffer->createBufferFormattedView(RenderFormat::R32_FLOAT);
        std::unique_ptr<RenderBufferFormattedView> velocityView = velocityBuffer->createBufferFormattedView(RenderFormat::R32_FLOAT);
        std::unique_ptr<RenderBufferFormattedView> worldIndicesView = worldIndicesBuffer->createBufferFormattedView(RenderFormat::R16_UINT);

        // The game's projection is 4:3. In widescreen, RT64 widens the projection matrix and the render target.
        const float nativeAspect = NativeWidth / NativeHeight;
        const float targetAspect = float(variant.width) / float(variant.height);
        interop::float4x4 projection = perspective(45.0f, nativeAspect, nearPlane, farPlane);
        if (variant.widescreen) {
            const float aspectRatioScale = targetAspect / nativeAspect;
            for (int r = 0; r < 4; r++) {
                projection[r][0] /= aspectRatioScale;
            }
        }

        CameraSetup camera;
        camera.viewProj = hlslpp::mul(lookAtRotation(cameraPosition, cameraTarget), hlslpp::float4x4(projection));
        camera.viewport.scale = interop::float3(NativeWidth / 2.0f, NativeHeight / 2.0f, ViewportDepth);
        camera.viewport.translate = interop::float3(NativeWidth / 2.0f, NativeHeight / 2.0f, ViewportDepth);
        camera.screenScale = hlslpp::float2(1.0f, 1.0f);
        camera.screenOffset = hlslpp::float2(1.0f / float(variant.width), -1.0f / float(variant.height));
        camera.targetWidth = float(variant.width);
        camera.targetHeight = float(variant.height);

        const float reconstructionError = checkReconstruction(camera, scene);
        const bool reconstructionValid = reconstructionError < 1e-3f;
        fprintf(stdout, "[%s] CPU reconstruction max relative error: %g (%s)\n", variant.name.c_str(), reconstructionError, reconstructionValid ? "OK" : "FAILED");
        failures += reconstructionValid ? 0 : 1;

        // Raster geometry with flat baked vertex lighting, as most of the game's level geometry is pre-lit.
        std::vector<RasterVertex> rasterVertices;
        const hlslpp::float3 bakedLight = hlslpp::normalize(hlslpp::float3(-0.4f, 1.0f, 0.6f));
        for (const SceneObject &object : scene.objects) {
            for (size_t t = 0; t < object.indices.size() / 3; t++) {
                const hlslpp::float3 a = transformPoint(object.positions[object.indices[t * 3 + 0]], object.model);
                const hlslpp::float3 b = transformPoint(object.positions[object.indices[t * 3 + 1]], object.model);
                const hlslpp::float3 c = transformPoint(object.positions[object.indices[t * 3 + 2]], object.model);
                const hlslpp::float3 normal = hlslpp::normalize(hlslpp::cross(b - a, c - a));
                const float lighting = 0.6f + 0.4f * std::max(float(hlslpp::dot(normal, bakedLight)), 0.0f);
                const hlslpp::float3 color = object.faceColors[t] * lighting;
                for (const hlslpp::float3 &p : { a, b, c }) {
                    const hlslpp::float4 r = rasterPosition(camera, p);
                    rasterVertices.push_back({ { float(r.x), float(r.y), float(r.z), float(r.w) }, { float(color.x), float(color.y), float(color.z), 1.0f } });
                }
            }
        }

        std::unique_ptr<RenderBuffer> rasterVertexBuffer = createUploadBuffer(rasterVertices.data(), rasterVertices.size() * sizeof(RasterVertex), RenderBufferFlag::VERTEX);
        const RenderVertexBufferView rasterVertexView(rasterVertexBuffer->at(0), uint32_t(rasterVertices.size() * sizeof(RasterVertex)));

        // Targets.
        RenderMultisampling multisampling(variant.sampleCount);
        const RenderClearValue colorClear = RenderClearValue::Color(RenderColor(0.55f, 0.75f, 0.95f, 1.0f), ColorFormat);
        const RenderClearValue depthClear = RenderClearValue::Depth(RenderDepth(), DepthFormat);
        std::unique_ptr<RenderTexture> colorTexture = device->createTexture(RenderTextureDesc::ColorTarget(variant.width, variant.height, ColorFormat, multisampling, &colorClear));
        std::unique_ptr<RenderTexture> depthTexture = device->createTexture(RenderTextureDesc::DepthTarget(variant.width, variant.height, DepthFormat, multisampling, &depthClear));
        std::unique_ptr<RenderTextureView> depthTextureView = depthTexture->createTextureView(RenderTextureViewDesc::Texture2D(DepthFormat));
        std::unique_ptr<RenderTexture> resolvedTexture = device->createTexture(RenderTextureDesc::ColorTarget(variant.width, variant.height, ColorFormat));
        const RenderTexture *colorAttachment = colorTexture.get();
        RenderFramebufferDesc framebufferDesc(&colorAttachment, 1, depthTexture.get());
        std::unique_ptr<RenderFramebuffer> colorDepthWrite = device->createFramebuffer(framebufferDesc);
        framebufferDesc.depthAttachmentReadOnly = true;
        std::unique_ptr<RenderFramebuffer> colorWriteDepthRead = device->createFramebuffer(framebufferDesc);
        std::unique_ptr<RenderBuffer> readbackBuffer = device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(variant.width) * variant.height * 4, RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS));

        RenderGraphicsPipelineDesc pipelineDesc;
        pipelineDesc.pipelineLayout = rasterLayout.get();
        pipelineDesc.vertexShader = rasterVS.get();
        pipelineDesc.pixelShader = rasterPS.get();
        pipelineDesc.renderTargetFormat[0] = ColorFormat;
        pipelineDesc.renderTargetBlend[0] = RenderBlendDesc::Copy();
        pipelineDesc.renderTargetCount = 1;
        pipelineDesc.depthTargetFormat = DepthFormat;
        pipelineDesc.depthEnabled = true;
        pipelineDesc.depthWriteEnabled = true;
        pipelineDesc.depthFunction = RenderComparisonFunction::LESS;
        pipelineDesc.depthClipEnabled = true;
        pipelineDesc.multisampling = multisampling;
        pipelineDesc.inputSlots = &inputSlot;
        pipelineDesc.inputSlotsCount = 1;
        pipelineDesc.inputElements = inputElements;
        pipelineDesc.inputElementsCount = uint32_t(std::size(inputElements));
        std::unique_ptr<RenderPipeline> rasterPipeline = device->createGraphicsPipeline(pipelineDesc);

        HybridRaytracingGeometry geometry;
        geometry.positionBuffer = positionBuffer.get();
        geometry.positionView = positionView.get();
        geometry.velocityBuffer = velocityBuffer.get();
        geometry.velocityView = velocityView.get();
        geometry.worldIndicesBuffer = worldIndicesBuffer.get();
        geometry.worldIndicesView = worldIndicesView.get();
        geometry.worldTransformsBuffer = worldTransformsBuffer.get();
        geometry.vertexCount = uint32_t(worldIndices.size());
        geometry.curFrameWeight = 1.0f;
        geometry.indices = globalIndices.data();
        geometry.indexCount = uint32_t(globalIndices.size());

        for (int view = -1; view < int(std::size(debugViews)); view++) {
            const bool applyHybrid = (view >= 0);
            if (applyHybrid) {
                HybridRaytracingConfiguration config = baseConfig;
                config.debugView = debugViews[view];
                hybridRaytracer.setConfiguration(config);
                if (!hybridRaytracer.prepare(geometry, toHybridCamera(camera), 0)) {
                    fprintf(stderr, "[%s] HybridRaytracer::prepare failed.\n", variant.name.c_str());
                    failures++;
                    continue;
                }
            }

            commandList->begin();
            if (applyHybrid) {
                hybridRaytracer.recordSetup(commandList.get());
            }

            const RenderTextureBarrier startBarriers[] = {
                RenderTextureBarrier(colorTexture.get(), RenderTextureLayout::COLOR_WRITE),
                RenderTextureBarrier(depthTexture.get(), RenderTextureLayout::DEPTH_WRITE)
            };

            commandList->barriers(RenderBarrierStage::GRAPHICS, startBarriers, uint32_t(std::size(startBarriers)));
            commandList->setFramebuffer(colorDepthWrite.get());
            commandList->clearColor(0, RenderColor(0.55f, 0.75f, 0.95f, 1.0f));
            commandList->clearDepth(true, 1.0f);
            commandList->setViewports(RenderViewport(0.0f, 0.0f, float(variant.width), float(variant.height)));
            commandList->setScissors(RenderRect(0, 0, int32_t(variant.width), int32_t(variant.height)));
            commandList->setPipeline(rasterPipeline.get());
            commandList->setGraphicsPipelineLayout(rasterLayout.get());
            commandList->setVertexBuffers(0, &rasterVertexView, 1, &inputSlot);
            commandList->drawInstanced(uint32_t(rasterVertices.size()), 1, 0, 0);

            if (applyHybrid) {
                HybridRaytracingTarget target;
                target.colorTexture = colorTexture.get();
                target.depthTexture = depthTexture.get();
                target.depthTextureView = depthTextureView.get();
                target.framebuffer = colorWriteDepthRead.get();
                target.colorFormat = ColorFormat;
                target.depthFormat = DepthFormat;
                target.multisampling = multisampling;
                target.width = variant.width;
                target.height = variant.height;
                hybridRaytracer.recordPass(commandList.get(), target);
            }

            // Resolve if necessary and copy the image to the readback buffer.
            commandList->setFramebuffer(nullptr);
            RenderTexture *copySource = colorTexture.get();
            if (variant.sampleCount > RenderSampleCount::COUNT_1) {
                const RenderTextureBarrier resolveBarriers[] = {
                    RenderTextureBarrier(colorTexture.get(), RenderTextureLayout::RESOLVE_SOURCE),
                    RenderTextureBarrier(resolvedTexture.get(), RenderTextureLayout::RESOLVE_DEST)
                };

                commandList->barriers(RenderBarrierStage::COPY, resolveBarriers, uint32_t(std::size(resolveBarriers)));
                commandList->resolveTexture(resolvedTexture.get(), colorTexture.get());
                copySource = resolvedTexture.get();
            }

            readbackSet.setTexture(readbackSet.gInput, copySource, RenderTextureLayout::SHADER_READ);
            readbackSet.setBuffer(readbackSet.gOutput, readbackBuffer.get(), RenderBufferStructuredView(sizeof(uint32_t)));
            const uint32_t readbackSize[2] = { variant.width, variant.height };
            commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(copySource, RenderTextureLayout::SHADER_READ));
            commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(readbackBuffer.get(), RenderBufferAccess::WRITE));
            commandList->setPipeline(readbackPipeline.get());
            commandList->setComputePipelineLayout(readbackLayout.get());
            commandList->setComputePushConstants(0, readbackSize);
            commandList->setComputeDescriptorSet(readbackSet.get(), 0);
            commandList->dispatch((variant.width + 7) / 8, (variant.height + 7) / 8, 1);
            commandList->barriers(RenderBarrierStage::COPY, RenderBufferBarrier(readbackBuffer.get(), RenderBufferAccess::READ));
            commandList->end();

            const RenderCommandList *commandLists[] = { commandList.get() };
            commandQueue->executeCommandLists(commandLists, 1, nullptr, 0, nullptr, 0, commandFence.get());
            commandQueue->waitForCommandFence(commandFence.get());

            std::vector<uint8_t> pixels(size_t(variant.width) * variant.height * 4);
            const void *readbackBytes = readbackBuffer->map();
            memcpy(pixels.data(), readbackBytes, pixels.size());
            readbackBuffer->unmap();

            const std::string fileName = variant.name + "_" + (applyHybrid ? debugViewNames[view] : "raster") + ".png";
            const std::filesystem::path filePath = outputDirectory / fileName;
            if (writePNG(filePath, pixels, variant.width, variant.height)) {
                fprintf(stdout, "[%s] Wrote %s\n", variant.name.c_str(), filePath.string().c_str());
            }
            else {
                fprintf(stderr, "[%s] Failed to write %s\n", variant.name.c_str(), filePath.string().c_str());
                failures++;
            }
        }
    }

    fprintf(stdout, "%s\n", (failures == 0) ? "All checks passed." : "Some checks failed.");
    return (failures == 0) ? 0 : 1;
}
