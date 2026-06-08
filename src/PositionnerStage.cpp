#include "PositionnerStage.hpp"
#include "OrientationOptimizer.hpp"
#include "PackingPass.hpp"

#include <MeshPreprocess.hpp>
#include <MeshTypes.hpp>
#include <DataBuffer.hpp>
#include <Stage.hpp>

#include <Eigen/Core>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// ─── StageContract ───────────────────────────────────────────────────────────

StageContract PositionnerStage::contract() const {
    return {
        .cpuDataIn  = CPU_FORMAT_MESH_SCENE_GRAPH_V1,
        .cpuDataOut = CPU_FORMAT_MESH_SCENE_GRAPH_V1,
        .reads  = {},
        .writes = {},
    };
}

// ─── describeJson ─────────────────────────────────────────────────────────────

std::string PositionnerStage::describeJson() const {
    return
        R"j({"id":"Positionner","name":"Positionner",)j"
        R"j("cpuDataIn":"mesh/scene_graph_v1",)j"
        R"j("cpuDataOut":"mesh/scene_graph_v1","params":[)j"
        R"j({"key":"orient.enable","label":"Enable orientation","type":"bool","default":true},)j"
        R"j({"key":"orient.num_candidates","label":"Candidate directions","type":"int","min":6,"max":256,"default":32,"step":1},)j"
        R"j({"key":"orient.weight_peel","label":"Peel force weight","type":"float","min":0,"max":1,"default":0.6,"step":0.05},)j"
        R"j({"key":"orient.weight_suction","label":"Suction cup weight","type":"float","min":0,"max":1,"default":0.25,"step":0.05},)j"
        R"j({"key":"orient.weight_islands","label":"Floating island weight","type":"float","min":0,"max":1,"default":0.15,"step":0.05},)j"
        R"j({"key":"pack.enable","label":"Enable packing","type":"bool","default":true},)j"
        R"j({"key":"pack.resolution","label":"Heightmap resolution (mm/px)","type":"float","min":0.01,"max":5,"default":0.1,"step":0.01},)j"
        R"j({"key":"pack.clearance","label":"Part clearance (mm)","type":"float","min":0,"max":20,"default":0.5,"step":0.1},)j"
        R"j({"key":"pack.bed_offset","label":"Bed offset / support lift (mm)","type":"float","min":0,"max":50,"default":0,"step":0.5})j"
        R"j(]})j";
}

// ─── describe ────────────────────────────────────────────────────────────────

ParameterSchema PositionnerStage::describe() const {
    return {
        "positionner",
        {
            { "orient.enable",         "Enable orientation",             "Auto-rotate meshes for optimal DLP print orientation",         ParamType::Bool,  true, ParameterDescriptor::BoolMeta{true}  },
            { "orient.num_candidates", "Candidate directions",           "Number of orientations evaluated (6 principal axes + Fibonacci sphere fill)", ParamType::Int, true, ParameterDescriptor::IntMeta{6, 256, 32} },
            { "orient.weight_peel",    "Peel force weight",              "Weight for peel force (projected area) in cost function",      ParamType::Float, true, ParameterDescriptor::FloatMeta{0.f, 1.f, 0.6f,  0.05f} },
            { "orient.weight_suction", "Suction cup weight",             "Weight for suction cup penalty in cost function",              ParamType::Float, true, ParameterDescriptor::FloatMeta{0.f, 1.f, 0.25f, 0.05f} },
            { "orient.weight_islands", "Floating island weight",         "Weight for floating island penalty in cost function",          ParamType::Float, true, ParameterDescriptor::FloatMeta{0.f, 1.f, 0.15f, 0.05f} },
            { "pack.enable",           "Enable packing",                 "Arrange instances on the build plate automatically",           ParamType::Bool,  true, ParameterDescriptor::BoolMeta{true}  },
            { "pack.resolution",       "Heightmap resolution (mm/px)",   "Pixel size used for the packing heightmap",                    ParamType::Float, true, ParameterDescriptor::FloatMeta{0.01f, 5.f, 0.1f, 0.01f} },
            { "pack.clearance",        "Part clearance (mm)",            "Minimum gap between adjacent parts",                          ParamType::Float, true, ParameterDescriptor::FloatMeta{0.f, 20.f, 0.5f, 0.1f} },
            { "pack.bed_offset",       "Bed offset / support lift (mm)", "Minimum Z height for all parts; set 3-5 mm for supports",      ParamType::Float, true, ParameterDescriptor::FloatMeta{0.f, 50.f, 0.f, 0.5f} },
        }
    };
}

// ─── configure ───────────────────────────────────────────────────────────────

void PositionnerStage::configure(const ParameterValues& v) {
    if (v.count("orient.enable"))
        orientEnable_ = std::get<bool>(v.at("orient.enable"));
    if (v.count("orient.num_candidates"))
        numCandidates_ = std::get<int>(v.at("orient.num_candidates"));
    if (v.count("orient.weight_peel"))
        weightPeel_ = std::get<float>(v.at("orient.weight_peel"));
    if (v.count("orient.weight_suction"))
        weightSuction_ = std::get<float>(v.at("orient.weight_suction"));
    if (v.count("orient.weight_islands"))
        weightIslands_ = std::get<float>(v.at("orient.weight_islands"));
    if (v.count("pack.enable"))
        packEnable_ = std::get<bool>(v.at("pack.enable"));
    if (v.count("pack.resolution"))
        packResolution_ = std::get<float>(v.at("pack.resolution"));
    if (v.count("pack.clearance"))
        packClearance_ = std::get<float>(v.at("pack.clearance"));
    if (v.count("pack.bed_offset"))
        bedOffset_ = std::get<float>(v.at("pack.bed_offset"));
}

// ─── process ─────────────────────────────────────────────────────────────────

Result PositionnerStage::process(DataBuffer& data) {
    // Must run before parsing — it may resize data.cpuData (degenerate tri removal).
    precomputeWeldedMeshes(data);

    const auto* raw = data.cpuData.data();
    const size_t sz = data.cpuData.size();

    if (sz < sizeof(SceneGraphHeader))
        return Result::fail("positionner: cpuData too small for SceneGraphHeader");

    SceneGraphHeader hdr;
    std::memcpy(&hdr, raw, sizeof(hdr));
    if (hdr.magic != SCENE_GRAPH_MAGIC)
        return Result::fail("positionner: invalid scene-graph magic");

    // ── Parse unique meshes ──────────────────────────────────────────────────
    struct MeshEntry {
        Vec3f    aabbMin, aabbMax;
        uint32_t triangleCount;
        const Triangle* tris; // pointer into cpuData
    };

    const uint8_t* p = reinterpret_cast<const uint8_t*>(raw) + sizeof(SceneGraphHeader);
    const uint8_t* end = reinterpret_cast<const uint8_t*>(raw) + sz;

    std::vector<MeshEntry> meshes(hdr.meshCount);
    for (uint32_t m = 0; m < hdr.meshCount; ++m) {
        if (p + sizeof(Vec3f) * 2 + sizeof(uint32_t) > end)
            return Result::fail("positionner: buffer overrun reading mesh header");
        std::memcpy(&meshes[m].aabbMin, p, sizeof(Vec3f)); p += sizeof(Vec3f);
        std::memcpy(&meshes[m].aabbMax, p, sizeof(Vec3f)); p += sizeof(Vec3f);
        std::memcpy(&meshes[m].triangleCount, p, sizeof(uint32_t)); p += sizeof(uint32_t);
        size_t triBytes = meshes[m].triangleCount * sizeof(Triangle);
        if (p + triBytes > end)
            return Result::fail("positionner: buffer overrun reading triangles");
        meshes[m].tris = reinterpret_cast<const Triangle*>(p);
        p += triBytes;
    }

    // instances start here
    if (p + hdr.instanceCount * sizeof(MeshInstance) > end)
        return Result::fail("positionner: buffer overrun reading instances");
    const MeshInstance* instancesIn = reinterpret_cast<const MeshInstance*>(p);

    OrientWeights weights{ weightPeel_, weightSuction_, weightIslands_ };

    if (data.weldedMeshes.size() < hdr.meshCount)
        return Result::fail("positionner: weldedMeshes could not be built (malformed buffer?)");

    struct MeshResult {
        Eigen::Matrix3f R; // best rotation (identity if orient disabled)
    };
    std::vector<MeshResult> results(hdr.meshCount);

    using Clock = std::chrono::steady_clock;
    auto msec = [](Clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };

    for (uint32_t m = 0; m < hdr.meshCount; ++m) {
        std::cerr << "[positionner] mesh " << m
                  << " (" << meshes[m].triangleCount << " tris)\n";

        const WeldedMesh& wm = getWeldedMesh(data, m);

        if (orientEnable_ && meshes[m].triangleCount > 0) {
            auto t0 = Clock::now();
            auto ci = OrientationOptimizer::candidateAxes(wm.V, wm.F, numCandidates_);
            std::cerr << "  candidateAxes:  " << std::fixed << std::setprecision(1) << msec(t0) << " ms"
                      << "  (" << ci.rotations.size() << " directions)\n";

            if (suctionPass_) {
                std::cerr << "  suction: GPU\n";
                const int N = static_cast<int>(ci.rotations.size());
                std::vector<int> suctionScores(N);
                for (int k = 0; k < N; ++k) {
                    t0 = Clock::now();
                    Eigen::MatrixXf Vr =
                        (ci.rotations[k] * wm.V.transpose()).transpose();
                    suctionScores[k] =
                        suctionPass_->render(gpuCtx_, Vr, wm.F);
                    std::cerr << "  suction[" << k << "]:     " << msec(t0) << " ms"
                              << "  (score=" << suctionScores[k] << ")\n";
                }
                t0 = Clock::now();
                results[m].R = OrientationOptimizer::findBestRotation(
                    wm.V, wm.F, *wm.halfEdge, weights, suctionScores, ci);
                std::cerr << "  findBestRot:    " << msec(t0) << " ms\n";
            } else {
                std::cerr << "  suction: CPU\n";
                t0 = Clock::now();
                results[m].R = OrientationOptimizer::findBestRotation(
                    wm.V, wm.F, *wm.halfEdge, weights);
                std::cerr << "  findBestRot:    " << msec(t0) << " ms\n";
            }
        } else {
            results[m].R = Eigen::Matrix3f::Identity();
        }

        const Eigen::Matrix3f& R = results[m].R;
        std::cerr << "  bestRotation[" << m << "]:\n"
                  << "    " << R(0,0) << " " << R(0,1) << " " << R(0,2) << "\n"
                  << "    " << R(1,0) << " " << R(1,1) << " " << R(1,2) << "\n"
                  << "    " << R(2,0) << " " << R(2,1) << " " << R(2,2) << "\n";
    }

    // ── Rebuild cpuData with updated transforms ──────────────────────────────
    // We rewrite the buffer in-place: header + mesh blocks are unchanged;
    // only the MeshInstance transforms and AABBs are updated.

    // Build updated instances
    std::vector<MeshInstance> instancesOut(hdr.instanceCount);
    for (uint32_t i = 0; i < hdr.instanceCount; ++i) {
        instancesOut[i] = instancesIn[i];
        uint32_t m = instancesIn[i].meshIndex;
        if (m >= hdr.meshCount) continue;

        const Eigen::Matrix3f& R = results[m].R;

        // Compose R into the existing column-major 4×4 transform.
        // The existing transform stores rotation in columns 0-2, rows 0-2.
        // We left-multiply the rotation block by R.
        float* T = instancesOut[i].transform;
        // Extract current 3×3 rotation block (column-major)
        Eigen::Matrix3f curR;
        for (int col = 0; col < 3; ++col)
            for (int row = 0; row < 3; ++row)
                curR(row, col) = T[col * 4 + row];

        Eigen::Matrix3f newR = R * curR;

        for (int col = 0; col < 3; ++col)
            for (int row = 0; row < 3; ++row)
                T[col * 4 + row] = newR(row, col);

        // Recompute world-space AABB from rotated unique-mesh vertices
        const Eigen::MatrixXf& Vr = (R * getWeldedMesh(data, m).V.transpose()).transpose();
        Eigen::Vector3f wMin = Vr.colwise().minCoeff().transpose();
        Eigen::Vector3f wMax = Vr.colwise().maxCoeff().transpose();
        // Add translation (column 3, rows 0-2)
        for (int k = 0; k < 3; ++k) {
            wMin(k) += T[3 * 4 + k];
            wMax(k) += T[3 * 4 + k];
        }
        instancesOut[i].aabbMin = { wMin(0), wMin(1), wMin(2) };
        instancesOut[i].aabbMax = { wMax(0), wMax(1), wMax(2) };
    }

    // ── Packing ───────────────────────────────────────────────────────────────
    if (packEnable_) {
        if (data.printer.buildXMm == 0.f) {
            std::cerr << "[positionner] packing skipped: printer profile not set\n";
        } else {
            PackingPass::Params pp;
            pp.plateX       = data.printer.buildXMm;
            pp.plateY       = data.printer.buildYMm;
            pp.plateZ       = data.printer.buildZMm;
            pp.resolution   = packResolution_;
            pp.clearance    = packClearance_;
            pp.bedOffset    = bedOffset_;

            // Rotated object-space vertices per unique mesh.
            // Translation is owned entirely by the packer and will be set
            // absolutely below — do not pre-apply existing T here.
            std::vector<Eigen::MatrixXf> rotV(hdr.meshCount);
            std::vector<Eigen::MatrixXi> faces(hdr.meshCount);
            for (uint32_t m = 0; m < hdr.meshCount; ++m) {
                const WeldedMesh& wm = getWeldedMesh(data, m);
                rotV[m]  = (results[m].R * wm.V.transpose()).transpose();
                faces[m] = wm.F;
            }

            // Per-instance mesh index and object-space rotated AABBs.
            // Translation is excluded: the packer assigns it absolutely.
            std::vector<uint32_t>        instMesh(hdr.instanceCount);
            std::vector<Eigen::Vector3f> aabbMin(hdr.instanceCount);
            std::vector<Eigen::Vector3f> aabbMax(hdr.instanceCount);
            for (uint32_t i = 0; i < hdr.instanceCount; ++i) {
                uint32_t m = instancesOut[i].meshIndex;
                instMesh[i] = m;
                aabbMin[i]  = rotV[m].colwise().minCoeff().transpose();
                aabbMax[i]  = rotV[m].colwise().maxCoeff().transpose();
            }

            auto placed = packingPass_.pack(rotV, faces, instMesh, aabbMin, aabbMax, pp);

            for (const auto& pl : placed) {
                float* T = instancesOut[pl.instanceIdx].transform;
                T[12] = pl.tx;
                T[13] = pl.ty;
                T[14] = pl.tz;

                const Eigen::Vector3f& mn0 = aabbMin[pl.instanceIdx];
                const Eigen::Vector3f& mx0 = aabbMax[pl.instanceIdx];
                auto& mn = instancesOut[pl.instanceIdx].aabbMin;
                auto& mx = instancesOut[pl.instanceIdx].aabbMax;
                mn = { mn0.x() + pl.tx, mn0.y() + pl.ty, mn0.z() + pl.tz };
                mx = { mx0.x() + pl.tx, mx0.y() + pl.ty, mx0.z() + pl.tz };
            }
        }
    }

    // Write updated instances back into cpuData
    // (the mesh blocks preceding the instances are untouched)
    uint8_t* instDst = reinterpret_cast<uint8_t*>(data.cpuData.data())
                       + (p - reinterpret_cast<const uint8_t*>(raw));
    std::memcpy(instDst, instancesOut.data(),
                hdr.instanceCount * sizeof(MeshInstance));

    return Result::ok();
}

// ─── initGpu ─────────────────────────────────────────────────────────────────

void PositionnerStage::initGpu(const GpuContext& ctx) {
    std::cerr << "[positionner] initGpu called, available=" << ctx.isAvailable() << "\n";
    if (!ctx.isAvailable()) return;
    gpuCtx_ = ctx;
    try {
        suctionPass_ = std::make_unique<SuctionPass>();
        suctionPass_->init(ctx);
    } catch (const std::exception& e) {
        std::cerr << "[positionner] GPU suction pass init failed: " << e.what()
                  << " — falling back to CPU.\n";
        suctionPass_.reset();
    }
    try {
        packingPass_.init(ctx);
        std::cerr << "[positionner] packing GPU init OK\n";
    } catch (const std::exception& e) {
        std::cerr << "[positionner] packing GPU init FAILED: " << e.what() << "\n";
    }
}

// ─── Plugin ABI ──────────────────────────────────────────────────────────────

#ifndef PIPELINE_NO_PLUGIN_EXPORTS
extern "C" {
    int    pluginApiVersion()          { return PIPELINE_API_VERSION; }
    Stage* createStage()               { return new PositionnerStage(); }
    void   destroyStage(Stage* s)      { delete s; }
}
#endif
