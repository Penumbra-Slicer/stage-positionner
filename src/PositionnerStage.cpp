#include "PositionnerStage.hpp"
#include "OrientationOptimizer.hpp"
#include "PackingPass.hpp"

#include <MeshPreprocess.hpp>
#include <MeshTypes.hpp>
#include <MeshTransform.hpp>
#include <DataBuffer.hpp>
#include <Stage.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace {
constexpr float kDeg2Rad = 3.14159265358979323846f / 180.f;
constexpr float kRad2Deg = 180.f / 3.14159265358979323846f;

// Matches Core/include/MeshTransform.hpp's decodeTransformMatrix rotation convention:
// R = Rz(rz) * Ry(ry) * Rx(rx), degrees.
Eigen::Matrix3f eulerDegToMatrix(const Vec3f& deg) {
    return (Eigen::AngleAxisf(deg.z * kDeg2Rad, Eigen::Vector3f::UnitZ()) *
            Eigen::AngleAxisf(deg.y * kDeg2Rad, Eigen::Vector3f::UnitY()) *
            Eigen::AngleAxisf(deg.x * kDeg2Rad, Eigen::Vector3f::UnitX())).toRotationMatrix();
}

// Inverse of eulerDegToMatrix — recovers (rx,ry,rz) in degrees from R = Rz*Ry*Rx.
Vec3f matrixToEulerDeg(const Eigen::Matrix3f& R) {
    const Eigen::Vector3f zyx = R.eulerAngles(2, 1, 0); // (about Z, about Y, about X), radians
    return { zyx.z() * kRad2Deg, zyx.y() * kRad2Deg, zyx.x() * kRad2Deg };
}
} // namespace

// ─── StageContract ───────────────────────────────────────────────────────────

StageContract PositionnerStage::contract() const {
    return {
        .cpuDataIn  = std::vector<std::string>{CPU_FORMAT_MESH_SCENE_GRAPH_V1},
        .cpuDataOut = CPU_FORMAT_MESH_SCENE_GRAPH_V1,
        .reads  = {},
        .writes = {},
    };
}

// ─── describeJson ─────────────────────────────────────────────────────────────

std::string PositionnerStage::describeJson() const {
    return
        R"j({"id":"Positionner","name":"Positionner",)j"
        R"j("cpuDataIn":["mesh/scene_graph_v1"],)j"
        R"j("cpuDataOut":"mesh/scene_graph_v1","params":[)j"
        R"j({"key":"orient.enable","label":"Enable orientation","type":"bool","default":true},)j"
        R"j({"key":"orient.num_candidates","label":"Candidate directions","type":"int","min":6,"max":256,"default":32,"step":1},)j"
        R"j({"key":"orient.weight_peel","label":"Peel force weight","type":"float","min":0,"max":1,"default":0.6,"step":0.05},)j"
        R"j({"key":"orient.weight_suction","label":"Suction cup weight","type":"float","min":0,"max":1,"default":0.25,"step":0.05},)j"
        R"j({"key":"orient.weight_islands","label":"Floating island weight","type":"float","min":0,"max":1,"default":0.15,"step":0.05},)j"
        R"j({"key":"pack.enable","label":"Enable packing","type":"bool","default":true},)j"
        R"j({"key":"pack.resolution","label":"Heightmap resolution (mm/px)","type":"float","min":0.01,"max":5,"default":0.1,"step":0.01},)j"
        R"j({"key":"pack.clearance","label":"Part clearance (mm)","type":"float","min":0,"max":20,"default":0.5,"step":0.1},)j"
        R"j({"key":"pack.bed_offset","label":"Bed offset / support lift (mm)","type":"float","min":0,"max":50,"default":5,"step":0.5})j"
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

    const auto* raw = data.sceneGraph.data();
    const size_t sz = data.sceneGraph.size();

    if (sz < sizeof(SceneGraphHeader))
        return Result::fail("positionner: sceneGraph too small for SceneGraphHeader");

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

    // Per-mesh rotation lock: true if the mesh's instance has rotation
    // locked. In the current app every mesh has exactly one instance (see
    // MessageHandler::Model), so "the mesh's instance" is unambiguous; if
    // that ever changes, this takes the first instance found for the mesh.
    std::vector<bool> meshRotationLocked(hdr.meshCount, false);
    for (uint32_t i = 0; i < hdr.instanceCount; ++i) {
        uint32_t m = instancesIn[i].meshIndex;
        if (m < hdr.meshCount &&
            (instancesIn[i].lockMask & static_cast<uint32_t>(TransformLock::Rotation)))
            meshRotationLocked[m] = true;
    }

    struct MeshResult {
        Eigen::Matrix3f R; // best NEW rotation to compose onto each instance's
                           // stored rotation (identity if orient disabled or locked)
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

        if (meshRotationLocked[m]) {
            std::cerr << "  rotation locked — skipping auto-orient\n";
            results[m].R = Eigen::Matrix3f::Identity();
        } else if (orientEnable_ && meshes[m].triangleCount > 0) {
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
    // only the MeshInstance fields are updated. Position/rotation/scale are
    // the source of truth (see MeshTypes.hpp) — transform[16] is always
    // re-derived from them via decodeTransformMatrix, never hand-edited, so it can't
    // drift out of sync the way the old matrix-only version could.

    std::vector<MeshInstance> instancesOut(hdr.instanceCount);
    // Per-mesh FINAL rotation (auto ∘ stored, respecting locks) and scale —
    // used below by the packing step so a previously-rotated/scaled
    // instance's TRUE footprint is what gets packed and grounded, not this
    // pass's auto-rotation delta on an assumed-unscaled mesh (a scaled-down
    // object's drop-test needs its actual, smaller footprint/height, or it
    // gets grounded as if it were still full-size and ends up floating).
    std::vector<Eigen::Matrix3f>  finalR(hdr.meshCount, Eigen::Matrix3f::Identity());
    std::vector<Eigen::Vector3f>  meshScale(hdr.meshCount, Eigen::Vector3f(1.f, 1.f, 1.f));
    std::vector<uint32_t> fixedIdx, toPlaceIdx; // split by position lock

    for (uint32_t i = 0; i < hdr.instanceCount; ++i) {
        instancesOut[i] = instancesIn[i];
        uint32_t m = instancesIn[i].meshIndex;
        if (m >= hdr.meshCount) continue;

        const bool rotLocked = instancesIn[i].lockMask & static_cast<uint32_t>(TransformLock::Rotation);
        const bool posLocked = instancesIn[i].lockMask & static_cast<uint32_t>(TransformLock::Position);

        Eigen::Matrix3f newR;
        if (rotLocked) {
            // Untouched — go straight from stored Euler angles back to the
            // same angles, no matrix round-trip that could drift them.
            instancesOut[i].rotation = instancesIn[i].rotation;
            newR = eulerDegToMatrix(instancesOut[i].rotation);
        } else {
            newR = results[m].R * eulerDegToMatrix(instancesIn[i].rotation);
            instancesOut[i].rotation = matrixToEulerDeg(newR);
        }
        finalR[m] = newR;

        instancesOut[i].scale    = instancesIn[i].scale;    // positioner never touches scale
        meshScale[m] = Eigen::Vector3f(instancesOut[i].scale.x, instancesOut[i].scale.y, instancesOut[i].scale.z);
        instancesOut[i].position = instancesIn[i].position; // packing below may overwrite this if unlocked

        const Vec3f pivot = mesh::bboxCenter(meshes[m].aabbMin, meshes[m].aabbMax);
        mesh::decodeTransformMatrix(instancesOut[i].position, instancesOut[i].rotation,
                                    instancesOut[i].scale, meshes[m].aabbMin, meshes[m].aabbMax,
                                    instancesOut[i].transform);

        // Tight world AABB from the actual (welded) mesh geometry, scaled
        // and rotated about that same pivot — not just the rotated axis-
        // aligned box, which would over-estimate it for non-box-like meshes.
        const Eigen::Vector3f pivotVec(pivot.x, pivot.y, pivot.z);
        const Eigen::Vector3f scaleVec(instancesOut[i].scale.x, instancesOut[i].scale.y, instancesOut[i].scale.z);
        const Eigen::MatrixXf Vshifted = getWeldedMesh(data, m).V.rowwise() - pivotVec.transpose();
        const Eigen::MatrixXf Vscaled  = Vshifted * scaleVec.asDiagonal();
        const Eigen::MatrixXf Vr       = (newR * Vscaled.transpose()).transpose();
        const Eigen::Vector3f posVec(instancesOut[i].position.x, instancesOut[i].position.y, instancesOut[i].position.z);
        const Eigen::Vector3f wMin = Vr.colwise().minCoeff().transpose() + posVec;
        const Eigen::Vector3f wMax = Vr.colwise().maxCoeff().transpose() + posVec;
        instancesOut[i].aabbMin = { wMin.x(), wMin.y(), wMin.z() };
        instancesOut[i].aabbMax = { wMax.x(), wMax.y(), wMax.z() };

        (posLocked ? fixedIdx : toPlaceIdx).push_back(i);
    }

    // ── Packing ───────────────────────────────────────────────────────────────
    if (packEnable_) {
        if (data.printer.buildXMm == 0.f) {
            std::cerr << "[positionner] packing skipped: printer profile not set\n";
        } else if (toPlaceIdx.empty()) {
            std::cerr << "[positionner] packing skipped: every instance's position is locked\n";
        } else {
            PackingPass::Params pp;
            pp.plateX       = data.printer.buildXMm;
            pp.plateY       = data.printer.buildYMm;
            pp.plateZ       = data.printer.buildZMm;
            pp.resolution   = packResolution_;
            pp.clearance    = packClearance_;
            pp.bedOffset    = bedOffset_;

            // Scaled + rotated object-space vertices per unique mesh, using
            // the FINAL rotation/scale (auto ∘ stored, respecting locks)
            // about the SAME pivot every other consumer uses (bboxCenter —
            // see MeshTransform.hpp). Scale matters here: a scaled-down
            // object's drop-test needs its actual (smaller) footprint and
            // height, or the plate thinks it's still full-size and grounds
            // it too high, leaving the real (smaller) mesh floating. rotV is
            // pivot-relative: a vertex at the pivot maps to (0,0,0), so
            // whatever (tx,ty,tz) the packer assigns IS the world-space
            // location of the pivot, directly usable as `position`.
            std::vector<Eigen::MatrixXf> rotV(hdr.meshCount);
            std::vector<Eigen::MatrixXi> faces(hdr.meshCount);
            for (uint32_t m = 0; m < hdr.meshCount; ++m) {
                const WeldedMesh& wm = getWeldedMesh(data, m);
                const Vec3f pivot = mesh::bboxCenter(meshes[m].aabbMin, meshes[m].aabbMax);
                const Eigen::Vector3f pivotVec(pivot.x, pivot.y, pivot.z);
                const Eigen::MatrixXf Vshifted = wm.V.rowwise() - pivotVec.transpose();
                const Eigen::MatrixXf Vscaled  = Vshifted * meshScale[m].asDiagonal();
                rotV[m]  = (finalR[m] * Vscaled.transpose()).transpose();
                faces[m] = wm.F;
            }

            // Only instances NOT position-locked are handed to the packer as
            // things it may place; their mesh index / object-space rotated AABBs.
            std::vector<uint32_t>        instMesh(toPlaceIdx.size());
            std::vector<Eigen::Vector3f> aabbMin(toPlaceIdx.size());
            std::vector<Eigen::Vector3f> aabbMax(toPlaceIdx.size());
            for (size_t k = 0; k < toPlaceIdx.size(); ++k) {
                uint32_t m = instancesOut[toPlaceIdx[k]].meshIndex;
                instMesh[k] = m;
                aabbMin[k]  = rotV[m].colwise().minCoeff().transpose();
                aabbMax[k]  = rotV[m].colwise().maxCoeff().transpose();
            }

            // Position-locked instances: rotV is pivot-relative (see above),
            // so adding their current `position` directly gives their true
            // world footprint — stamped as a fixed obstacle rather than placed.
            std::vector<Eigen::MatrixXf> fixedWorldVerts(fixedIdx.size());
            for (size_t k = 0; k < fixedIdx.size(); ++k) {
                const MeshInstance& fi = instancesOut[fixedIdx[k]];
                fixedWorldVerts[k] = rotV[fi.meshIndex].rowwise()
                                   + Eigen::RowVector3f(fi.position.x, fi.position.y, fi.position.z);
            }

            auto placed = packingPass_.pack(rotV, faces, instMesh, aabbMin, aabbMax, pp, fixedWorldVerts);

            for (const auto& pl : placed) {
                const uint32_t instIdx = toPlaceIdx[pl.instanceIdx]; // pl.instanceIdx indexes the toPlaceIdx-filtered arrays above
                const uint32_t m       = instancesOut[instIdx].meshIndex;

                // (pl.tx,pl.ty,pl.tz) IS the pivot's world location (see the
                // rotV note above) — i.e. exactly `position`. Rebuild the
                // final transform through the same decode path everything
                // else uses, so scale gets applied correctly around this
                // pivot instead of being baked in via a hand-poked
                // translation (that mismatch was the root cause of scaled
                // objects landing below the plate).
                instancesOut[instIdx].position = { pl.tx, pl.ty, pl.tz };
                mesh::decodeTransformMatrix(instancesOut[instIdx].position, instancesOut[instIdx].rotation,
                                            instancesOut[instIdx].scale,
                                            meshes[m].aabbMin, meshes[m].aabbMax,
                                            instancesOut[instIdx].transform);

                // Tight, scale-correct world AABB — same approach as the
                // pre-packing pass above, now at the packer's final position.
                const Vec3f pivot = mesh::bboxCenter(meshes[m].aabbMin, meshes[m].aabbMax);
                const Eigen::Vector3f pivotVec(pivot.x, pivot.y, pivot.z);
                const Eigen::Vector3f scaleVec(instancesOut[instIdx].scale.x,
                                               instancesOut[instIdx].scale.y,
                                               instancesOut[instIdx].scale.z);
                const Eigen::MatrixXf Vshifted = getWeldedMesh(data, m).V.rowwise() - pivotVec.transpose();
                const Eigen::MatrixXf Vscaled  = Vshifted * scaleVec.asDiagonal();
                const Eigen::MatrixXf Vr       = (finalR[m] * Vscaled.transpose()).transpose();
                const Eigen::Vector3f posVec(pl.tx, pl.ty, pl.tz);
                const Eigen::Vector3f wMin = Vr.colwise().minCoeff().transpose() + posVec;
                const Eigen::Vector3f wMax = Vr.colwise().maxCoeff().transpose() + posVec;
                instancesOut[instIdx].aabbMin = { wMin.x(), wMin.y(), wMin.z() };
                instancesOut[instIdx].aabbMax = { wMax.x(), wMax.y(), wMax.z() };
            }
        }
    }

    // Write updated instances back into sceneGraph
    // (the mesh blocks preceding the instances are untouched)
    uint8_t* instDst = reinterpret_cast<uint8_t*>(data.sceneGraph.data())
                       + (p - reinterpret_cast<const uint8_t*>(raw));
    std::memcpy(instDst, instancesOut.data(),
                hdr.instanceCount * sizeof(MeshInstance));

    return Result::ok();
}

// ─── initGpu ─────────────────────────────────────────────────────────────────

void PositionnerStage::initGpu(const GpuContext& ctx) {
    std::cerr << "[positionner] initGpu called, available=" << ctx.isAvailable() << "\n";
    if (!ctx.isAvailable()) return;
    VULKAN_HPP_DEFAULT_DISPATCHER.init(
        static_cast<VkInstance>(ctx.instance), ::vkGetInstanceProcAddr,
        static_cast<VkDevice>(ctx.device));
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
    PIPELINE_PLUGIN_API int    pluginApiVersion()          { return PIPELINE_API_VERSION; }
    PIPELINE_PLUGIN_API Stage* createStage()               { return new PositionnerStage(); }
    PIPELINE_PLUGIN_API void   destroyStage(Stage* s)      { delete s; }
}
#endif
