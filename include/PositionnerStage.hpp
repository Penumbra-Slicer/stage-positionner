#pragma once
#include <Stage.hpp>
#include "SuctionPass.hpp"
#include "PackingPass.hpp"
#include <memory>

/// Positions meshes on the build plate:
///   1. Orientation optimizer — rotates each instance to minimise peel force,
///      suction cups, and floating islands.
///   2. Packing              — translates each oriented instance to fill the
///      build plate efficiently (see plan-packing.md).
///
/// Reads  : mesh/scene_graph_v1
/// Writes : mesh/scene_graph_v1  (same format, updated transforms + AABBs)
class PositionnerStage : public Stage {
public:
    Result          process(DataBuffer& data) override;
    ParameterSchema describe() const override;
    void            configure(const ParameterValues& values) override;
    StageContract   contract() const override;
    void            initGpu(const GpuContext& ctx) override;
    std::string     name() const override { return "Positionner"; }
    std::string     describeJson() const override;

private:
    // Orientation
    bool  orientEnable_    = true;
    float weightPeel_      = 0.6f;
    float weightSuction_   = 0.25f;
    float weightIslands_   = 0.15f;

    // Packing
    bool  packEnable_      = true;
    bool  keepPosition_    = false;
    float packResolution_  = 0.1f;
    float packClearance_   = 0.5f;
    float bedOffset_       = 0.0f;

    // GPU suction pass — null when no GPU context is available.
    std::unique_ptr<SuctionPass> suctionPass_;
    GpuContext gpuCtx_;

    PackingPass packingPass_;
};
