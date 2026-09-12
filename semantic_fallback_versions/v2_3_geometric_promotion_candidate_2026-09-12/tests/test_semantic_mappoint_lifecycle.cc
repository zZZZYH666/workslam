#include "MapPoint.h"

#include <cmath>
#include <iostream>

using namespace ORB_SLAM3;

namespace
{
int failures = 0;
void Check(bool condition, const char *message)
{
    if(!condition)
    {
        std::cerr << "FAIL: " << message << std::endl;
        ++failures;
    }
}
}

int main()
{
    MapPoint::ClearSemanticLifecycleEvents();
    MapPoint point;
    point.SetSemanticState(SemanticMapPointState::PROVISIONAL);

    point.RegisterSemanticVisibility(10);
    point.RegisterSemanticVisibility(10); // idempotent within a frame
    Check(point.SemanticVisibleFrames() == 1, "visibility is counted once per frame");

    point.RegisterSemanticMatch(true, 1.0f, true, 10);
    point.RegisterSemanticMatch(true, 1.0f, true, 10); // idempotent within a frame
    Check(point.SemanticMatchedFrames() == 1, "match is counted once per frame");
    Check(point.SemanticValidReprojectionObservations() == 1, "valid reprojection is counted");
    Check(std::fabs(point.SemanticMeanReprojectionError() - 1.0f) < 1e-5f, "mean reprojection error is correct");

    point.RegisterSemanticKeyFrameObservation(3, true);
    point.RegisterSemanticKeyFrameObservation(3, true);
    point.RegisterSemanticKeyFrameObservation(3, false);
    Check(point.SemanticKeyFrameObservations() == 1, "keyframe count is idempotent");
    Check(std::fabs(point.SemanticStaticRatio() - 1.0f) < 1e-5f, "mixed stereo observation keeps static support");

    SemanticPromotionConfig config;
    config.minVisibleFrames = 1;
    config.minKeyFrames = 1;
    config.minMatchRatio = 1.0f;
    config.maxReprojectionError = 2.0f;
    config.minStaticRatio = 0.5f;
    config.maxAgeKeyFrames = 8;
    config.maxDynamicObservations = 0;
    Check(point.IsLoopEligible(config, 3), "eligible point passes loop gate");
    Check(!point.IsLoopEligible(config, 12), "stale point fails loop gate");

    point.Promote("promotion_semantic");
    Check(point.IsPromoted(), "point can be promoted");
    Check(MapPoint::GetSemanticLifecycleEvents().size() == 1, "promotion event is recorded");
    Check(MapPoint::GetSemanticLifecycleEvents()[0].reason == "promotion_semantic", "semantic promotion reason is recorded");

    MapPoint geometricPoint;
    geometricPoint.SetSemanticState(SemanticMapPointState::PROVISIONAL);
    for(int frame = 20; frame < 28; ++frame)
    {
        geometricPoint.RegisterSemanticVisibility(frame);
        geometricPoint.RegisterSemanticMatch(false, 0.75f, true, frame);
    }
    geometricPoint.RegisterSemanticKeyFrameObservation(4, false);
    geometricPoint.RegisterSemanticKeyFrameObservation(5, false);
    geometricPoint.RegisterSemanticKeyFrameObservation(6, false);

    SemanticPromotionConfig semanticConfig = config;
    semanticConfig.minVisibleFrames = 5;
    semanticConfig.minKeyFrames = 2;
    semanticConfig.minMatchRatio = 0.60f;
    semanticConfig.minStaticRatio = 0.70f;
    Check(!geometricPoint.CanPromote(semanticConfig), "unknown-only point fails semantic promotion");

    SemanticPromotionConfig geometricConfig;
    geometricConfig.minVisibleFrames = 8;
    geometricConfig.minKeyFrames = 3;
    geometricConfig.minMatchRatio = 0.70f;
    geometricConfig.maxReprojectionError = 1.5f;
    geometricConfig.minStaticRatio = 0.0f;
    geometricConfig.maxDynamicObservations = 0;
    Check(geometricPoint.CanPromote(geometricConfig), "geometrically stable unknown point can promote");
    geometricPoint.Promote("promotion_geometric");
    Check(geometricPoint.IsPromoted(), "geometric point is promoted");
    Check(MapPoint::GetSemanticLifecycleEvents().back().reason == "promotion_geometric", "geometric promotion reason is recorded");

    return failures == 0 ? 0 : 1;
}
