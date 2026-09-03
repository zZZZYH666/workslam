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

    point.Promote();
    Check(point.IsPromoted(), "point can be promoted");
    Check(MapPoint::GetSemanticLifecycleEvents().size() == 1, "promotion event is recorded");

    return failures == 0 ? 0 : 1;
}
