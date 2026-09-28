#include <System.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct Timestamp
{
    long long nanoseconds;
    double seconds;
};

std::vector<Timestamp> LoadTimestamps(const std::string &path)
{
    std::ifstream input(path);
    if(!input)
        throw std::runtime_error("Cannot open timestamp file: " + path);

    std::vector<Timestamp> timestamps;
    long long seconds = 0;
    long double nanoseconds = 0.0;
    while(input >> seconds >> nanoseconds)
    {
        const long long timestampNs = seconds * 1000000000LL + static_cast<long long>(nanoseconds + 0.5L);
        timestamps.push_back({timestampNs, static_cast<double>(timestampNs) * 1e-9});
    }
    return timestamps;
}

std::string FrameName(size_t index)
{
    std::ostringstream name;
    name << std::setfill('0') << std::setw(6) << index << ".png";
    return name.str();
}

struct MapTrajectoryStats
{
    std::string filename;
    size_t firstFrame = 0;
    size_t lastFrame = 0;
    size_t frameCount = 0;
};
}

int main(int argc, char **argv)
{
    if(argc < 7 || argc > 9)
    {
        std::cerr << "Usage: finnforest_stereo_semantic vocabulary settings dataset fused_masks timestamps output_dir [max_frames] [start_index]\n";
        return 1;
    }

    try
    {
        const std::string dataset = argv[3];
        const std::string masks = argv[4];
        const std::string output = argv[6];
        std::vector<Timestamp> timestamps = LoadTimestamps(argv[5]);
        size_t startIndex = 0;
        size_t maxFrames = 0;
        if(argc >= 8)
            maxFrames = std::stoul(argv[7]);
        if(argc == 9)
            startIndex = std::stoul(argv[8]);
        if(startIndex >= timestamps.size())
            throw std::runtime_error("start_index is outside the timestamp range");
        const size_t endIndex = maxFrames == 0 ? timestamps.size() : std::min(timestamps.size(), startIndex + maxFrames);

        std::ofstream timing(output + "/frame_times_semantic.csv");
        std::ofstream stats(output + "/frame_stats_semantic.csv");
        std::ofstream mapManifest(output + "/trajectory_maps_semantic.csv");
        std::map<unsigned long, std::ofstream> mapTrajectories;
        std::map<unsigned long, MapTrajectoryStats> mapStats;
        std::ofstream lifecycle(output + "/semantic_lifecycle_events.csv");
        std::ofstream backend(output + "/backend_interval_stats.csv");
        if(!timing || !stats || !mapManifest || !lifecycle || !backend)
            throw std::runtime_error("Cannot create result CSV files in: " + output);
        timing << "index,timestamp_ns,track_time_sec\n";
        stats << "index,timestamp_ns,raw_left_features,raw_right_features,static_left_features,static_right_features,fallback_left_features,fallback_right_features,stereo_matches,grid_coverage,fallback_used,turn_fallback_used,fallback_reason,tracking_inliers,tracking_state,left_static_pixels,right_static_pixels,track_time_ms,static_map_points,provisional_map_points,promoted_map_points,provisional_visible,provisional_matched,promotion_count,rejection_count\n";
        mapManifest << "map_id,trajectory_file,first_frame,last_frame,frame_count\n";
        lifecycle << "frame_id,keyframe_id,map_id,map_point_id,old_state,new_state,reason,visible_frames,matched_frames,match_ratio,keyframe_observations,static_ratio,mean_reprojection_error,valid_reprojection_observations\n";
        backend << "frame_id,timestamp_ns,map_id,tracking_state,local_map_points,tracking_inliers,last_keyframe_id,atlas_maps,atlas_keyframes,atlas_points,active_provisional,promoted_points,rejected_points,loop_candidates,loop_matches,loop_trusted_inliers,loop_provisional_inliers\n";

        std::cout << "initializing SLAM" << std::endl;
        ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::STEREO, false);
        slam.ClearSemanticLifecycleEvents();
        std::cout << "SLAM initialized" << std::endl;

        size_t lifecycleCursor = 0;

        for(size_t index = startIndex; index < endIndex; ++index)
        {
            const std::string name = FrameName(index);
            const std::string leftPath = dataset + "/images_cam2_sr22555667/" + name;
            const std::string rightPath = dataset + "/images_cam3_sr22555660/" + name;
            const std::string leftMaskPath = masks + "/cam2/" + name;
            const std::string rightMaskPath = masks + "/cam3/" + name;

            cv::Mat left = cv::imread(leftPath, cv::IMREAD_UNCHANGED);
            cv::Mat right = cv::imread(rightPath, cv::IMREAD_UNCHANGED);
            cv::Mat leftMask = cv::imread(leftMaskPath, cv::IMREAD_UNCHANGED);
            cv::Mat rightMask = cv::imread(rightMaskPath, cv::IMREAD_UNCHANGED);
            if(left.empty() || right.empty() || leftMask.empty() || rightMask.empty())
                throw std::runtime_error("Missing image or mask for frame " + name);
            if(left.type() != CV_8UC1 || right.type() != CV_8UC1 || leftMask.type() != CV_8UC1 || rightMask.type() != CV_8UC1 ||
               left.size() != leftMask.size() || right.size() != rightMask.size())
                throw std::runtime_error("Invalid image/mask type or dimensions for frame " + name);

            const auto start = std::chrono::steady_clock::now();
            const Sophus::SE3f Tcw = slam.TrackStereo(left, right, timestamps[index].seconds, leftMask, rightMask);
            const auto end = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(end - start).count();

            const std::vector<cv::KeyPoint> keys = slam.GetTrackedKeyPointsUn();
            const std::vector<ORB_SLAM3::MapPoint*> points = slam.GetTrackedMapPoints();
            size_t inliers = 0;
            for(ORB_SLAM3::MapPoint *point : points)
                inliers += point != nullptr;

            timing << index << ',' << timestamps[index].nanoseconds << ',' << std::fixed << std::setprecision(9) << elapsed << '\n';
            stats << index << ',' << timestamps[index].nanoseconds << ',' << slam.GetSemanticRawLeftFeatures() << ','
                  << slam.GetSemanticRawRightFeatures() << ',' << slam.GetSemanticStaticFeatures() << ','
                  << slam.GetSemanticStaticRightFeatures() << ',' << slam.GetSemanticFallbackFeatures() << ','
                  << slam.GetSemanticFallbackRightFeatures() << ','
                  << slam.GetSemanticStereoMatches() << ',' << std::fixed << std::setprecision(6) << slam.GetSemanticGridCoverage() << ','
                  << (slam.SemanticFallbackUsed() ? 1 : 0) << ',' << (slam.SemanticTurnFallbackUsed() ? 1 : 0) << ',' << slam.GetSemanticFallbackReason() << ',' << inliers << ','
                  << slam.GetTrackingState() << ',' << cv::countNonZero(leftMask) << ',' << cv::countNonZero(rightMask) << ','
                  << std::fixed << std::setprecision(6) << elapsed * 1000.0 << ','
                  << slam.GetSemanticStaticMapPoints() << ',' << slam.GetSemanticProvisionalMapPoints() << ','
                  << slam.GetSemanticPromotedMapPoints() << ',' << slam.GetSemanticProvisionalVisible() << ','
                  << slam.GetSemanticProvisionalMatched() << ',' << slam.GetSemanticPromotionCount() << ','
                  << slam.GetSemanticRejectionCount() << '\n';

            const int trackingState = slam.GetTrackingState();
            const bool poseValid = (trackingState == 2 || trackingState == 5) && Tcw.matrix().allFinite();
            const Sophus::SE3f Twc = Tcw.inverse();
            const Eigen::Vector3f translation = Twc.translation();
            const Eigen::Quaternionf quaternion = Twc.unit_quaternion();
            const unsigned long mapId = slam.GetCurrentMapId();
            std::ofstream &mapTrajectory = mapTrajectories[mapId];
            if(!mapTrajectory.is_open())
            {
                const std::string filename = "trajectory_map_" + std::to_string(mapId) + "_semantic.csv";
                mapTrajectory.open((output + "/" + filename).c_str());
                if(!mapTrajectory)
                    throw std::runtime_error("Cannot create map trajectory file: " + output + "/" + filename);
                mapTrajectory << "frame,timestamp_ns,tracking_state,pose_valid,tx,ty,tz,qx,qy,qz,qw,map_id\n";
                MapTrajectoryStats &trajectoryStats = mapStats[mapId];
                trajectoryStats.filename = filename;
                trajectoryStats.firstFrame = index;
                trajectoryStats.lastFrame = index;
                trajectoryStats.frameCount = 0;
            }
            mapTrajectory << index << ',' << timestamps[index].nanoseconds << ','
                       << trackingState << ',' << (poseValid ? 1 : 0) << ','
                       << std::fixed << std::setprecision(9) << translation.x() << ',' << translation.y() << ','
                       << translation.z() << ',' << quaternion.x() << ',' << quaternion.y() << ','
                       << quaternion.z() << ',' << quaternion.w() << ',' << mapId << '\n';
            MapTrajectoryStats &trajectoryStats = mapStats[mapId];
            trajectoryStats.lastFrame = index;
            ++trajectoryStats.frameCount;

            const std::vector<ORB_SLAM3::MapPoint::SemanticLifecycleEvent> events = slam.GetSemanticLifecycleEventsSince(lifecycleCursor);
            for(const auto &event : events)
            {
                lifecycle << event.frameId << ',' << event.keyframeId << ',' << event.mapId << ',' << event.mapPointId << ','
                          << event.oldState << ',' << event.newState << ',' << event.reason << ',' << event.visibleFrames << ','
                          << event.matchedFrames << ',' << std::fixed << std::setprecision(6) << event.matchRatio << ','
                          << event.keyframeObservations << ',' << event.staticRatio << ',' << event.meanReprojectionError << ','
                          << event.validReprojectionObservations << '\n';
            }
            lifecycleCursor += events.size();

            backend << index << ',' << timestamps[index].nanoseconds << ',' << slam.GetCurrentMapId() << ','
                    << slam.GetTrackingState() << ',' << slam.GetLocalMapPointCount() << ',' << slam.GetTrackingInliers() << ','
                    << slam.GetLastKeyFrameId() << ',' << slam.GetAtlasMapCount() << ',' << slam.GetAtlasKeyFrames() << ','
                    << slam.GetAtlasMapPoints() << ',' << slam.GetSemanticProvisionalMapPoints() << ','
                    << slam.GetSemanticPromotedMapPoints() << ',' << slam.GetSemanticRejectionCount() << ','
                    << slam.GetSemanticLoopCandidates() << ',' << slam.GetSemanticLoopMatches() << ','
                    << slam.GetSemanticLoopTrustedInliers() << ',' << slam.GetSemanticLoopProvisionalInliers() << '\n';

            if((index - startIndex + 1) % 100 == 0 || index + 1 == endIndex)
                std::cout << "processed " << (index - startIndex + 1) << '/' << (endIndex - startIndex) << std::endl;
        }

        slam.Shutdown();
        for(std::map<unsigned long, std::ofstream>::iterator it = mapTrajectories.begin(); it != mapTrajectories.end(); ++it)
            it->second.close();
        for(std::map<unsigned long, MapTrajectoryStats>::const_iterator it = mapStats.begin(); it != mapStats.end(); ++it)
        {
            const MapTrajectoryStats &trajectoryStats = it->second;
            mapManifest << it->first << ',' << trajectoryStats.filename << ',' << trajectoryStats.firstFrame << ','
                        << trajectoryStats.lastFrame << ',' << trajectoryStats.frameCount << '\n';
        }
        const std::vector<ORB_SLAM3::MapPoint::SemanticLifecycleEvent> finalEvents = slam.GetSemanticLifecycleEventsSince(lifecycleCursor);
        for(const auto &event : finalEvents)
        {
            lifecycle << event.frameId << ',' << event.keyframeId << ',' << event.mapId << ',' << event.mapPointId << ','
                      << event.oldState << ',' << event.newState << ',' << event.reason << ',' << event.visibleFrames << ','
                      << event.matchedFrames << ',' << std::fixed << std::setprecision(6) << event.matchRatio << ','
                      << event.keyframeObservations << ',' << event.staticRatio << ',' << event.meanReprojectionError << ','
                      << event.validReprojectionObservations << '\n';
        }
        lifecycleCursor += finalEvents.size();
        std::ofstream summary(output + "/run_summary.csv");
        if(!summary)
            throw std::runtime_error("Cannot create run summary in: " + output);
        summary << "processed_frames,atlas_maps,current_map_keyframes,current_map_points,static_map_points,provisional_map_points,promoted_map_points,promotion_count,rejection_count\n";
        summary << (endIndex - startIndex) << ',' << slam.GetAtlasMapCount() << ',' << slam.GetAtlasKeyFrames() << ','
                << slam.GetAtlasMapPoints() << ',' << slam.GetSemanticStaticMapPoints() << ','
                << slam.GetSemanticProvisionalMapPoints() << ',' << slam.GetSemanticPromotedMapPoints() << ','
                << slam.GetSemanticPromotionCount() << ',' << slam.GetSemanticRejectionCount() << '\n';
        slam.SaveTrajectoryEuRoC(output + "/CameraTrajectory_semantic.txt");
        slam.SaveKeyFrameTrajectoryEuRoC(output + "/KeyFrameTrajectory_semantic.txt");
        std::cout << "processed_frames=" << (endIndex - startIndex) << std::endl;
    }
    catch(const std::exception &error)
    {
        std::cerr << "error: " << error.what() << std::endl;
        return 2;
    }
    return 0;
}
