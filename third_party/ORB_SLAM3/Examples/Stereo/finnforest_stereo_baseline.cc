#include <System.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
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
}

int main(int argc, char **argv)
{
    if(argc < 6 || argc > 7)
    {
        std::cerr << "Usage: finnforest_stereo_baseline vocabulary settings dataset timestamps output_dir [max_frames]\n";
        return 1;
    }

    try
    {
        const std::string dataset = argv[3];
        const std::string output = argv[5];
        std::vector<Timestamp> timestamps = LoadTimestamps(argv[4]);
        const size_t maxFrames = argc == 7 ? std::stoul(argv[6]) : 0;
        const size_t endIndex = maxFrames == 0 ? timestamps.size() : std::min(timestamps.size(), maxFrames);

        std::ofstream timing(output + "/frame_times_baseline.csv");
        std::ofstream stats(output + "/frame_stats_baseline.csv");
        if(!timing || !stats)
            throw std::runtime_error("Cannot create result CSV files in: " + output);
        timing << "index,timestamp_ns,track_time_sec\n";
        stats << "index,timestamp_ns,tracked_keypoints,tracking_inliers,tracking_state,track_time_ms\n";

        std::cout << "initializing SLAM" << std::endl;
        ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::STEREO, false);
        std::cout << "SLAM initialized" << std::endl;

        for(size_t index = 0; index < endIndex; ++index)
        {
            const std::string name = FrameName(index);
            const cv::Mat left = cv::imread(dataset + "/images_cam2_sr22555667/" + name, cv::IMREAD_UNCHANGED);
            const cv::Mat right = cv::imread(dataset + "/images_cam3_sr22555660/" + name, cv::IMREAD_UNCHANGED);
            if(left.empty() || right.empty())
                throw std::runtime_error("Missing image for frame " + name);

            const auto start = std::chrono::steady_clock::now();
            slam.TrackStereo(left, right, timestamps[index].seconds);
            const auto end = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(end - start).count();

            const std::vector<cv::KeyPoint> keys = slam.GetTrackedKeyPointsUn();
            const std::vector<ORB_SLAM3::MapPoint*> points = slam.GetTrackedMapPoints();
            size_t inliers = 0;
            for(ORB_SLAM3::MapPoint *point : points)
                inliers += point != nullptr;

            timing << index << ',' << timestamps[index].nanoseconds << ',' << std::fixed << std::setprecision(9) << elapsed << '\n';
            stats << index << ',' << timestamps[index].nanoseconds << ',' << keys.size() << ',' << inliers << ','
                  << slam.GetTrackingState() << ',' << std::fixed << std::setprecision(6) << elapsed * 1000.0 << '\n';

            if((index + 1) % 100 == 0 || index + 1 == endIndex)
                std::cout << "processed " << (index + 1) << '/' << endIndex << std::endl;
        }

        slam.Shutdown();
        std::ofstream summary(output + "/run_summary.csv");
        if(!summary)
            throw std::runtime_error("Cannot create run summary in: " + output);
        summary << "processed_frames,atlas_maps,current_map_keyframes,current_map_points\n";
        summary << endIndex << ',' << slam.GetAtlasMapCount() << ',' << slam.GetAtlasKeyFrames() << ','
                << slam.GetAtlasMapPoints() << '\n';
        slam.SaveTrajectoryEuRoC(output + "/CameraTrajectory_baseline.txt");
        slam.SaveKeyFrameTrajectoryEuRoC(output + "/KeyFrameTrajectory_baseline.txt");
        std::cout << "processed_frames=" << endIndex << std::endl;
    }
    catch(const std::exception &error)
    {
        std::cerr << "error: " << error.what() << std::endl;
        return 2;
    }
    return 0;
}
