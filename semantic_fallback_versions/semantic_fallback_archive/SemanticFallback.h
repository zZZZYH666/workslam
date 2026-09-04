/**
 * Semantic feature selection and adaptive fallback used by the FinnForest
 * stereo experiment.
 *
 * The implementation is header-only so the complete fallback policy lives in
 * one file and can be reused by other frame constructors without duplicating
 * the selection rules.
 */
#ifndef ORB_SLAM3_SEMANTIC_FALLBACK_H
#define ORB_SLAM3_SEMANTIC_FALLBACK_H

#include "Settings.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

namespace ORB_SLAM3 {
namespace SemanticFallback {

struct SelectionStats
{
    int staticCount = 0;
    int selectedStaticCount = 0;
    int fallbackCount = 0;
    float gridCoverage = 0.0f;
};

// Trigger bits are persisted in Frame::mnSemanticFallbackReason and the CSV
// experiment output: 1=insufficient static features, 2=poor static grid
// coverage, 4=previous-frame tracking weak, 8=stereo matches weak.
struct FallbackDecision
{
    bool requested = false;
    int reason = 0;
};

inline FallbackDecision EvaluateRequest(const SemanticConfig &config,
                                        const SelectionStats &left,
                                        const SelectionStats &right,
                                        const bool previousMetricsLow)
{
    FallbackDecision decision;
    if(left.staticCount < config.minStaticFeatures ||
       right.staticCount < config.minStaticFeatures)
    {
        decision.reason |= 1;
    }
    if(left.gridCoverage < config.minGridCoverage ||
       right.gridCoverage < config.minGridCoverage)
    {
        decision.reason |= 2;
    }
    if(previousMetricsLow)
        decision.reason |= 4;
    decision.requested = config.enableFallback && decision.reason != 0;
    return decision;
}

inline bool ShouldRetryForStereo(const SemanticConfig &config,
                                 const bool fallbackAlreadyRequested,
                                 const int stereoMatches)
{
    return config.enableFallback && !fallbackAlreadyRequested &&
           stereoMatches < config.minStereoMatches;
}

// Select strongest features in round-robin grid order. This keeps the
// selected set spatially distributed while respecting the feature budget.
inline std::vector<int> SelectByGrid(const std::vector<int> &indices,
                                     const std::vector<cv::KeyPoint> &keys,
                                     const cv::Size &size,
                                     const int limit)
{
    if(limit <= 0 || indices.empty())
        return std::vector<int>();

    const int cols = 8;
    const int rows = 6;
    std::vector<std::vector<int> > cells(static_cast<size_t>(cols * rows));
    for(int index : indices)
    {
        const int col = std::max(0, std::min(cols - 1,
            static_cast<int>(keys[index].pt.x * cols / std::max(1, size.width))));
        const int row = std::max(0, std::min(rows - 1,
            static_cast<int>(keys[index].pt.y * rows / std::max(1, size.height))));
        cells[static_cast<size_t>(row * cols + col)].push_back(index);
    }

    for(std::vector<int> &cell : cells)
        std::sort(cell.begin(), cell.end(), [&keys](int a, int b) {
            return keys[a].response > keys[b].response;
        });

    std::vector<int> selected;
    selected.reserve(static_cast<size_t>(std::min(limit, static_cast<int>(indices.size()))));
    for(size_t rank = 0; selected.size() < static_cast<size_t>(limit); ++rank)
    {
        bool added = false;
        for(std::vector<int> &cell : cells)
        {
            if(rank < cell.size())
            {
                selected.push_back(cell[rank]);
                added = true;
                if(selected.size() == static_cast<size_t>(limit))
                    break;
            }
        }
        if(!added)
            break;
    }
    return selected;
}

// Filter keypoints/descriptors according to a semantic mask. Source values
// are 0 for static-mask features and 1 for unknown-region fallback features.
inline SelectionStats SelectFeatures(const cv::Mat &mask,
                                     std::vector<cv::KeyPoint> &keys,
                                     cv::Mat &descriptors,
                                     const SemanticConfig &config,
                                     const bool allowFallback,
                                     const bool fallbackRequested,
                                     std::vector<unsigned char> &sources)
{
    SelectionStats stats;
    std::vector<int> staticIndices;
    std::vector<int> unknownIndices;
    staticIndices.reserve(keys.size());
    unknownIndices.reserve(keys.size());

    const cv::Size maskSize = mask.empty() ? cv::Size(1, 1) : mask.size();
    for(size_t i = 0; i < keys.size(); ++i)
    {
        const int x = cvRound(keys[i].pt.x);
        const int y = cvRound(keys[i].pt.y);
        const bool isStatic = mask.empty() ||
            (x >= 0 && y >= 0 && x < mask.cols && y < mask.rows &&
             mask.at<unsigned char>(y, x) != 0);
        (isStatic ? staticIndices : unknownIndices).push_back(static_cast<int>(i));
    }

    stats.staticCount = static_cast<int>(staticIndices.size());
    bool occupied[48] = {false};
    for(int index : staticIndices)
    {
        const int col = std::max(0, std::min(7,
            static_cast<int>(keys[index].pt.x * 8 / std::max(1, maskSize.width))));
        const int row = std::max(0, std::min(5,
            static_cast<int>(keys[index].pt.y * 6 / std::max(1, maskSize.height))));
        occupied[row * 8 + col] = true;
    }
    int occupiedCells = 0;
    for(bool value : occupied)
        occupiedCells += value ? 1 : 0;
    stats.gridCoverage = static_cast<float>(occupiedCells) / 48.0f;

    std::vector<int> selected = allowFallback
        ? SelectByGrid(staticIndices, keys, maskSize, config.targetFeatures)
        : staticIndices;
    stats.selectedStaticCount = static_cast<int>(selected.size());

    const int fallbackLimit = static_cast<int>(std::floor(
        config.targetFeatures * config.maxFallbackRatio));
    if(allowFallback && fallbackRequested && fallbackLimit > 0 &&
       selected.size() < static_cast<size_t>(config.targetFeatures))
    {
        const int deficit = config.targetFeatures - static_cast<int>(selected.size());
        const std::vector<int> fallback = SelectByGrid(unknownIndices, keys, maskSize,
            std::min(deficit, fallbackLimit));
        selected.insert(selected.end(), fallback.begin(), fallback.end());
        stats.fallbackCount = static_cast<int>(fallback.size());
    }

    std::vector<cv::KeyPoint> filteredKeys;
    cv::Mat filteredDescriptors;
    filteredKeys.reserve(selected.size());
    sources.clear();
    sources.reserve(selected.size());
    for(int index : selected)
    {
        filteredKeys.push_back(keys[index]);
        filteredDescriptors.push_back(descriptors.row(index));
        sources.push_back(std::find(staticIndices.begin(), staticIndices.end(), index) !=
                          staticIndices.end() ? 0 : 1);
    }
    keys.swap(filteredKeys);
    descriptors = filteredDescriptors;
    return stats;
}

} // namespace SemanticFallback
} // namespace ORB_SLAM3

#endif // ORB_SLAM3_SEMANTIC_FALLBACK_H
