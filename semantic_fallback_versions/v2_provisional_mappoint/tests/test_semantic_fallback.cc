#include "SemanticFallback.h"
#include <opencv2/core.hpp>
#include <fstream>
#include <iostream>
#include <cassert>

using namespace ORB_SLAM3;
using namespace ORB_SLAM3::SemanticFallback;

static std::vector<cv::KeyPoint> makeKeys(int n, float response=1.0f) {
    std::vector<cv::KeyPoint> k;
    for(int i=0;i<n;++i) k.emplace_back(cv::Point2f(float((i%8)*10+2), float((i/8)*10+2)), 1.0f, -1, response + i*0.001f);
    return k;
}

int main() {
    SemanticConfig c;
    c.targetFeatures=6; c.minStaticFeatures=4; c.maxFallbackRatio=0.5f; c.minGridCoverage=0.0f;
    cv::Mat mask(80,80,CV_8UC1,cv::Scalar(0));
    std::vector<cv::KeyPoint> keys=makeKeys(10);
    cv::Mat d(10,32,CV_8U,cv::Scalar(1));
    std::vector<unsigned char> src; std::vector<float> weights;
    SelectionStats s=SelectFeatures(mask,keys,d,c,true,true,src,&weights);
    assert(s.staticCount==0 && s.fallbackCount==3 && keys.size()==3 && src.size()==3);
    c.enableFallback=false; keys=makeKeys(10); d=cv::Mat(10,32,CV_8U,cv::Scalar(1));
    s=SelectFeatures(mask,keys,d,c,true,true,src,&weights); assert(s.fallbackCount==0 && keys.empty());
    c.enableFallback=true; c.minStaticFeatures=1; mask.setTo(cv::Scalar(255)); keys=makeKeys(10); d=cv::Mat(10,32,CV_8U,cv::Scalar(1));
    s=SelectFeatures(mask,keys,d,c,true,false,src,&weights); assert(s.fallbackCount==0 && keys.size()==6);
    SelectionStats low; low.staticCount=1; low.gridCoverage=0.1f; SelectionStats high=low;
    FallbackDecision dec=EvaluateRequest(c,low,high,true); assert(dec.reason==(1|2|4) && dec.requested);
    assert(ShouldRetryForStereo(c,false,0)); assert(!ShouldRetryForStereo(c,true,0));
    SelectionStats ok; ok.staticCount=8; ok.gridCoverage=0.8f;
    FallbackDecision stereo=EvaluateRequest(c,ok,ok,false); assert(stereo.reason==0 && !stereo.requested);
    assert(ShouldRetryForStereo(c,false,c.minStereoMatches-1));
    // Empty masks are treated as fully static and cannot create fallback points.
    cv::Mat emptyMask; keys=makeKeys(8); d=cv::Mat(8,32,CV_8U,cv::Scalar(1));
    s=SelectFeatures(emptyMask,keys,d,c,true,true,src,&weights); assert(s.fallbackCount==0 && src.size()==keys.size());
    // A right-eye selection uses its own source vector; descriptor rows stay aligned.
    cv::Mat rightMask(80,80,CV_8UC1,cv::Scalar(0)); std::vector<cv::KeyPoint> rightKeys=makeKeys(5);
    cv::Mat rightDesc(5,32,CV_8U); for(int r=0;r<5;++r) rightDesc.row(r).setTo(r);
    std::vector<unsigned char> rightSrc; s=SelectFeatures(rightMask,rightKeys,rightDesc,c,true,true,rightSrc);
    assert(rightKeys.size()==rightSrc.size() && rightDesc.rows==static_cast<int>(rightKeys.size()));
    c.enableSoftSelection=true; c.staticFeatureWeight=1.0f; c.fallbackFeatureWeight=0.4f; c.minFeatureWeight=0.1f;
    mask.setTo(cv::Scalar(0)); mask.at<unsigned char>(2,2)=255; keys=makeKeys(2,1.0f); keys[1].response=3.0f; d=cv::Mat(2,32,CV_8U,cv::Scalar(1));
    s=SelectFeatures(mask,keys,d,c,true,true,src,&weights); assert(weights.size()==keys.size());
    std::ofstream log("semantic_fallback_unit_test.log"); log << "PASS\n";
    std::ofstream csv("semantic_fallback_test_summary.csv"); csv << "test,status\nall,PASS\n";
    std::cout << "semantic fallback tests passed\n";
    return 0;
}
