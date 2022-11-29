#ifndef DETECTION_ENGINE_
#define DETECTION_ENGINE_

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>

#include "opencv2/opencv.hpp"

#include "../inference_helper/inference_helper.h"
#include "../inference_helper/inference_helper_tensorrt.h"

#include "../common_helper/common_helper.h"
#include "../common_helper/common_helper_cv.h"
#include "../common_helper/bounding_box.h"

class DetectionEngine
{
public:
    DetectionEngine()
    {
        threshold_box_confidence_ = 0.2f;
        threshold_class_confidence_ = 0.2f;
        threshold_nms_iou_ = 0.6f;
    }
    ~DetectionEngine() {}

    enum
    {
        kRetOk = 0,
        kRetErr = -1
    };

    typedef struct Result_
    {
        std::vector<BoundingBox> bbox_list;
        struct crop_
        {
            int32_t x;
            int32_t y;
            int32_t w;
            int32_t h;
            crop_() : x(0), y(0), w(0), h(0) {}
        } crop;
        double time_pre_process;  // [msec]
        double time_inference;    // [msec]
        double time_post_process; // [msec]
        Result_() : time_pre_process(0), time_inference(0), time_post_process(0)
        {
        }
    } Result;

    void SetThreshold(float threshold_box_confidence, float threshold_class_confidence, float threshold_nms_iou)
    {
        threshold_box_confidence_ = threshold_box_confidence;
        threshold_class_confidence_ = threshold_class_confidence;
        threshold_nms_iou_ = threshold_nms_iou;
    }

    int32_t Initialize(const std::string &work_dir, const int32_t num_threads);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat &original_mat, Result &result);

private:
    float threshold_box_confidence_;
    float threshold_class_confidence_;
    float threshold_nms_iou_;

    int32_t ReadLabel(const std::string &filename, std::vector<std::string> &label_list);
    void GetBoundingBox(const float *data, int32_t anchor_box_num, float scale_x, float scale_y, std::vector<BoundingBox> &bbox_list);

    std::unique_ptr<InferenceHelper> inference_helper_;
    std::vector<InputTensorInfo> input_tensor_info_list_;
    std::vector<OutputTensorInfo> output_tensor_info_list_;
    std::vector<std::string> label_list_;
};

#endif