#include "detection_engine.h"
#include "iostream"

using namespace std;

#define TAG "DetectionEngine"
#define PRINT(...) COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

#define MODEL_NAME "yolov7-tiny_384x640.onnx"
#define INPUT_DIMS     \
    {                  \
        1, 3, 384, 640 \
    }

#define TENSORTYPE TensorInfo::kTensorTypeFp32
#define INPUT_NAME "images"
#define IS_NCHW true
#define IS_RGB true
#define OUTPUT_NAME "output"

static constexpr int32_t kNumberOfClass = 80;
static constexpr int32_t kElementNumOfAnchor = kNumberOfClass + 5; // x, y, w, h, bbox confidence, [class confidence]

#define LABEL_NAME "label_coco_80.txt"

int32_t DetectionEngine::Initialize(const std::string &work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + MODEL_NAME;
    std::string labelFilename = work_dir + LABEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    /* normalize to [0.0, 1.0] */
    input_tensor_info.normalize.mean[0] = 0.0f;
    input_tensor_info.normalize.mean[1] = 0.0f;
    input_tensor_info.normalize.mean[2] = 0.0f;
    input_tensor_info.normalize.norm[0] = 1.0f;
    input_tensor_info.normalize.norm[1] = 1.0f;
    input_tensor_info.normalize.norm[2] = 1.0f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME, TENSORTYPE));

    /* Create and Initialize Inference Helper */
    // inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOnnxRuntime));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorrt));

    if (!inference_helper_)
    {
        return kRetErr;
    }
    InferenceHelperTensorRt *p = dynamic_cast<InferenceHelperTensorRt *>(inference_helper_.get());
    if (p)
        p->SetDlaCore(-1); /* Use GPU */
    if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk)
    {
        inference_helper_.reset();
        return kRetErr;
    }

    if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk)
    {
        inference_helper_.reset();
        return kRetErr;
    }

    /* read label */
    if (ReadLabel(labelFilename, label_list_) != kRetOk)
    {
        return kRetErr;
    }

    return kRetOk;
}

int32_t DetectionEngine::Finalize()
{
    if (!inference_helper_)
    {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}

void DetectionEngine::GetBoundingBox(const float *data, int32_t anchor_box_num, float scale_x, float scale_y, std::vector<BoundingBox> &bbox_list)
{
    int32_t index = 0;
    for (int32_t i = 0; i < anchor_box_num; i++)
    {
        float box_confidence = data[index + 4];
        if (box_confidence >= threshold_box_confidence_)
        {
            int32_t class_id = 0;
            float confidence = 0;
            for (int32_t class_index = 0; class_index < kNumberOfClass; class_index++)
            {
                float confidence_of_class = data[index + 5 + class_index];
                if (confidence_of_class > confidence)
                {
                    confidence = confidence_of_class;
                    class_id = class_index;
                }
            }

            if (confidence >= threshold_class_confidence_)
            {
                int32_t cx = static_cast<int32_t>(data[index + 0] * scale_x);
                int32_t cy = static_cast<int32_t>(data[index + 1] * scale_y);
                int32_t w = static_cast<int32_t>(data[index + 2] * scale_x);
                int32_t h = static_cast<int32_t>(data[index + 3] * scale_y);
                int32_t x = cx - w / 2;
                int32_t y = cy - h / 2;
                bbox_list.push_back(BoundingBox(class_id, "", confidence, x, y, w, h));
            }
        }
        index += kElementNumOfAnchor;
    }
}

int32_t DetectionEngine::Process(const cv::Mat &original_mat, Result &result)
{
    if (!inference_helper_)
    {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }

    // Preprocess
    const auto &t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo &input_tensor_info = input_tensor_info_list_[0];

    /* do crop, resize and color conversion here because some inference engine doesn't support these operations */
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);

    input_tensor_info.data = img_src.data;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.image_info.width = img_src.cols;
    input_tensor_info.image_info.height = img_src.rows;
    input_tensor_info.image_info.channel = img_src.channels();
    input_tensor_info.image_info.crop_x = 0;
    input_tensor_info.image_info.crop_y = 0;
    input_tensor_info.image_info.crop_width = img_src.cols;
    input_tensor_info.image_info.crop_height = img_src.rows;
    input_tensor_info.image_info.is_bgr = false;
    input_tensor_info.image_info.swap_color = false;

    if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk)
    {
        return kRetErr;
    }
    const auto &t_pre_process1 = std::chrono::steady_clock::now();

    // Inference
    const auto &t_inference0 = std::chrono::steady_clock::now();
    if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk)
    {
        return kRetErr;
    }
    const auto &t_inference1 = std::chrono::steady_clock::now();

    // PostProcess
    const auto &t_post_process0 = std::chrono::steady_clock::now();

    // Get boundig box
    std::vector<BoundingBox> bbox_list;
    float *output_data = output_tensor_info_list_[0].GetDataAsFloat();
    int32_t anchor_box_num = output_tensor_info_list_[0].tensor_dims[1];
    float scale_x = static_cast<float>(crop_w) / input_tensor_info.GetWidth(); /* scale to original image */
    float scale_y = static_cast<float>(crop_h) / input_tensor_info.GetHeight();
    GetBoundingBox(output_data, anchor_box_num, scale_x, scale_y, bbox_list);

    // Adjust bounding box
    for (auto &bbox : bbox_list)
    {
        bbox.x += crop_x;
        bbox.y += crop_y;
        bbox.label = label_list_[bbox.class_id];
    }

    // NMS
    std::vector<BoundingBox> bbox_nms_list;
    BoundingBoxUtils::Nms(bbox_list, bbox_nms_list, threshold_nms_iou_);

    const auto &t_post_process1 = std::chrono::steady_clock::now();

    result.bbox_list = bbox_nms_list;
    result.crop.x = (std::max)(0, crop_x);
    result.crop.y = (std::max)(0, crop_y);
    result.crop.w = (std::min)(crop_w, original_mat.cols - result.crop.x);
    result.crop.h = (std::min)(crop_h, original_mat.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;

    return kRetOk;
}

int32_t DetectionEngine::ReadLabel(const std::string &filename, std::vector<std::string> &label_list)
{
    std::ifstream ifs(filename);
    if (ifs.fail())
    {
        PRINT_E("Failed to read %s\n", filename.c_str());
        return kRetErr;
    }
    label_list.clear();
    std::string str;
    while (getline(ifs, str))
    {
        label_list.push_back(str);
    }
    return kRetOk;
}