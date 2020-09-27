// Takes input video and mask as input, applies a virtual background using alpha blending

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include <iostream>
#include <vector>
#include <opencv2/photo.hpp>

//calculator SeamlessCloningCalculator
namespace mediapipe {

class SeamlessCloningCalculator : public CalculatorBase {
public:

    SeamlessCloningCalculator() = default;
    ~SeamlessCloningCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  };

  REGISTER_CALCULATOR(SeamlessCloningCalculator);

::mediapipe::Status SeamlessCloningCalculator::GetContract (CalculatorContract *cc){

    cc->Inputs().Tag("IMAGE_CPU").Set<ImageFrame>();
    cc->Inputs().Tag("BACKGROUND_CPU").Set<ImageFrame>();
    cc->Inputs().Tag("MASK_CPU").Set<ImageFrame>();
    cc->Outputs().Tag("OUTPUT_VIDEO").Set<ImageFrame>();

    return ::mediapipe::OkStatus();

}

::mediapipe::Status SeamlessCloningCalculator::Open(CalculatorContext* cc) {

    return ::mediapipe::OkStatus();
}

::mediapipe::Status SeamlessCloningCalculator::Process(CalculatorContext* cc) {

    std::cout << " Process() of SeamlessCloning" << std::endl;

    // Initialize the input image frames as opencv mat
    const auto& input_img = cc->Inputs().Tag("IMAGE_CPU").Get<ImageFrame>();
    cv::Mat input_mat = formats::MatView(&input_img);

    const auto& bgnd_img = cc->Inputs().Tag("BACKGROUND_CPU").Get<ImageFrame>();
    cv::Mat bgnd_mat = formats::MatView(&bgnd_img);

    const auto& mask_img = cc->Inputs().Tag("MASK_CPU").Get<ImageFrame>();
    cv::Mat mask_mat = formats::MatView(&mask_img);

    cv::resize(bgnd_mat, bgnd_mat, input_mat.size());
    cv::cvtColor(mask_mat, mask_mat, cv::COLOR_BGR2GRAY);
    
    // Preprocess the mask before seamless cloning
    cv::threshold(mask_mat, mask_mat, 0, 255, cv::THRESH_BINARY);
    cv::dilate(mask_mat, mask_mat, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
    cv::resize(mask_mat, mask_mat, input_mat.size(),cv::INTER_NEAREST);


    //Find biggets conotur and crop bounding rect
    std::vector< std::vector<cv::Point> > contours;
    findContours(mask_mat, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    auto largest_contour = *std::max_element(contours.begin(),
                           contours.end(),
                           [](std::vector<cv::Point> const& lhs, std::vector<cv::Point> const& rhs)
          {
              return contourArea(lhs, false) < contourArea(rhs, false);
          });

   cv::Rect roi = cv::boundingRect(largest_contour); 

   cv::cvtColor(mask_mat, mask_mat, cv::COLOR_GRAY2BGR);
 
  
   // Crop source and mask regions, using bounding-rect of roi
   input_mat=input_mat(roi);
   mask_mat=mask_mat(roi); 
   mask_mat = mask_mat*255;

   // The location of the center of the src in the dst
   cv::Point center(bgnd_mat.cols-(input_mat.cols/2), bgnd_mat.rows-(input_mat.rows/2));
    
   // Seamlessly clone src into dst and put the results in output
   cv::Mat normal_clone;

   cv::seamlessClone(input_mat, bgnd_mat, mask_mat, center, normal_clone, cv::NORMAL_CLONE);
  
    std::unique_ptr<ImageFrame> output_frame(
        new ImageFrame(input_img.Format(), input_img.Width(), input_img.Height()));

    // Convert and return the output as an image frame
    cv::Mat output_mat = formats::MatView(output_frame.get());

    normal_clone.copyTo(output_mat);
    cc->Outputs().Tag("OUTPUT_VIDEO").Add(output_frame.release(), cc->InputTimestamp());

    return ::mediapipe::OkStatus();

    }

// after defining calculator class, we need to register it with a macro invocation 
// REGISTER_CALCULATOR(calculator_class_name).
REGISTER_CALCULATOR(::mediapipe::SeamlessCloningCalculator);
}
 //end namespace
