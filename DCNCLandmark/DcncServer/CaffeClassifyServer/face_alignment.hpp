//
//  face_alignment.hpp
//  FaceCrop
//
//  Created by TJia on 2017/12/6.
//  Copyright © 2017年 XiangChuanping. All rights reserved.
//

#ifndef face_alignment_hpp
#define face_alignment_hpp

#include <opencv2/opencv.hpp>

namespace FR
{
    enum IoU_TYPE
    {
        IoU_UNION,
        IoU_MIN,
        IoU_MAX
    };

    enum WEIGHT_TYPE
    {
        WEIGHT_LINEAR,
        WEIGHT_GAUSSIAN,
        WEIGHT_ORIGINAL
    };

    const std::vector<cv::Point3f> worldPts =
    {
        cv::Point3f(-0.361,  0.309, -0.521),
        cv::Point3f( 0.235,  0.296, -0.46),
        cv::Point3f(-0.071, -0.023, -0.13),
        cv::Point3f(-0.296, -0.348, -0.472),
        cv::Point3f( 0.156, -0.366, -0.417),
    };

    class CFaceAlignment
    {
        public:
            CFaceAlignment();
            ~CFaceAlignment();
        public:
             cv::Mat FindSimilarityTransform( std::vector<cv::Point2d> source_points,
                                              std::vector<cv::Point2d> target_points,
                                              cv::Mat& Tinv);
            void TransTemplatePoints( std::vector<cv::Point2d> source_points,
                                      int width,
                                      int height,
                                      double width_scale,
                                      double height_scale,
                                      std::vector<cv::Point2d> &target_points);
        private:
            inline std::vector<cv::Point2d> GetVertexFromBox(cv::Rect box);
            inline bool CheckRect(cv::Rect rect, cv::Size image_size);
            inline bool FixRect(cv::Rect& rect, cv::Size image_size, bool only_center = false);
            inline void MakeRectSquare(cv::Rect& rect);

            inline cv::Mat CropImage(cv::Mat& input_image, cv::Rect roi,
                                     cv::Size2d target_size, int flags = 1,
                                     int borderMode = 0,
                                     cv::Scalar borderValue = cv::Scalar(0,0,0));

            inline double IoU(cv::Rect rect, cv::RotatedRect ellip,
                              cv::Size image_size = cv::Size(0,0));

            inline double IoU(cv::Rect rect1, cv::Rect rect2);

            inline cv::Rect BoundingBoxRegressionTarget(cv::Rect data_rect, cv::Rect ground_truth);

            void ModernPosit( cv::Mat &rot, cv::Point3f &trans, std::vector<cv::Point2d> imagePts,
                              std::vector<cv::Point3f> worldPts, double focalLength,
                              cv::Point2d center = cv::Point2d(0.0, 0.0), int maxIterations = 100);

            cv::Point2d Project(cv::Point3f pt, double focalLength, cv::Point2d imgCenter);
            cv::Point3f Transform(cv::Point3f pt, cv::Mat rot, cv::Point3f trans);
            bool CalcCenterScaleAndUp(cv::Mat faceData, std::vector<cv::Point2d> imagePts,
                                      double normEyeDist, cv::Point2d &center, double &scale,
                                      cv::Point2d &upv);

            cv::Rect CalcRect(cv::Mat faceData, std::vector<cv::Point2d> imagePts);

            void NMS(std::vector<std::pair<cv::Rect, float>>& rects, double nms_threshold);
            std::vector<int> Nms_max(   std::vector<std::pair<cv::Rect, float>>& rects,
                                        double overlap, double min_confidence,
                                        IoU_TYPE type = IoU_UNION);
            std::vector<int> Soft_nms_max(  std::vector<std::pair<cv::Rect, float>>& rects,
                                            double overlap, double min_confidence,
                                            IoU_TYPE iou_type = IoU_UNION,
                                            WEIGHT_TYPE weight_type = WEIGHT_LINEAR);
            std::vector<int> Nms_avg(std::vector<std::pair<cv::Rect, float>>& rects, double overlap);

            cv::Mat GetPyramidStitchingImage2(  cv::Mat& input_image,
                                                std::vector<std::pair<cv::Rect, double>>& location_and_scale,
                                                double scaling = 0.707,
                                                cv::Scalar background_color = cv::Scalar(0,0,0),
                                                int min_side = 12, int interval = 2);

            cv::Mat FindNonReflectiveTransform( std::vector<cv::Point2d> source_points,
                                                std::vector<cv::Point2d> target_points,
                                                cv::Mat& Tinv);

    };
}

#endif /* face_alignment_hpp */
