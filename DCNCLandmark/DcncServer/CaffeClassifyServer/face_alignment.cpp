//
//  face_alignment.cpp
//  FaceCrop
//
//  Created by TJia on 2017/12/6.
//  Copyright © 2017年 XiangChuanping. All rights reserved.
//

#include "face_alignment.hpp"

namespace FR
{
    CFaceAlignment::CFaceAlignment()
    {

    }
    CFaceAlignment::~CFaceAlignment()
    {

    }

   cv::Mat CFaceAlignment::FindSimilarityTransform( std::vector<cv::Point2d> source_points,
                                                    std::vector<cv::Point2d> target_points,
                                                    cv::Mat& Tinv)
    {
        cv::Mat Tinv1, Tinv2;
        cv::Mat trans1 = FindNonReflectiveTransform(source_points, target_points, Tinv1);
        std::vector<cv::Point2d> source_point_reflect;
        for (auto sp : source_points)
        //for(int i = 0; i < source_points.size(); i++)
        {
            //cv::Point2d sp = source_points.at(i);
            source_point_reflect.push_back(cv::Point2d(-sp.x, sp.y));
        }
        cv::Mat trans2 = FindNonReflectiveTransform(source_point_reflect, target_points, Tinv2);
        trans2.colRange(0,1) *= -1;
        std::vector<cv::Point2d> trans_points1, trans_points2;
        cv::transform(source_points, trans_points1, trans1);
        cv::transform(source_points, trans_points2, trans2);
        double norm1 = norm(cv::Mat(trans_points1), cv::Mat(target_points), cv::NORM_L2);
        double norm2 = norm(cv::Mat(trans_points2), cv::Mat(target_points), cv::NORM_L2);
        Tinv = norm1 < norm2 ? Tinv1 : Tinv2;
        return norm1 < norm2 ? trans1 : trans2;
  }

  void CFaceAlignment::TransTemplatePoints( std::vector<cv::Point2d> source_points,
                                            int width,
                                            int height,
                                            double width_scale,
                                            double height_scale,
                                            std::vector<cv::Point2d> &target_points)
  {
      double centor_x = (double) width * 0.5;
      double centor_y = (double) height * 0.5;
      target_points.clear();
      cv::Point2d temp;
      for (int i = 0; i < source_points.size(); i++)
      {
          temp.x = (source_points[i].x - centor_x) * (1.0 - width_scale) +  centor_x;
          temp.y = (source_points[i].y - centor_y) * (1.0 - height_scale) +  centor_y;
          target_points.push_back(temp);
      }
  }

  double CFaceAlignment::IoU(cv::Rect rect1, cv::Rect rect2)
  {
        double left     = std::max<double>(rect1.x, rect2.x);
        double top      = std::max<double>(rect1.y, rect2.y);
        double right    = std::min<double>(rect1.x + rect1.width, rect2.x + rect2.width);
        double bottom   = std::min<double>(rect1.y + rect1.height, rect2.y + rect2.height);
        double overlap  = std::max<double>(right - left, 0) * std::max<double>(bottom - top, 0);
        return overlap / (rect1.width * rect1.height + rect2.width * rect2.height - overlap);
  }

  cv::Rect CFaceAlignment::BoundingBoxRegressionTarget(cv::Rect data_rect, cv::Rect ground_truth)
  {
        return cv::Rect((ground_truth.x - data_rect.x) / data_rect.width,
                        (ground_truth.y - data_rect.y) / data_rect.height,
                        log(ground_truth.width / data_rect.width),
                        log(ground_truth.height / data_rect.height));
  }

  void CFaceAlignment::ModernPosit( cv::Mat &rot, cv::Point3f &trans, std::vector<cv::Point2d> imagePts,
                                    std::vector<cv::Point3f> worldPts, double focalLength,
                                    cv::Point2d center, int maxIterations)
 {
    int nbPoints = imagePts.size();

    std::vector<cv::Point2d>::iterator imagePtsIt;
    std::vector<cv::Point2d> centeredImage;

    for (imagePtsIt = imagePts.begin(); imagePtsIt != imagePts.end(); ++imagePtsIt)
    {
        centeredImage.push_back(*imagePtsIt - center);
    }

    for (imagePtsIt = centeredImage.begin(); imagePtsIt != centeredImage.end(); ++imagePtsIt)
    {
        imagePtsIt->x /= focalLength;
        imagePtsIt->y /= focalLength;
    }

    std::vector<double> ui(nbPoints);
    std::vector<double> vi(nbPoints);
    std::vector<double> oldUi(nbPoints);
    std::vector<double> oldVi(nbPoints);
    std::vector<double> deltaUi(nbPoints);
    std::vector<double> deltaVi(nbPoints);

    for (int i = 0; i < nbPoints; ++i)
    {
        ui[i] = centeredImage[i].x;
        vi[i] = centeredImage[i].y;
    }

    cv::Mat homogeneousWorldPts(nbPoints, 4, CV_32F);
    for (int i = 0; i < nbPoints; ++i)
    {
        cv::Point3f worldPoint = worldPts[i];
        homogeneousWorldPts.at<float>(i, 0) = worldPoint.x;
        homogeneousWorldPts.at<float>(i, 1) = worldPoint.y;
        homogeneousWorldPts.at<float>(i, 2) = worldPoint.z;
        homogeneousWorldPts.at<float>(i, 3) = 1; // homogeneous
    }

    cv::Mat objectMat;
    cv::invert(homogeneousWorldPts, objectMat, cv::DECOMP_SVD);

    bool converged = false;
    int iterationCount = 0;

    double Tx = 0.0;
    double Ty = 0.0;
    double Tz = 0.0;
    double r1T[4];
    double r2T[4];
    double r1N[4];
    double r2N[4];
    double r3[4];

    while ((!converged) && ((maxIterations < 0) || (iterationCount < maxIterations)))
    {
        for (int j = 0; j < 4; ++j)
        {
            r1T[j] = 0;
            r2T[j] = 0;
            for (int i = 0; i < nbPoints; ++i)
            {
              r1T[j] += ui[i] * objectMat.at<float>(j, i);
              r2T[j] += vi[i] * objectMat.at<float>(j, i);
            }
        }

        double Tz1, Tz2;
        Tz1 = 1 / sqrt(r1T[0] * r1T[0] + r1T[1] * r1T[1] + r1T[2] * r1T[2]);
        Tz2 = 1 / sqrt(r2T[0] * r2T[0] + r2T[1] * r2T[1] + r2T[2] * r2T[2]);

        Tz = sqrt(Tz1*Tz2);

        for (int j = 0; j < 4; ++j)
        {
            r1N[j] = r1T[j] * Tz;
            r2N[j] = r2T[j] * Tz;
        }

      // DEBUG
      for (int j = 0; j < 3; ++j)
      {
            if ((r1N[j] > 1.0) || (r1N[j] < -1.0))
            {
                r1N[j] = std::max<double>(-1.0, std::min<double>(1.0, r1N[j]));
            }
            if ((r2N[j] > 1.0) || (r2N[j] < -1.0))
            {
                r2N[j] = std::max<double>(-1.0, std::min<double>(1.0, r2N[j]));
            }
      }

      r3[0] = r1N[1] * r2N[2] - r1N[2] * r2N[1];
      r3[1] = r1N[2] * r2N[0] - r1N[0] * r2N[2];
      r3[2] = r1N[0] * r2N[1] - r1N[1] * r2N[0];
      r3[3] = Tz;

      Tx = r1N[3];
      Ty = r2N[3];

      std::vector<double> wi(nbPoints);

      for (int i = 0; i < nbPoints; ++i)
      {
            wi[i] = 0;
            for (int j = 0; j < 4; ++j)
            {
                wi[i] += homogeneousWorldPts.at<float>(i, j) * r3[j] / Tz;
            }
      }

      for (int i = 0; i < nbPoints; ++i)
      {
        oldUi[i] = ui[i];
        oldVi[i] = vi[i];
        ui[i] = wi[i] * centeredImage[i].x;
        vi[i] = wi[i] * centeredImage[i].y;
        deltaUi[i] = ui[i] - oldUi[i];
        deltaVi[i] = vi[i] - oldVi[i];
      }

      double delta = 0.0;
      for (int i = 0; i < nbPoints; ++i)
      {
            delta += deltaUi[i] * deltaUi[i] + deltaVi[i] * deltaVi[i];
      }
      delta = delta*focalLength * focalLength;

      converged = (iterationCount > 0) && (delta < 0.01);
      ++iterationCount;
    }

    trans.x = Tx;
    trans.y = Ty;
    trans.z = Tz;

    rot.create(3, 3, CV_32F);
    for (int i = 0; i < 3; ++i)
    {
      rot.at<float>(0, i) = r1N[i];
      rot.at<float>(1, i) = r2N[i];
      rot.at<float>(2, i) = r3[i];
    }
  }

  cv::Point2d CFaceAlignment::Project(cv::Point3f pt, double focalLength, cv::Point2d imgCenter)
  {
        cv::Point2d res;
        res.x = (pt.x / pt.z) * focalLength + imgCenter.x;
        res.y = (pt.y / pt.z) * focalLength + imgCenter.y;
        return res;
  }

  cv::Point3f CFaceAlignment::Transform(cv::Point3f pt, cv::Mat rot, cv::Point3f trans)
  {
        cv::Point3f res;
        res.x = rot.at<float>(0, 0)*pt.x + rot.at<float>(0, 1)*pt.y + rot.at<float>(0, 2)*pt.z + trans.x;
        res.y = rot.at<float>(1, 0)*pt.x + rot.at<float>(1, 1)*pt.y + rot.at<float>(1, 2)*pt.z + trans.y;
        res.z = rot.at<float>(2, 0)*pt.x + rot.at<float>(2, 1)*pt.y + rot.at<float>(2, 2)*pt.z + trans.z;
        return res;
  }

  bool CFaceAlignment::CalcCenterScaleAndUp(cv::Mat faceData, std::vector<cv::Point2d> imagePts,
                                            double normEyeDist, cv::Point2d &center, double &scale,
                                            cv::Point2d &upv)
  {
        double focalLength = static_cast<double>(faceData.cols)* 1.5;
        cv::Point2d imgCenter = cv::Point2d(static_cast<float>(faceData.cols) / 2.0f, static_cast<float>(faceData.rows) / 2.0f);

        cv::Mat rot;
        cv::Point3f trans;
        ModernPosit(rot, trans, imagePts, worldPts, focalLength, imgCenter);

        cv::Point3f modelCenter = cv::Point3f(-0.056, 0.3, -0.530);
        cv::Point3f rotatedCenter = Transform(modelCenter, rot, trans);
        center = Project(rotatedCenter, focalLength, imgCenter);

        double modelCenterDist = sqrt(rotatedCenter.x*rotatedCenter.x + rotatedCenter.y*rotatedCenter.y + rotatedCenter.z*rotatedCenter.z);
        double cameraModel3dYAngle = atan(rotatedCenter.y / sqrt(rotatedCenter.z*rotatedCenter.z + rotatedCenter.x*rotatedCenter.x));
        double cameraModel3dXAngle = atan(rotatedCenter.x / sqrt(rotatedCenter.z*rotatedCenter.z + rotatedCenter.y*rotatedCenter.y));
        double sphereCenterBorderAngle = asin(0.63 / 2.0 / modelCenterDist);
        double sphereProjTop = tan(cameraModel3dYAngle - sphereCenterBorderAngle) * focalLength;
        double sphereProjBottom = tan(cameraModel3dYAngle + sphereCenterBorderAngle) * focalLength;
        double sphereProjLeft = tan(cameraModel3dXAngle - sphereCenterBorderAngle) * focalLength;
        double sphereProjRight = tan(cameraModel3dXAngle + sphereCenterBorderAngle) * focalLength;

        scale = std::max<double>(abs(sphereProjRight - sphereProjLeft), abs(sphereProjBottom - sphereProjTop)) / normEyeDist;

        cv::Point2d lrV = imagePts[1] - imagePts[0];
        double vlen = sqrt(lrV.x*lrV.x + lrV.y*lrV.y);
        upv.x = lrV.y / vlen;
        upv.y = -lrV.x / vlen;
        return true;
  }

  cv::Rect CFaceAlignment::CalcRect(cv::Mat faceData, std::vector<cv::Point2d> imagePts)
  {
        // normalized rect
        double faceRectWidth = 128;
        double faceRectHeight = 128;
        double normEyeDist = 50.0;              // distance between eyes in normalized rectangle
        cv::Point2d centerOffset(0.0f, 25.0f); // shift from CENTER_BETWEEN_EYES to rectangle center

        cv::Point2d center;
        cv::Point2d upv;
        double scale = 0.0;
        CalcCenterScaleAndUp(faceData, imagePts, normEyeDist, center, scale, upv);

        double w = faceRectWidth*scale;
        double h = faceRectHeight*scale;

        cv::Point2d rectCenter = center;
        rectCenter -= cv::Point2d(upv.x*centerOffset.y*scale, upv.y*centerOffset.y*scale);
        rectCenter -= cv::Point2d(upv.y*centerOffset.x*scale, -upv.x*centerOffset.x*scale);

        double x = rectCenter.x - faceRectWidth*scale / 2;
        double y = rectCenter.y - faceRectHeight*scale / 2;

        return cv::Rect(x, y, w, h);
  }

  static bool strict_weak_ordering(const std::pair<cv::Rect, float> a,
                            const std::pair<cv::Rect, float> b)
  {
        return a.second < b.second;
  }

  void CFaceAlignment::NMS(std::vector<std::pair<cv::Rect, float>>& rects, double nms_threshold)
  {
        std::sort(rects.begin(), rects.end(), strict_weak_ordering);
        int remain = rects.size();
        do
        {
          auto best_rect = rects.end() - 1;
          remain--;
          for (auto rect = best_rect - 1; rect<rects.end(); rect++)
          {
            if (IoU(rect->first, best_rect->first) > nms_threshold)
            {
              rects.erase(rect);
              remain--;
            }
          }
        } while (remain > 0);
  }

  std::vector<int> CFaceAlignment::Nms_max(std::vector<std::pair<cv::Rect, float>>& rects,
                                            double overlap, double min_confidence,
                                            IoU_TYPE type)
  {
        const int n = rects.size();
        std::vector<double> areas(n);

        typedef std::multimap<double, int> ScoreMapper;
        ScoreMapper map;
        for (int i = 0; i < n; i++)
        {
              map.insert(ScoreMapper::value_type(rects[i].second, i));
              areas[i] = rects[i].first.width*rects[i].first.height;
        }

        int picked_n = 0;
        std::vector<int> picked(n);
        while (map.size() != 0)
        {
              auto last_item = map.rbegin();
              int last = map.rbegin()->second; // get the index of maximum score value
              //std::cout << map.rbegin()->first << " " << last << std::endl;
              picked[picked_n] = last;
              picked_n++;

              for (ScoreMapper::iterator it = map.begin(); it != map.end();)
              {
                    int idx = it->second;
                    if (idx == last)
                    {
                          ScoreMapper::iterator tmp = it;
                          tmp++;
                          map.erase(it);
                          it = tmp;
                          continue;
                    }
                    double x1 = std::max<double>(rects[idx].first.x, rects[last].first.x);
                    double x2 = std::min<double>(rects[idx].first.x + rects[idx].first.width, rects[last].first.x + rects[last].first.width);
                    double w = x2 - x1;
                    if (w <= 0)
                    {
                        it++;
                        continue;
                    }
                    double y1 = std::max<double>(rects[idx].first.y, rects[last].first.y);
                    double y2 = std::min<double>(rects[idx].first.y + rects[idx].first.height, rects[last].first.y + rects[last].first.height);
                    double h = y2 - y1;
                    if (h <= 0)
                    {
                        it++;
                        continue;
                    }
                    double ov;
                    switch (type)
                    {
                        case IoU_MAX:
                            ov = w*h / std::max(areas[idx], areas[last]);
                            break;
                        case IoU_MIN:
                            ov = w*h / std::min(areas[idx], areas[last]);
                            break;
                        case IoU_UNION:

                        default:
                            ov = w*h / (areas[idx] + areas[last] - w*h);
                            break;
                    }

                    if (ov > overlap)
                    {
                          ScoreMapper::iterator tmp = it;
                          tmp++;
                          map.erase(it);
                          it = tmp;
                    }
                    else
                    {
                        it++;
                    }
                }
        }

        picked.resize(picked_n);
        return picked;
  }

  std::vector<int> CFaceAlignment::Soft_nms_max(std::vector<std::pair<cv::Rect, float>>& rects,
                                                double overlap, double min_confidence,
                                                IoU_TYPE iou_type,
                                                WEIGHT_TYPE weight_type)
 {
        const int n = rects.size();
        std::vector<double> areas(n);

        typedef std::multimap<double, int> ScoreMapper;
        ScoreMapper map;
        for (int i = 0; i < n; i++)
        {
              map.insert(ScoreMapper::value_type(rects[i].second, i));
              areas[i] = rects[i].first.width*rects[i].first.height;
        }

        int picked_n = 0;
        std::vector<int> picked(n);
        while (map.size() != 0)
        {
              auto last_item = map.rbegin();
              int last = map.rbegin()->second; // get the index of maximum score value
                                               //std::cout << map.rbegin()->first << " " << last << std::endl;
              picked[picked_n] = last;
              picked_n++;

              for (ScoreMapper::iterator it = map.begin(); it != map.end();)
              {
                    int idx = it->second;
                    if (idx == last)
                    {
                      ScoreMapper::iterator tmp = it;
                      tmp++;
                      map.erase(it);
                      it = tmp;
                      continue;
                    }
                    double x1 = std::max<double>(rects[idx].first.x, rects[last].first.x);
                    double x2 = std::min<double>(rects[idx].first.x + rects[idx].first.width, rects[last].first.x + rects[last].first.width);
                    double w = x2 - x1;
                    if (w <= 0)
                    {
                      it++; continue;
                    }
                    double y1 = std::max<double>(rects[idx].first.y, rects[last].first.y);
                    double y2 = std::min<double>(rects[idx].first.y + rects[idx].first.height, rects[last].first.y + rects[last].first.height);
                    double h = y2 - y1;
                    if (h <= 0)
                    {
                        it++;
                        continue;
                    }
                    double ov;
                    switch (iou_type)
                    {
                        case IoU_MAX:
                          ov = w*h / std::max(areas[idx], areas[last]);
                          break;
                        case IoU_MIN:
                          ov = w*h / std::min(areas[idx], areas[last]);
                          break;
                        case IoU_UNION:
                        default:
                          ov = w*h / (areas[idx] + areas[last] - w*h);
                          break;
                    }

                    double weight = 1.0;

                    switch (weight_type)
                    {
                        case WEIGHT_LINEAR:
                              if (ov > overlap)
                              {
                                    weight = 1.0 - ov;
                              }
                              else
                              {
                                    weight = 1.0;
                              }
                              break;

                        case WEIGHT_GAUSSIAN:
                              weight = exp((-ov*ov) / 0.5);
                              break;
                        case WEIGHT_ORIGINAL:
                              if (ov > overlap)
                              {
                                weight = 0.0;
                              }
                              else
                              {
                                weight = 1.0;
                              }
                              break;
                        default:
                            break;

                    }

                    rects[idx].second *= weight;

                    if (rects[idx].second < min_confidence)
                    {
                          ScoreMapper::iterator tmp = it;
                          tmp++;
                          map.erase(it);
                          it = tmp;
                    }
                    else
                    {
                        it++;
                    }
                }
        }

        picked.resize(picked_n);
        return picked;
  }

  std::vector<int> CFaceAlignment::Nms_avg(std::vector<std::pair<cv::Rect, float>>& rects, double overlap)
  {
        const int n = rects.size();
        std::vector<double> areas(n);

        typedef std::multimap<double, int> ScoreMapper;
        ScoreMapper map;
        for (int i = 0; i < n; i++)
        {
              map.insert(ScoreMapper::value_type(rects[i].second, i));
              areas[i] = rects[i].first.width*rects[i].first.height;
        }

        int picked_n = 0;
        std::vector<int> picked(n);
        while (map.size() != 0)
        {
              int last = map.rbegin()->second; // get the index of maximum score value
              picked[picked_n] = last;
              picked_n++;

              int overlap_count = 1;
              double mean_x = rects[last].first.x, mean_y = rects[last].first.y, mean_w = log(rects[last].first.width), mean_h = log(rects[last].first.height);
              //double mean_x = rects[last].first.x, mean_y = rects[last].first.y, mean_w = rects[last].first.width, mean_h = rects[last].first.height;

              for (ScoreMapper::iterator it = map.begin(); it != map.end();)
              {
                    int idx = it->second;
                    double x1 = std::max<double>(rects[idx].first.x, rects[last].first.x);
                    double y1 = std::max<double>(rects[idx].first.y, rects[last].first.y);
                    double x2 = std::min<double>(rects[idx].first.x + rects[idx].first.width, rects[last].first.x + rects[last].first.width);
                    double y2 = std::min<double>(rects[idx].first.y + rects[idx].first.height, rects[last].first.y + rects[last].first.height);
                    double w = std::max<double>(0., x2 - x1);
                    double h = std::max<double>(0., y2 - y1);
                    double ov = w*h / (areas[idx] + areas[last] - w*h);
                    if (ov > overlap)
                    {
                          ScoreMapper::iterator tmp = it;
                          tmp++;
                          map.erase(it);
                          it = tmp;
                          if (rects[idx].second > rects[last].second * 0.9)
                          {
                            mean_x += rects[idx].first.x;
                            mean_y += rects[idx].first.y;
                            mean_w += log(rects[idx].first.width);
                            mean_h += log(rects[idx].first.height);
                            //mean_w += rects[idx].first.width;
                            //mean_h += rects[idx].first.height;
                            overlap_count++;
                          }
                    }
                    else
                    {
                        it++;
                    }
             }
             rects[last].first.x = mean_x / overlap_count;
             rects[last].first.y = mean_y / overlap_count;
             rects[last].first.width = exp(mean_w / overlap_count);
             rects[last].first.height = exp(mean_h / overlap_count);
        }

        picked.resize(picked_n);
        return picked;
  }

  bool CFaceAlignment::CheckRect(cv::Rect rect, cv::Size image_size)
  {
        if (rect.x < 0 || rect.y < 0 || rect.width <= 0 || rect.height <= 0 ||
            (rect.x + rect.width) > (image_size.width - 1) || (rect.y + rect.height) > (image_size.height - 1))
        {
            return false;
        }
        else
        {
            return true;
        }
  }
  inline bool CFaceAlignment::FixRect(cv::Rect& rect, cv::Size image_size, bool only_center)
  {
        if (rect.width <= 0 || rect.height <= 0)
        {
            return false;
        }

        if (only_center)
        {
              cv::Point2d center = cv::Point2d(rect.x + rect.width / 2, rect.y + rect.height / 2);
              center.x = std::max<double>(center.x, 0.0);
              center.y = std::max<double>(center.y, 0.0);
              center.x = std::min<double>(center.x, image_size.width - 1);
              center.y = std::min<double>(center.y, image_size.height - 1);
              rect.x = center.x - rect.width / 2;
              rect.y = center.y - rect.height / 2;
        }
        else
        {
              rect.x = std::max<double>(rect.x, 0.0);
              rect.y = std::max<double>(rect.y, 0.0);
              rect.width = std::min<double>(rect.width, image_size.width - 1 - rect.x);
              rect.height = std::min<double>(rect.height, image_size.height - 1 - rect.y);
        }
        return true;
  }

  inline void CFaceAlignment::MakeRectSquare(cv::Rect& rect)
  {
        double max_len = std::max(rect.width, rect.height);
        rect.x += rect.width / 2 - max_len / 2;
        rect.y += rect.height / 2 - max_len / 2;
        rect.width = rect.height = max_len;
  }

  double CFaceAlignment::IoU(cv::Rect rect, cv::RotatedRect ellip, cv::Size image_size)
  {
        cv::Rect baseRect,ellip_br = ellip.boundingRect();
        baseRect.x = std::min(rect.x, ellip_br.x);
        baseRect.y = std::min(rect.y, ellip_br.y);
        baseRect.width = std::max(rect.x + rect.width - baseRect.x, ellip_br.x + ellip_br.width - baseRect.x);
        baseRect.height = std::max(rect.y + rect.height - baseRect.y, ellip_br.y + ellip_br.height - baseRect.y);
        baseRect.x -= 10;
        baseRect.y -= 10;
        baseRect.width += 20;
        baseRect.height += 20;
        if (image_size.width != 0)
        {
            FixRect(baseRect,image_size);
        }
        rect.x -= baseRect.x; rect.y -= baseRect.y;
        ellip.center.x -= baseRect.x; ellip.center.y -= baseRect.y;
        cv::Mat rect_image = cv::Mat::zeros(cv::Size(baseRect.width, baseRect.height), CV_8UC1);
        cv::Mat ellipse_image = cv::Mat::zeros(cv::Size(baseRect.width, baseRect.height), CV_8UC1);
        rectangle(rect_image, rect, cv::Scalar(255), -1);
        ellipse(ellipse_image, ellip, cv::Scalar(255), -1);
        cv::Mat Overlap = rect_image.mul(ellipse_image);

        double rect_size = countNonZero(rect_image);//rect.width * rect.height;
        double ellipse_size = countNonZero(ellipse_image);
        double overlap = countNonZero(Overlap);
        return overlap / (rect_size + ellipse_size - overlap);
  }

  cv::Mat CFaceAlignment::CropImage(cv::Mat& input_image, cv::Rect roi, cv::Size2d target_size, int flags,
                                    int borderMode, cv::Scalar borderValue)
  {
        cv::Mat M = (cv::Mat_<float>(2, 3) << target_size.width / roi.width, 0, -roi.x*target_size.width / roi.width, 0, target_size.height / roi.height, -roi.y*target_size.height / roi.height);
        cv::Mat result;
        cv::warpAffine(input_image, result, M, target_size, flags, borderMode, borderValue);
        return result;
  }

  //Only support 2 scaling
  cv::Mat CFaceAlignment::GetPyramidStitchingImage2(cv::Mat& input_image,
                                std::vector<std::pair<cv::Rect, double>>& location_and_scale,
                                double scaling,cv::Scalar background_color,int min_side, int interval)
  {
        using namespace std;
        bool stitch_x = input_image.cols < input_image.rows;
        cv::Size current_size = input_image.size();
        cv::Point left_top = cv::Point(0, 0);
        int width, height;
        if (stitch_x)
        {
              width = ceil(input_image.cols * (1 + scaling)) + interval * 2;
              height = 0;
              map<int, pair<int, int>> height_index; // (width_start, width, height)
              height_index[0] = pair<int, int>(width, 0);
              do
              {
                    int min_h = INT_MAX, min_start = 0;
                    for (auto h : height_index)
                    {
                          if (h.second.first > current_size.width + interval && h.second.second < min_h)
                          {
                                min_h =  h.second.second;
                                min_start = h.first;
                          }
                    }

                    location_and_scale.push_back(make_pair(cv::Rect(min_start, height_index[min_start].second, current_size.width, current_size.height),
                                                            (double)current_size.height / (double)input_image.rows));

                    height_index[min_start + current_size.width + interval] = height_index[min_start];
                    height_index[min_start + current_size.width + interval].first -= current_size.width + interval;
                    height_index[min_start].first = current_size.width;
                    height_index[min_start].second += current_size.height + interval;
                    if (height_index[min_start].second > height)
                    {
                        height = height_index[min_start].second;
                    }

                    current_size.width *= scaling;
                    current_size.height *= scaling;
                } while (current_size.width > min_side);
                height += interval;
        }
        else
        {
              height = ceil(input_image.rows * (1 + scaling)) + interval * 2;
              width = 0;
              map<int, pair<int, int>> width_index; // (height_start, height, width)
              width_index[0] = pair<int, int>(height, 0);
              do
              {
                    int min_w = INT_MAX, min_start = 0;
                    for (auto w : width_index)
                    {
                          if (w.second.first > current_size.height + interval && w.second.second < min_w)
                          {
                                min_w = w.second.second;
                                min_start = w.first;
                          }
                    }

                    location_and_scale.push_back(make_pair(cv::Rect(width_index[min_start].second, min_start, current_size.width, current_size.height),
                                                (double)current_size.width / (double)input_image.cols));
                    width_index[min_start + current_size.height + interval] = width_index[min_start];
                    width_index[min_start + current_size.height + interval].first -= current_size.height + interval;
                    width_index[min_start].first = current_size.height;
                    width_index[min_start].second += current_size.width + interval;
                    if (width_index[min_start].second > width)
                    {
                        width = width_index[min_start].second;
                    }

                    current_size.width *= scaling;
                    current_size.height *= scaling;
                } while (current_size.height > min_side);

                width += interval;
        }

        cv::Mat big_image = cv::Mat::zeros(height, width, input_image.type());
        big_image = background_color;
        cv::Mat resized_image = input_image;

        for (auto ls : location_and_scale)
        {
              cv::resize(resized_image, resized_image, cv::Size(ls.first.width, ls.first.height));
              resized_image.copyTo(big_image(ls.first));
        }
        return big_image;
  }

  cv::Mat CFaceAlignment::FindNonReflectiveTransform(std::vector<cv::Point2d> source_points,
                                                    std::vector<cv::Point2d> target_points,
                                                    cv::Mat& Tinv)
  {
        assert(source_points.size() == target_points.size());
        assert(source_points.size() >= 2);
        cv::Mat U = cv::Mat::zeros(target_points.size() * 2, 1, CV_64F);
        cv::Mat X = cv::Mat::zeros(source_points.size() * 2, 4, CV_64F);
        for (int i = 0; i < target_points.size(); i++)
        {
              U.at<double>(i * 2, 0) = source_points[i].x;
              U.at<double>(i * 2 + 1, 0) = source_points[i].y;
              X.at<double>(i * 2, 0) = target_points[i].x;
              X.at<double>(i * 2, 1) = target_points[i].y;
              X.at<double>(i * 2, 2) = 1;
              X.at<double>(i * 2, 3) = 0;
              X.at<double>(i * 2 + 1, 0) = target_points[i].y;
              X.at<double>(i * 2 + 1, 1) = -target_points[i].x;
              X.at<double>(i * 2 + 1, 2) = 0;
              X.at<double>(i * 2 + 1, 3) = 1;
        }

        cv::Mat r = X.inv(cv::DECOMP_SVD)*U;
        Tinv = (cv::Mat_<double>(3, 3) << r.at<double>(0), -r.at<double>(1), 0,
                         r.at<double>(1), r.at<double>(0), 0,
                         r.at<double>(2), r.at<double>(3), 1);
        cv::Mat T = Tinv.inv(cv::DECOMP_SVD);
        Tinv = Tinv(cv::Rect(0, 0, 2, 3)).t();
        return T(cv::Rect(0,0,2,3)).t();
  }

  std::vector<cv::Point2d> CFaceAlignment::GetVertexFromBox(cv::Rect box)
  {
        return{ cv::Point2d(box.x, box.y), cv::Point2d(box.x + box.width, box.y), cv::Point2d(box.x + box.width, box.y + box.height), cv::Point2d(box.x, box.y + box.height) };
  }
}
