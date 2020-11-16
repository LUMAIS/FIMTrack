/*****************************************************************************
 * Copyright (c) 2011-2016 The FIMTrack Team as listed in CREDITS.txt        *
 * http://fim.uni-muenster.de                                             	 *
 *                                                                           *
 * This file is part of FIMTrack.                                            *
 * FIMTrack is available under multiple licenses.                            *
 * The different licenses are subject to terms and condition as provided     *
 * in the files specifying the license. See "LICENSE.txt" for details        *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 * FIMTrack is free software: you can redistribute it and/or modify          *
 * it under the terms of the GNU General Public License as published by      *
 * the Free Software Foundation, either version 3 of the License, or         *
 * (at your option) any later version. See "LICENSE-gpl.txt" for details.    *
 *                                                                           *
 * FIMTrack is distributed in the hope that it will be useful,               *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of            *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the              *
 * GNU General Public License for more details.                              *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 * For non-commercial academic use see the license specified in the file     *
 * "LICENSE-academic.txt".                                                   *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 * If you are interested in other licensing models, including a commercial-  *
 * license, please contact the author at fim@uni-muenster.de      			 *
 *                                                                           *
 *****************************************************************************/

#include <array>
#include <algorithm>
#include "Preprocessor.hpp"

using namespace cv;
using std::vector;

Preprocessor::Preprocessor()
{
}

void Preprocessor::graythresh(Mat const & src,
                              int const thresh,
                              Mat & dst)
{
    threshold(src, dst, thresh, 255.0, THRESH_BINARY);
}

void Preprocessor::calcContours(Mat const & src, contours_t & contours)
{
    findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);  // cv::CHAIN_APPROX_TC89_L1 or cv::CHAIN_APPROX_SIMPLE for the approximate compressed contours; CHAIN_APPROX_NONE to retain all points as they are
}

void Preprocessor::sizethreshold(const contours_t &contoursSrc, const int minSizeThresh, const int maxSizeThresh, contours_t &correctContoursDst, contours_t &biggerContoursDst)
{
    correctContoursDst.clear();
    biggerContoursDst.clear();
    
    // iterate over all contours
    for(auto const& c : contoursSrc)
    {
        // calculate the current size of the contour area
        double current_size = cv::contourArea(c);
        
        // check the size (maxSizeThresh > current_size > minSizeThresh)
        if(current_size <= maxSizeThresh && current_size > minSizeThresh)
        {
            correctContoursDst.push_back(c);
        }
        else if(current_size > maxSizeThresh)
        {
            biggerContoursDst.push_back(c);
        }
    }
}

void Preprocessor::borderRestriction(contours_t &contours, const Mat& img, bool checkRoiBorders)
{
    contours_t validContours;
    validContours.reserve(contours.size());
    
    for(auto const& contour : contours)
    {
        cv::Rect rect = cv::boundingRect(contour);
        
        // necessity of extending the rect by 2 pixels, because findContours returns
        // contours that have at least 1 pixel distance to image-borders
        int x1 = rect.x - 2;
        int y1 = rect.y - 2;
        int x2 = rect.x + rect.width + 2;
        int y2 = rect.y + rect.height + 2;
        
        // at first: check, if bounding-rect outside of the images range
        if(x1 < 0 || y1 < 0 || x2 >= img.cols || y2 >= img.rows)
        {
            continue;
        }
        
        bool valid = true;
        if(checkRoiBorders)
        {
            // at second: check, if convex-hull not within ROI
            FIMTypes::contour_t convexHull;
            cv::convexHull(contour, convexHull);
            
            // for every point of the convex hull ...
            foreach(cv::Point p, convexHull)
            {
                // check, if at least one of the four neighbours lies outside the ROI
                if(img.at<uchar>(p.y-1,p.x-1) == 0
                        || img.at<uchar>(p.y+1,p.x-1) == 0
                        || img.at<uchar>(p.y-1,p.x+1) == 0
                        || img.at<uchar>(p.y+1,p.x+1) == 0)
                {
                    valid = false;
                    break;
                }
            }
        }
        
        // at this point, check if contour is valid
        if(valid)
        {
            validContours.push_back(contour);
        }
    }
    contours = validContours;
}

void Preprocessor::estimateThresholds(int& grayThresh, int& minSizeThresh, int& maxSizeThresh, Rect& foreground,
                                      const Mat& img, const dlc::Larvae& larvae, const dlc::MatchStat& matchStat)
{
    if(larvae.empty())
        return;
    dlc::Larva::Points  hull;
    //hull.reserve(larvae[0].points.size());
    vector<float>  areas;
    areas.reserve(larvae.size());
    std::array<unsigned, 255>  larvaHist = {0};
    std::array<unsigned, 255>  bgHist = {0};

    // Identify an approximate foregroud ROI
    ////const auto  lp0 = larvae[0].points[0];
    //// Note: take width and height to 1 pixel to properly form the rect
    Rect  fgrect = {img.cols, img.rows, 0, 0};  // 1, 1
    for(const auto& lv: larvae)
        for(const auto& pt: lv.points) {
            if(pt.x < fgrect.x)
                fgrect.x = pt.x;
            else if(pt.x > fgrect.width)
                fgrect.width = pt.x;

            if(pt.y < fgrect.y)
                fgrect.y = pt.y;
            else if(pt.y > fgrect.height)
                fgrect.height = pt.y;
        }
    //grabCut(InputArray img, InputOutputArray mask, Rect rect,
    //       InputOutputArray bgdModel, InputOutputArray fgdModel,
    //       int iterCount, int mode = GC_EVAL);
    unsigned  grayTheshGlob = 0;
    if(fgrect.width && fgrect.height) {
        // Expand the foreground ROI with the statistical span
        const int  span = matchStat.distAvg + 2 * matchStat.distStd;  // Note: we expend from the border points rather thatn from the center => *2 rather than *3
        fgrect.x -= span;
        if(fgrect.x < 0)
            fgrect.x = 0;
        fgrect.y -= span;
        if(fgrect.y < 0)
            fgrect.y = 0;
        fgrect.width += span;
        if(fgrect.x + fgrect.width >= img.cols)
            fgrect.width = img.cols - fgrect.x;
        fgrect.height += span;
        if(fgrect.y + fgrect.height >= img.rows)
            fgrect.height = img.rows - fgrect.y;
        foreground = fgrect;
        //printf("%s> fgrect: %d + %d, %d + %d\n", __FUNCTION__, fgrect.x, fgrect.y, fgrect.width, fgrect.height);

        Mat mask = Mat::zeros(img.size(), CV_8UC1);  // resulting mask
        Mat bgdModel, fgdModel;
        Mat imgClr;
        cv::cvtColor(img, imgClr, cv::COLOR_GRAY2BGR);  // CV_8UC3
        cv::grabCut(imgClr, mask, fgrect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
        cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);  // Retain only the foreground
        Mat imgFg;  // Foreground image
        img.copyTo(imgFg, mask);
        cv::imshow("Foreground", imgFg);

        // Evaluate brightness
        const int  xEnd = fgrect.x + fgrect.width;
        const int  yEnd = fgrect.y + fgrect.height;
        //fprintf(stderr, "%s> Brightness ROI (%d, %d; %d, %d) content:\n", __FUNCTION__, brect.x, brect.y, brect.width, brect.height);
        for(int y = fgrect.y; y < yEnd; ++y) {
            for(int x = fgrect.x; x < xEnd; ++x) {
                //fprintf(stderr, "%u ", img.at<uchar>(y, x));
                uint8_t  bval = img.at<uint8_t>(y, x);
                // Omit zero mask
                if(bval)
                    ++larvaHist[bval];  // brightness
            }
            //fprintf(stderr, "\n");
        }
        // Calculate the number of values in foreground
        int32_t count = 0;
        for(auto num: larvaHist)
            count += num;
        // Cut foreground to ~<70% of the brightest values considering that the hull includes some background.
        // Note that the convex hull cause inclusion of some background pixel in the larva area.
        count *= 0.5f;  // 0.04 .. 0.08
        unsigned  ifgmin = 0;
        while(count > 0)
            count -= larvaHist[ifgmin++];
        grayTheshGlob = ifgmin;
        //printf("%s> grayThreshGlobal: %d\n", __FUNCTION__, ifgmin);
        // Reser larvaHist
        larvaHist = {0};
    }

//    let color = new cv.Scalar(0, 0, 255);
//    let point1 = new cv.Point(rect.x, rect.y);
//    let point2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
//    cv.rectangle(src, point1, point2, color);
//    cv.imshow('canvasOutput', src);
//    src.delete(); mask.delete(); bgdModel.delete(); fgdModel.delete();

    for(const auto& lv: larvae) {
        // Note: OpenCV expects points to be ordered in contours, so convexHull() is used
        cv::convexHull(lv.points, hull);
        areas.push_back(cv::contourArea(hull));

        // Evaluate brightness
        const Rect  brect = boundingRect(hull);
        const int  xEnd = brect.x + brect.width;
        const int  yEnd = brect.y + brect.height;
        //fprintf(stderr, "%s> Brightness ROI (%d, %d; %d, %d) content:\n", __FUNCTION__, brect.x, brect.y, brect.width, brect.height);
        for(int y = brect.y; y < yEnd; ++y) {
            for(int x = brect.x; x < xEnd; ++x) {
                //fprintf(stderr, "%u ", img.at<uchar>(y, x));
                uint8_t  bval = img.at<uint8_t>(y, x);
                if(pointPolygonTest(hull, Point2f(x, y), false) >= 0)
                    ++larvaHist[bval];  // brightness
                else ++bgHist[bval];
            }
            //fprintf(stderr, "\n");
        }
        //fprintf(stderr, "area: %f, brightness: %u\n", area, bval);

        hull.clear();
    }
    cv::Scalar mean, stddev;
    cv::meanStdDev(areas, mean, stddev);
    std::sort(areas.begin(), areas.end());
    //const folat  rstd = 4;  // (3 .. 5+);  Note: 3 convers ~ 96% of results, but we should extend those margins
    //const int  minSizeThreshLim = 0.56f * areas[0]
    const float  expPerimMin = matchStat.distAvg - 0.36f * matchStat.distStd;  // 0.3 .. 0.4
    minSizeThresh = (max<int>(min<int>(mean[0] - 4 * stddev[0], 0.56f * areas[0]), areas[0] * 0.36f) + expPerimMin * expPerimMin) / 2;  // 0.56 area ~= 0.75 perimiter; 0.36 a ~= 0.6 p
    const float  expPerimMax = matchStat.distAvg + 3.8f * matchStat.distStd;  // 3.6 .. 3.8
    maxSizeThresh = (min<int>(max<int>(mean[0] + 5 * stddev[0], 2.5f * areas[areas.size() - 1]), areas[areas.size() - 1] * 3.24f) + expPerimMax * expPerimMax) / 2;  // 2.5 area ~= 1.58 perimiter; 3.24 a ~= 1.8 p
    //printf("%s> minSizeThresh: %d (meanSdev: %d, areaMinLim: %u)\n", __FUNCTION__
    //    , minSizeThresh, int(mean[0] - 4.f * stddev[0]), unsigned(0.56f * areas[0]));
    //printf("%s> maxSizeThresh: %d (meanSdev: %d, areaMaxLim: %u)\n", __FUNCTION__
    //    , maxSizeThresh, int(mean[0] + 5.f * stddev[0]), unsigned(2.5f * areas[areas.size() - 1]));

    // Calculate the number of values in foreground
    int32_t count = 0;
    for(auto num: larvaHist)
        count += num;
    // Cut foreground to ~<96% of the brightest values considering that the hull includes some background.
    // Note that the convex hull cause inclusion of some background pixel in the larva area.
    count *= 0.12f;  // 0.04 .. 0.08 [0.16]
    unsigned  ifgmin = 0;
    while(count > 0)
        count -= larvaHist[ifgmin++];
    // Evaluate the number of background values that are smaller that foreground ones
    count = 0;
    unsigned  ibg = 0;
    for(;ibg < ifgmin; ++ibg)
        count += bgHist[ibg];
    // Take >=96% of the background values that are lower than foreground ones
    count *= 0.96f;
    for(ibg = 0; count > 0 && ibg < ifgmin; ++ibg)
        count -= bgHist[ibg];
    // Take average index of background an foreground to identify the thresholding brightness
    //grayThresh = max<int>((ibg + ifgmin) / 2, round(ifgmin * 0.96f));  // 0.75 .. 0.96
    //grayThresh = max<float>((ibg + ifgmin) * 0.5f, ifgmin * 0.96f);  // 0.75 .. 0.96
    //grayThresh = (max(ibg, grayTheshGlob) * 3 + grayTheshGlob) / 4;
    grayThresh = (ibg * 3 + grayTheshGlob) / 4;
    printf("%s> grayThresh: %d (bgEst: %u, avg: %u, xFgMin: %u; grayTheshGlob: %u)\n", __FUNCTION__, grayThresh, ibg, (ibg + ifgmin) / 2, unsigned(round(ifgmin * 0.96f)), grayTheshGlob);
}

void Preprocessor::preprocessPreview(const Mat &src,
                                     contours_t &acceptedContoursDst,
                                     contours_t &biggerContoursDst,
                                     const int grayThresh,
                                     const int minSizeThresh,
                                     const int maxSizeThresh)
{
    // generate a scratch image
    Mat tmpImg = Mat::zeros(src.size(), src.type());
    
    // generate a contours container scratch
    contours_t contours;
    
    // perform gray threshold
    Preprocessor::graythresh(src,grayThresh,tmpImg);
    
    // calculate the contours
    Preprocessor::calcContours(tmpImg,contours);
    
    // filter the contours
    Preprocessor::sizethreshold(contours, minSizeThresh, maxSizeThresh, acceptedContoursDst, biggerContoursDst);
}

void Preprocessor::preprocessTracking(Mat const & src,
                                      contours_t & acceptedContoursDst,
                                      contours_t & biggerContoursDst,
                                      int const grayThresh,
                                      int const minSizeThresh,
                                      int const maxSizeThresh,
                                      Backgroundsubtractor const & bs,
                                      bool checkRoiBorders)
{
    // generate a scratch image
    Mat tmpImg = Mat::zeros(src.size(), src.type());
    
    bs.subtractViaThresh(src,grayThresh,tmpImg);
    
    // generate a contours container scratch
    contours_t contours;
    
    // perform gray threshold
    Preprocessor::graythresh(tmpImg,grayThresh,tmpImg);
    
    // calculate the contours
    Preprocessor::calcContours(tmpImg,contours);
    
    // check if contours overrun image borders (as well as ROI-borders, if ROI selected)
    Preprocessor::borderRestriction(contours, src, checkRoiBorders);
    
    // filter the contours
    Preprocessor::sizethreshold(contours, minSizeThresh, maxSizeThresh, acceptedContoursDst, biggerContoursDst);
}
