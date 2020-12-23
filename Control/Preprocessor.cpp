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

void Preprocessor::estimateThresholds(int& grayThresh, int& minSizeThresh, int& maxSizeThresh, Mat& imgFgOut,
                                      const Mat& img, const dlc::Larvae& larvae, const dlc::MatchStat& matchStat,
                                      bool smooth, const char* wndName)
{
    if(larvae.empty())
        return;
    vector<float>  areas;
    areas.reserve(larvae.size());
    constexpr  unsigned  histSize = 256;
    std::array<unsigned, histSize>  larvaHist = {0};
    std::array<unsigned, histSize>  bgHist = {0};

    // Refine the foreground an approximaiton (convexHull) of the DLC-tracked larva contours
    dlc::Larva::Points  hull;
    vector<dlc::Larva::Points> hulls;
    // Identify an approximate foregroud ROI
    // Note: initially, store top-left and bottom-right points in the rect, and ther change the latter to height-width
    Rect  fgrect = {img.cols, img.rows, 0, 0};
    for(const auto& lv: larvae) {
        // Note: OpenCV expects points to be ordered in contours, so convexHull() is used
        hull.reserve(lv.points.size());
        cv::convexHull(lv.points, hull);
        hulls.push_back(std::move(hull));

        for(const auto& pt: lv.points) {
            if(pt.x < fgrect.x)
                fgrect.x = pt.x;
            if(pt.x > fgrect.width)
                fgrect.width = pt.x;

            if(pt.y < fgrect.y)
                fgrect.y = pt.y;
            if(pt.y > fgrect.height)
                fgrect.height = pt.y;
        }
    }
    // Convert bottom-right to height-width
    fgrect.width -= fgrect.x;
    fgrect.height -= fgrect.y;
    //if(!(fgrect.x >= 0 && fgrect.y >= 0))
    //    printf("%s> fgrect initial: %d + %d of %d, %d + %d of %d\n", __FUNCTION__, fgrect.x, fgrect.width, img.cols, fgrect.y, fgrect.height, img.rows);
    //assert(fgrect.x >= 0 && fgrect.y >= 0 && "Coordinates validation failed");
    //printf("%s> fgrect initial: %d + %d of %d, %d + %d of %f\n", __FUNCTION__, fgrect.x, fgrect.width, img.cols, fgrect.y, fgrect.height, img.rows);

    // Note: grayThresh retains the former value if can't be defined
    //if(grayThresh > GeneralParameters::iGrayThreshold * 1.5f || grayThresh * 1.5f < GeneralParameters::iGrayThreshold)
    //    grayThresh = GeneralParameters::iGrayThreshold;
    if(fgrect.width && fgrect.height) {
        // Expand the foreground ROI with the statistical span
        const int  span = matchStat.distAvg + matchStat.distStd;  // Note: we expend from the border points rather thatn from the center => *1..2 rather than *3
        int dx = span;
        fgrect.x -= dx;
        if(fgrect.x < 0) {
            dx += fgrect.x;
            fgrect.x = 0;
        }
        int dy = span;
        fgrect.y -= dy;
        if(fgrect.y < 0) {
            dy += fgrect.y;
            fgrect.y = 0;
        }
        fgrect.width += dx + span;
        if(fgrect.x + fgrect.width >= img.cols)
            fgrect.width = img.cols - fgrect.x;
        fgrect.height += dy + span;
        if(fgrect.y + fgrect.height >= img.rows)
            fgrect.height = img.rows - fgrect.y;
        //foreground = fgrect;
        //printf("%s> fgrect: (%d + %d of %d, %d + %d of %d), span: %d\n", __FUNCTION__, fgrect.x, fgrect.width, img.cols, fgrect.y, fgrect.height, img.rows, span);

        Mat imgFg;  // Foreground image
        {
            constexpr uint8_t  CLR_BG = 0;
            constexpr uint8_t  CLR_BG_PROB = 0x44;
            constexpr uint8_t  CLR_FG_PROB = 0xAA;
            constexpr uint8_t  CLR_FG = 0xFF;

            Mat mask(img.size(), CV_8UC1, Scalar(cv::GC_BGD));  // Resulting mask;  GC_PR_BGD, GC_BGD
            Mat maskLight;  // Lest strict mask for the FIMTrack processing
            Mat maskProbBg(fgrect.size(), CV_8UC1, Scalar(0x77));

            Mat maskProbFgOrig(maskProbBg.size(), CV_8UC1, Scalar(0));
            //cv::drawContours(maskProbFgOrig, hulls, -1, Scalar(cv::GC_PR_FGD), cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, Point(-fgrect.x, -fgrect.y));  // index, color; v or Scalar(v), cv::GC_FGD, GC_PR_FGD
            // Note: the color of nested (overlaping) contours is inverted, so each hull should be drawn separately
            for(const auto& hull: hulls)
                cv::drawContours(maskProbFgOrig, vector<dlc::Larva::Points>(1, hull), 0, cv::GC_PR_FGD, cv::FILLED, cv::LINE_8, cv::noArray(), 0, Point(-fgrect.x, -fgrect.y)); // Scalar(cv::GC_PR_FGD)
            // Dilate convex to extract probable foreground
            Mat maskProbFgx;
            const int  opIters = round(1 + matchStat.distAvg / 4.f);  // Operation iterations
            cv::dilate(maskProbFgOrig, maskProbFgx, Mat(), Point(-1, -1), opIters, cv::BORDER_CONSTANT, Scalar(cv::GC_PR_FGD));  // 2.5 .. 4; Iterations: ~= Ravg / 8 + 1
            Mat imgMask(maskProbFgx.size(), CV_8UC1, Scalar(0));  // Visualizing combined masks
            if(wndName)
                imgMask.setTo(CLR_FG_PROB, maskProbFgx);
            Mat maskFg;
            // Erode excluded convex hulls from the mask
            cv::erode(maskProbFgOrig, maskFg, Mat(), Point(-1, -1), opIters);  // 4..6..8; Iterations: ~= Ravg / 8 + 1
            if(wndName)
                imgMask.setTo(CLR_FG, maskFg);
            maskProbBg.setTo(0, maskProbFgx);  // cv::GC_BGD, GC_PR_BGD
            Mat maskProbBgLight;
            maskProbBg.copyTo(maskProbBgLight);
            // Erode convex to extract Probable background
            //Mat maskProbBg(maskProbFg.size(), CV_8UC1, Scalar(0xFF));
            //cv::erode(maskProbBg, maskProbBg, Mat(), Point(-1, -1), 1 + matchStat.distAvg / 2.6f);  // 2.5 .. 4; Iterations: ~= Ravg / 8 + 1
            cv::erode(maskProbBg, maskProbBg, Mat(), Point(-1, -1), 2);  // Iterations: ~= 1 .. 2
            if(wndName) {
                imgMask.setTo(CLR_BG_PROB, maskProbBg);
                cv::imshow("Masks", imgMask);
            }
            //maskProbBg.setTo(cv::GC_BGD, maskProbBg);  // cv::GC_BGD, GC_PR_BGD

            // Form the mask
            Mat maskRoi = mask(fgrect);
            //maskProbBg.copyTo(maskRoi, maskProbBg);
            //maskProbFg.copyTo(maskRoi, maskProbFgx);
            maskRoi.setTo(cv::GC_PR_BGD);  // GC_PR_FGD, GC_PR_BGD
            //maskRoi.setTo(cv::GC_PR_FGD, maskProbFgx);  // cv::GC_PR_FGD, GC_FGD
            if(wndName)
                imgMask.setTo(CLR_BG_PROB);  // Probable aAckground / Foreground; CLR_FG_PROB, CLR_BG_PROB

            // Add CLAHE/OTSU based probable foreground
            cv::Ptr<CLAHE> clahe = createCLAHE();
            //const int  grain = 1 + matchStat.distAvg / 2.f;  // Square 3 or 4
            //clahe->setTilesGridSize(cv::Size(grain, grain));
            clahe->setClipLimit(8);  // 40; 2,4, 32; 16; 8
            const Mat imgRoi = img(fgrect);  // Image with the corrected contrast
            if(wndName)
                cv::imshow("OrigROI", imgRoi);
            Mat claheRoi;  // Image with the corrected contrast
            clahe->apply(imgRoi, claheRoi);
            // Remove noise if any
            cv::fastNlMeansDenoising(claheRoi, claheRoi, 6);  // Note: h = 3 (default), 5 (larger denoising with a minor loss in details), or 7 (works the best for the CLAHE-processed images of narrow contrast bandwidth)
            if(wndName)
                cv::imshow("ClaheROI Denoised", claheRoi);
            //// Note: contours in overlapping rava can be identified using adaptiveThreshold
            //int blockSize = 1 + matchStat.distAvg / 4.f;
            //if (blockSize % 2 == 0)
            //    ++blockSize;
            //cv::adaptiveThreshold(imgCorr, imgCorr, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 1 + matchStat.distAvg / 4.f, 2)

            // Adaprive optimal thresholds: THRESH_OTSU, THRESH_TRIANGLE
            // Identifies true background (THRESH_BINARY_INV | THRESH_TRIANGLE), and propable foreground (THRESH_BINARY | THRESH_OTSU)!
            // Note: reuse maskProbFgx for the true foreground
            cv::threshold(claheRoi, maskProbFgx, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  // THRESH_TRIANGLE, THRESH_OTSU;  0 o 8; 255 or 196
            // Noise removal to remov holes in the forground larva mask
            // kernel = np.ones((3, 3), np.uint8)  # 3, 3
            //Mat kernel = (Mat_<int>(3, 3) <<
            //    0, 1, 0,
            //    1, -1, 1,
            //    0, 1, 0);
            const int  opClaheIters = round(1 + matchStat.distAvg / 20.f);  // Operation iterations; 24 -> 12 for FG; 16 -> 8
            {
                const int  MORPH_SIZE = opClaheIters;  // 1-3
                const Mat morphKern = getStructuringElement(cv::MORPH_ELLIPSE, Size(2*MORPH_SIZE + 1, 2*MORPH_SIZE + 1));  // MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE; kernel size = 2*MORPH_SIZE + 1; Point(morph_size, morph_size)
                //cv::morphologyEx(claheRoi, claheRoi, cv::MORPH_OPEN, kernel, Point(-1, -1), opClaheIters);  // Iterations: 1, 2, opClaheIters
                cv::morphologyEx(claheRoi, claheRoi, cv::MORPH_CLOSE, morphKern);  // MORPH_OPEN
            }
            // Erode excessive probable foreground
            // 12..16 for probable foreground; 6 .. 8 for the foreground
            cv::erode(maskProbFgx, maskProbFgx, Mat(), Point(-1, -1), opClaheIters);  // 4..6, 8..12; Iterations: ~= Ravg / 8 + 1
            maskRoi.setTo(cv::GC_PR_FGD, maskProbFgx);  // GC_PR_FGD, GC_FGD
            if(wndName)
                imgMask.setTo(CLR_FG_PROB, maskProbFgx);  // Foreground;  CLR_FG_PROB, CLR_FG
            cv::erode(maskProbFgx, maskProbFgx, Mat(), Point(-1, -1), opClaheIters);  // 4..6, 8..12; Iterations: ~= Ravg / 8 + 1
            maskRoi.setTo(cv::GC_FGD, maskProbFgx);  // GC_PR_FGD, GC_FGD
            maskRoi.setTo(cv::GC_FGD, maskFg);  // Foreground
            if(wndName) {
                //cv::imshow("ProbFgOtsuErdClahe", maskProbFgx);
                imgMask.setTo(CLR_FG, maskProbFgx);  // Foreground;  CLR_FG_PROB, CLR_FG
                imgMask.setTo(CLR_FG, maskFg);  // Foreground;
            }

            mask.copyTo(maskLight);
            maskRoi.setTo(cv::GC_PR_BGD, maskProbBg);  // GC_PR_BGD, GC_BGD
            // Set CLAHE/TRIANGLEbased true background
            cv::threshold(claheRoi, maskProbBg, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_TRIANGLE);  // THRESH_TRIANGLE, THRESH_OTSU;  0 o 8; 255 or 196
            //if(wndName) {
            //    cv::imshow("BgTrianErdClahe", maskProbBg);
            //}
            // Note: reuse maskProbFgx for the probable extended background
            cv::dilate(maskProbBg, maskProbFgx, Mat(), Point(-1, -1), round(1 + matchStat.distAvg / 14.f), cv::BORDER_CONSTANT);  // 8 .. 12 .. 16; Scalar(cv::GC_PR_FGD)
            maskRoi.setTo(cv::GC_PR_BGD, maskProbFgx);
            maskRoi.setTo(cv::GC_BGD, maskProbBg);
            if(wndName) {
                //cv::imshow("ProbBgTrianErdClahe", maskProbFgx);
                imgMask.setTo(CLR_BG_PROB, maskProbFgx);
                imgMask.setTo(CLR_BG, maskProbBg);
                cv::imshow("ThrClaheMasks", imgMask);
            }

            maskRoi = maskLight(fgrect);
            maskRoi.setTo(cv::GC_PR_BGD, maskProbBgLight);
            // Set CLAHE/TRIANGLEbased true background
            maskRoi.setTo(cv::GC_PR_BGD, maskProbFgx);
            maskRoi.setTo(cv::GC_BGD, maskProbBg);

            // Apply Grapcut to segment larva foreground vs background using hints
            {
                //cv::imshow("Orig Img", img);
                //// Apply Contrast Limited Adaptive Histogram Equalization before GrabCut
                //cv::Ptr<CLAHE> clahe = createCLAHE();
                ////const int  grain = 1 + matchStat.distAvg / 2.f;
                ////clahe->setTilesGridSize(cv::Size(grain, grain));
                //clahe->setClipLimit(8);  // 40; 2,4, 32; 16; 8
                //Mat imgCorr;  // Image with the corrected contrast
                //clahe->apply(img, imgCorr);
                //cv::imshow("Clahe Img", imgCorr);
                //// Remove noise if any
                //cv::fastNlMeansDenoising(imgCorr, imgCorr, 6);  // Note: h = 3 (default), 5 (larger denoising with a minor loss in details), or 7 (works the best for the CLAHE-processed images of narrow contrast bandwidth)
                //cv::imshow("Clahe Img Denoised", imgCorr);
                //// Adaprive optimal thresholds: THRESH_OTSU, THRESH_TRIANGLE
                //// Identifies true background (THRESH_BINARY_INV | THRESH_TRIANGLE), and propable foreground (THRESH_BINARY | THRESH_OTSU)!
                //cv::threshold(imgCorr, imgCorr, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  // THRESH_TRIANGLE, THRESH_OTSU;  0 o 8; 255 or 196
                //cv::imshow("Clahe+Th Img", imgCorr);

                Mat bgdModel, fgdModel;
                Mat imgClr;
                cv::cvtColor(img, imgClr, cv::COLOR_GRAY2BGR);  // CV_8UC3; imgCorr
                cv::grabCut(imgClr, mask, fgrect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT | cv::GC_INIT_WITH_MASK);  // GC_INIT_WITH_RECT |
                Mat maskFg;
                //cv::compare(mask, cv::GC_PR_FGD, maskFg, cv::CMP_EQ);  // Retain only the foreground
                cv::threshold(mask, maskFg, cv::GC_PR_FGD-1, cv::GC_PR_FGD, cv::THRESH_BINARY);  // thresh, maxval, type: ThresholdTypes
                img.copyTo(imgFg, maskFg);
                cv::threshold(mask, maskFg, cv::GC_FGD, cv::GC_FGD, cv::THRESH_TOZERO_INV);  // thresh, maxval, type
                img.copyTo(imgFg, maskFg);
                // Remove remained background
                Mat imgFgRoi = imgFg(fgrect);
                if(wndName) {
                    imgFgRoi.setTo(0, maskProbBg);
                    cv::imshow("ProbFgRoi", imgFgRoi);
                }

                //// Execute one more iteration of GrabCut
                //Mat maskRoi = mask(fgrect);
                //imgFgRoi.setTo(cv::GC_BGD, maskProbBg);
                //cv::grabCut(imgClr, mask, fgrect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT | cv::GC_INIT_WITH_MASK);

                //// Find Contours
                //vector<vector<cv::Point>>  conts;
                //cv::findContours(imgFgRoi, conts, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
                //cv::drawContours(imgFgRoi, conts, -1, 0x77, 1);  // index, color; v or Scalar(v), cv::GC_FGD, GC_PR_FGD

                //imgFgOut = imgFg;
                // Apply less strict foreground segmentation for the larva detection by FIMTrack
                //Mat bgdModelLight, fgdModelLight;
                //cv::cvtColor(img, imgClr, cv::COLOR_GRAY2BGR);  // CV_8UC3
                cv::grabCut(imgClr, maskLight, fgrect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT | cv::GC_INIT_WITH_MASK);
                cv::threshold(maskLight, maskFg, cv::GC_PR_FGD-1, cv::GC_PR_FGD, cv::THRESH_BINARY);  // thresh, maxval, type: ThresholdTypes
                img.copyTo(imgFgOut, maskFg);
                cv::threshold(maskLight, maskFg, cv::GC_FGD, cv::GC_FGD, cv::THRESH_TOZERO_INV);  // thresh, maxval, type
                img.copyTo(imgFgOut, maskFg);
                // Remove remained background
                imgFgRoi = imgFgOut(fgrect);
                imgFgRoi.setTo(0, maskProbBg);
                if(wndName) {
                    Mat imgFgVis;  // Visualizing foreground image
                    imgFgOut.copyTo(imgFgVis);
                    cv::rectangle(imgFgVis, fgrect, cv::Scalar(0xFF), 1);
                    cv::drawContours(imgFgVis, hulls, -1, 0xFF, 2);  // index, color; v or Scalar(v), cv::GC_FGD, GC_PR_FGD
        //            cv::drawContours(maskProbFg, hulls, -1, cv::Scalar(255, 0, 0), 2, cv::LINE_8, cv::noArray(), INT_MAX, Point(-fgrect.x, -fgrect.y));  // index, color; v or Scalar(v), cv::GC_FGD, GC_PR_FGD
                    cv::imshow(wndName, imgFgVis);
                    //// Set image to black outside the mask
                    //bitwise_not(mask, mask);  // Invert the mask
                    //img.setTo(0, mask);  // Zeroize image by mask (outside the ROI)
                }
            }
        }

        // Evaluate brightness
        const int  xEnd = fgrect.x + fgrect.width;
        const int  yEnd = fgrect.y + fgrect.height;
        //fprintf(stderr, "%s> Brightness ROI (%d, %d; %d, %d) content:\n", __FUNCTION__, brect.x, brect.y, brect.width, brect.height);
        for(int y = fgrect.y; y < yEnd; ++y) {
            for(int x = fgrect.x; x < xEnd; ++x) {
                //fprintf(stderr, "%u ", img.at<uchar>(y, x));
                uint8_t  bval = imgFg.at<uint8_t>(y, x);  // img
                // Omit zero mask
                if(bval)
                    ++larvaHist[bval];  // brightness
            }
            //fprintf(stderr, "\n");
        }
        // Calculate the number of values in foreground
        int32_t count = 0;
        unsigned  hvMax = 0;  // Global absolute max
        const uint8_t  binsz = 3;  // 3 .. 16
        assert(binsz >= 1 && "Bin size shuld be positive");
        cv::Point  binMax1(0, 0);  // First smoothed local max
        cv::Point  binMin1(0, INT_MAX);  // First smoothed local min
        cv::Point  bin(0, 0);  // Soothed val (bin)
        bool binsFixed = false;
        vector<cv::Point>  histBinPts;
        histBinPts.reserve(histSize / 2);
        for(auto num: larvaHist) {
            count += num;
            if(hvMax < num)
                hvMax = num;
            bin.y = (bin.y * (binsz - 1) + num) / binsz;
            // Evaluate binMin1 only after binMax1
            if(binsFixed)
                continue;
            if(bin.y >= 16) {
                if(binMax1.y > bin.y) {
                    if(binMin1.y > bin.y) {
                        binMin1.y = bin.y;
                        binMin1.x = bin.x;
                    } else if(bin.x >= binMax1.x + 2 * binsz)
                        binsFixed = true;
                } else if(binMin1.y == INT_MAX) {
                    binMax1.y = bin.y;
                    binMax1.x = bin.x;
                } else if(bin.x > binMax1.x + round(1.5f * binsz))
                    binsFixed = true;
                if(binsFixed)
                    continue;
            }
            // Save approximate centers of the bins
            histBinPts.push_back(cv::Point(max<int>(bin.x - binsz / 2, 0), bin.y));
            ++bin.x;
        }
        // Cut the background candidates if applicable
        const uint8_t  bgSoft = histSize / 6;  // 42
        printf("%s> foreground reduction: binMax(%d [%d], %d), binMin1(%d, %d), hvMax: %d\n", __FUNCTION__
            , binMax1.x, bgSoft, binMax1.y, binMin1.x, binMin1.y, hvMax);
        vector<cv::Point>  histCutPts;
        if(binsFixed && binMin1.x <= binMax1.x + binsz * 5 && binMin1.y * (1 + 2.f / binsz) < binMax1.y  // 1 + 2.25 / binsz -> 1.75;  + 2 -> 1.68
        && binMin1.x >= binMax1.x + binsz / 2) {  //  && binMax1.x <= bgSoft
            if(binMax1.x > bgSoft)
                printf("WARNING %s> foreground reduction might affect larva: binMax(%d > %d, %d), binMin1(%d, %d), hvMax: %d\n", __FUNCTION__
                    , binMax1.x, bgSoft, binMax1.y, binMin1.x, binMin1.y, hvMax);
            // Correct bin margins to represent approximate centers of the bins
            binMax1.x -= binsz / 4;
            binMin1.x = max(binMin1.x - binsz * 3 / 4, binMax1.x + 1);
            unsigned  bgVal = 0;
            for(unsigned i = 0; i <= binMin1.x; ++i) {
                const auto& val = larvaHist[i];
                histCutPts.push_back(cv::Point(i, val));
                bgVal += val;
            }
            // Ensure that BG candidate is not a dark part of larva
            if(bgVal * 2 < count) {
                count -= bgVal;
                for(unsigned i = 0; i <= binMin1.x; ++i)
                    larvaHist[i] = 0;
            } else binsFixed = false;
        } else binsFixed = false;

        // Show larva histogram
        const uint16_t  rsz = 2;
        Mat imgHistFg(cv::Size(rsz*histSize + 2, histSize + 2), CV_8UC1, Scalar(cv::GC_BGD));
        vector<cv::Point>  histPts;
        histPts.reserve(histSize);
        for(uint16_t i = 0; i < histSize; ++i) {
            histPts.push_back(cv::Point(1 + rsz * i, 1 + histSize - round(float(larvaHist[i]) * histSize / hvMax)));
            if(i % 10 == 0)
                cv::putText(imgHistFg, std::to_string(i), cv::Point(1 + rsz * i, 1 + histSize), cv::FONT_HERSHEY_PLAIN, 1, 0x77);
        }
        cv::polylines(imgHistFg, vector<vector<cv::Point>>(1, std::move(histPts)), false, 0xFF);  // Scalar(0xFF)
        if(binsFixed) {
            unsigned xBg = 1 + rsz * binMin1.x;
            cv::line(imgHistFg, cv::Point(xBg, 1), cv::Point(xBg, 1 + histSize), 0x44);
            vector<vector<cv::Point>>  hsPts;
            hsPts.push_back(move(histBinPts));
            hsPts.push_back(move(histCutPts));
            for(auto& pts: hsPts)
                for(auto& hp: pts) {
                    hp.x = 1 + rsz * hp.x;
                    hp.y = 1 + histSize - round(float(hp.y) * histSize / hvMax);
                }
            histCutPts = move(hsPts.back());
            hsPts.pop_back();
            cv::polylines(imgHistFg, hsPts, false, 0xAA);  // Scalar(0xFF)
            cv::polylines(imgHistFg, vector<vector<cv::Point>>(1, std::move(histCutPts)), false, 0x77);  // Scalar(0xFF)
        }
        if(wndName)
            cv::imshow("FG Hist", imgHistFg);

        // Cut foreground to ~<70% of the brightest values considering that the hull includes some background.
        // Note that the convex hull cause inclusion of some background pixel in the larva area.
        unsigned  ifgmin = 0;
        if(!binsFixed) {
            count *= 0.001f;  // 0.06f; 0.04 .. 0.08; 0.2f
            while(count > 0)
                if(larvaHist[ifgmin] <  count * 2)
                    count -= larvaHist[ifgmin++];
                else break;
        } else ifgmin = binMin1.x;

        //printf("%s> grayThreshGlobal: %d\n", __FUNCTION__, ifgmin);
        // Reset larvaHist
        larvaHist = {0};

        if(smooth) {
            constexpr float rGtEvol = 0.36;
            int resThresh = round(grayThresh * (1 - rGtEvol) + ifgmin * rGtEvol);
            // Prevent sticking to the past value when the difference is 1
            if(abs(resThresh - ifgmin) == 1 && resThresh == grayThresh)
                grayThresh = ifgmin;
            else grayThresh = resThresh;
        } else grayThresh = ifgmin;
        printf("%s> grayThresh: %d (from %d), binsFixed: %d, binMin1.x: %d, binMax1.x: %d\n", __FUNCTION__,
            grayThresh, ifgmin, binsFixed, binMin1.x, binMax1.x);
    } else imgFgOut = img;

//    // Evaluate brightness histograms
//    // Refine the foreground an approximaiton (convexHull) of the DLC-tracked larva contours
//    for(const auto& hull: hulls) {
//        // Note: OpenCV expects points to be ordered in contours, so convexHull() is used
//        areas.push_back(cv::contourArea(hull));

//        // Evaluate brightness
//        const Rect  brect = boundingRect(hull);
//        const int  xEnd = brect.x + brect.width;
//        const int  yEnd = brect.y + brect.height;
//        //fprintf(stderr, "%s> Brightness ROI (%d, %d; %d, %d) content:\n", __FUNCTION__, brect.x, brect.y, brect.width, brect.height);
//        for(int y = brect.y; y < yEnd; ++y) {
//            for(int x = brect.x; x < xEnd; ++x) {
//                //fprintf(stderr, "%u ", img.at<uchar>(y, x));
//                uint8_t  bval = imgFg.at<uint8_t>(y, x);  // img
//                // Omit 0 values, which are definitely background
//                if(bval < grayThesh)  // !bval
//                    continue;
//                if(pointPolygonTest(hull, Point2f(x, y), false) >= 0)
//                    ++larvaHist[bval];  // brightness
//                else ++bgHist[bval];
//            }
//            //fprintf(stderr, "\n");
//        }
//        //fprintf(stderr, "area: %f, brightness: %u\n", area, bval);
//    }
//    cv::Scalar mean, stddev;
//    cv::meanStdDev(areas, mean, stddev);
//    std::sort(areas.begin(), areas.end());
//    //const folat  rstd = 4;  // (3 .. 5+);  Note: 3 convers ~ 96% of results, but we should extend those margins
//    //const int  minSizeThreshLim = 0.56f * areas[0]
//    const float  expPerimMin = matchStat.distAvg - 0.36f * matchStat.distStd;  // 0.3 .. 0.4
//    //minSizeThresh = (max<int>(min<int>(mean[0] - 4 * stddev[0], 0.56f * areas[0]), areas[0] * 0.36f) + expPerimMin * expPerimMin) / 2;  // 0.56 area ~= 0.75 perimiter; 0.36 a ~= 0.6 p
//    // Note: only DLC-tracked larvae are reqrured to be detected, so teh thresholds can be strict
//    minSizeThresh = (max<int>(min<int>(mean[0] - stddev[0], 0.92f * areas[0]), areas[0] * 0.82f) + expPerimMin * expPerimMin) / 2;  // 0.56 area ~= 0.75 perimiter; 0.36 a ~= 0.6 p
//    const float  expPerimMax = matchStat.distAvg + 3.8f * matchStat.distStd;  // 3.6 .. 3.8
////    maxSizeThresh = (min<int>(max<int>(mean[0] + 5 * stddev[0], 2.5f * areas[areas.size() - 1]), areas[areas.size() - 1] * 3.24f) + expPerimMax * expPerimMax) / 2;  // 2.5 area ~= 1.58 perimiter; 3.24 a ~= 1.8 p
//    maxSizeThresh = (min<int>(max<int>(mean[0] + 2 * stddev[0], 2.5f * areas[areas.size() - 1]), areas[areas.size() - 1] * 3.24f) + expPerimMax * expPerimMax) / 2;  // 2.5 area ~= 1.58 perimiter; 3.24 a ~= 1.8 p
//    //printf("%s> minSizeThresh: %d (meanSdev: %d, areaMinLim: %u)\n", __FUNCTION__
//    //    , minSizeThresh, int(mean[0] - 4.f * stddev[0]), unsigned(0.56f * areas[0]));
//    //printf("%s> maxSizeThresh: %d (meanSdev: %d, areaMaxLim: %u)\n", __FUNCTION__
//    //    , maxSizeThresh, int(mean[0] + 5.f * stddev[0]), unsigned(2.5f * areas[areas.size() - 1]));

//    // Calculate the number of values in foreground
//    int32_t count = 0;
//    for(auto num: larvaHist)
//        count += num;
//    // Cut foreground to ~<96% of the brightest values considering that the hull includes some background.
//    // Note that the convex hull cause inclusion of some background pixel in the larva area.
//    count *= 0.04f;  // 0.04 .. 0.08 [0.16]; 0.12
//    unsigned  ifgmin = 0;
//    while(count > 0)
//        if(larvaHist[ifgmin] <  count * 2)
//            count -= larvaHist[ifgmin++];
//        else break;
//    // Evaluate the number of background values that are smaller that foreground ones
//    count = 0;
//    unsigned  ibg = 0;
//    for(;ibg < ifgmin; ++ibg)
//        count += bgHist[ibg];
//    // Take >=96% of the background values that are lower than foreground ones
//    count *= 0.96f;
//    for(ibg = 0; count > 0 && ibg < ifgmin; ++ibg)
//        if(bgHist[ibg] <  count * 2)
//            count -= bgHist[ibg];
//        else break;
//    // Take average index of background an foreground to identify the thresholding brightness
//    //grayThresh = max<int>((ibg + ifgmin) / 2, round(ifgmin * 0.96f));  // 0.75 .. 0.96
//    //grayThresh = max<float>((ibg + ifgmin) * 0.5f, ifgmin * 0.96f);  // 0.75 .. 0.96
//    //grayThresh = (max(ibg, grayThesh) * 3 + grayThesh) / 4;
//    grayThresh = (ibg * 4 + grayThesh) / 5;
//    printf("%s> grayThresh: %d (bgEst: %u, avg: %u, xFgMin: %u; grayThesh: %u)\n", __FUNCTION__, grayThresh, ibg, (ibg + ifgmin) / 2, unsigned(round(ifgmin * 0.96f)), grayThesh);
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
