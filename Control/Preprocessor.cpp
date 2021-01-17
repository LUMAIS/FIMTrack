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
#include <unordered_set>
#include "Preprocessor.hpp"

using namespace cv;
using std::vector;
using std::unordered_set;
using std::to_string;

Preprocessor::Preprocessor()
{
}

Preprocessor::~Preprocessor()
{
    destroyAllWindows();  // Destroy OpenCV windows
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

void showCvWnd(const String& wname, InputArray mat, unordered_set<String>& cvWnds) {
    if(!cvWnds.count(wname)) {
        cvWnds.insert(wname);  // A new window is being created
        printf("%s> %s, cvWnds: %lu\n", __FUNCTION__, wname.c_str(), cvWnds.size());
    }
    cv::imshow(wname, mat);
}

void showGrabCutMask(const Mat& mask, const char* imgName, unordered_set<String>& cvWnds) {
    constexpr uint8_t  CLR_BG = 0;
    constexpr uint8_t  CLR_BG_PROB = 0x44;
    constexpr uint8_t  CLR_FG_PROB = 0xAA;
    constexpr uint8_t  CLR_FG = 0xFF;

    Mat img(mask.size(), CV_8UC1, Scalar(CLR_BG));
    for(unsigned y = 0; y < mask.rows; ++y) {
        uint8_t* bval = img.ptr<uint8_t>(y);
        const uint8_t* clr = mask.ptr<uint8_t>(y);
        for(unsigned x = 0; x < mask.cols; ++x, ++bval, ++clr) {
            //uint8_t  c = mask.at<uint8_t>(y, x);
            uint8_t  c = *clr;
            if(c == cv::GC_BGD)
                continue;
            if(c == cv::GC_PR_BGD)
                c = CLR_BG_PROB;
            else if(c == cv::GC_PR_FGD)
                c = CLR_FG_PROB;
            else // if(c == cv::GC_FGD)
                c = CLR_FG;
            //img.at<uint8_t>(y, x) = c;
            *bval = c;
        }
    }
    showCvWnd(imgName, img, cvWnds);
}

void clearCvWnds(const char* wndName, unordered_set<String>& cvWnds)
{
    for(auto& wnd: cvWnds)
        cv::imshow(wnd, cv::Mat(cv::Size(0xFF, 1), CV_8UC1, Scalar(0)));
}

void resetCvWnds(const char* wndName, unordered_set<String>& cvWnds)
{
    if(cvWnds.size() <= 1)
        return;
    cv::destroyAllWindows();  // Destroy OpenCV windows
    cvWnds.clear();
}

void updateConditional2(const Mat& src, Mat& upd, uint8_t clrSrc, uint8_t clrUpd, uint8_t clrRes)
{
    assert(upd.size() == src.size() && upd.type() == src.type() && upd.type() == CV_8UC1 && "Input matrices should be of the same kind");
    for(unsigned y = 0; y < upd.rows; ++y) {
        const uint8_t* sval = src.ptr<uint8_t>(y);  // Note: other type for non CV_8UC1
        uint8_t* uval = upd.ptr<uint8_t>(y);  // Note: other type for non CV_8UC1
        for(unsigned x = 0; x < upd.cols; ++x, ++sval, ++uval)
            if(*uval == clrUpd && *sval == clrSrc)  // Note: typically, uval matches less frequently than sval
                *uval = clrRes;
    }
}

void Preprocessor::estimateThresholds(int& grayThresh, int& minSizeThresh, int& maxSizeThresh, Mat& imgFgOut,
                                      const Mat& img, const dlc::Larvae& larvae, const dlc::MatchStat& matchStat,
                                      bool smooth, const char* wndName, bool extraVis)
{
//    static uint8_t  cvWnds = 0;  // The number of created OpenCV windows
    static unordered_set<String>  cvWnds;  // Names of the operating OpenCV windows

    if(larvae.empty()) {
        //printf("%s> empty larvae, cvWnds: %lu\n", __FUNCTION__, cvWnds.size());
        clearCvWnds(wndName, cvWnds);
        return;
    }
    extraVis = extraVis && wndName;  // extraVis is applicable only when wndName
    if(!extraVis) {
        //printf("%s> !extraVis, destroying cvWnds: %lu\n", __FUNCTION__, cvWnds.size());
        resetCvWnds(wndName, cvWnds);
    }
    //printf("%s> cvWnds: %lu\n", __FUNCTION__, cvWnds.size());

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

    constexpr  unsigned  histSize = 256;
    std::array<unsigned, histSize>  larvaHist = {0};
    std::array<unsigned, histSize>  bgHist = {0};
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
        //printf("%s> fgrect: (%d + %d of %d, %d + %d of %d), span: %d\n", __FUNCTION__, fgrect.x, fgrect.width, img.cols, fgrect.y, fgrect.height, img.rows, span);

        Mat imgRoiFg;  // Foreground image
        Mat claheRoi;  // Image with the corrected contrast
        {
            constexpr uint8_t  CLR_BG = 0;
            constexpr uint8_t  CLR_BG_PROB = 0x44;
            constexpr uint8_t  CLR_FG_PROB = 0xAA;
            constexpr uint8_t  CLR_FG = 0xFF;

            Mat maskProbFg;  // Probable Foreground mask
            {
                Mat maskProbFgOrig(fgrect.size(), CV_8UC1, Scalar(0));  // Empty mask
                //cv::drawContours(maskProbFgOrig, hulls, -1, Scalar(cv::GC_PR_FGD), cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, Point(-fgrect.x, -fgrect.y));  // index, color; v or Scalar(v), cv::GC_FGD, GC_PR_FGD
                // Note: the color of nested (overlaping) contours is inverted, so each hull should be drawn separately
                for(const auto& hull: hulls)
                    cv::drawContours(maskProbFgOrig, vector<dlc::Larva::Points>(1, hull), 0, cv::GC_PR_FGD, cv::FILLED, cv::LINE_8, cv::noArray(), 0, Point(-fgrect.x, -fgrect.y)); // Scalar(cv::GC_PR_FGD)
                // Dilate convex to extract probable foreground
                const unsigned  opIters = round(1 + matchStat.distAvg / 4.f);  // Operation iterations
                cv::dilate(maskProbFgOrig, maskProbFg, Mat(), Point(-1, -1), opIters, cv::BORDER_CONSTANT, Scalar(cv::GC_PR_FGD));  // 2.5 .. 4; Iterations: ~= Ravg / 8 + 1
                if(extraVis) {
                    // Erode excluded convex hulls in the original foreground mask to produce the actual Foreground
                    Mat maskFg;  // Foreground mask
                    cv::erode(maskProbFgOrig, maskFg, Mat(), Point(-1, -1), opIters);  // 4..6..8; Iterations: ~= Ravg / 8 + 1
                    // Use stricter processing for more accurate results if possible (almost always)
                    if(cv::countNonZero(maskFg) <= opIters * opIters)
                        maskProbFgOrig.copyTo(maskFg);

                    // Produce the Probable Background from the extended inversion of the probable foreground
                    Mat inpMask(fgrect.size(), CV_8UC1, Scalar(GC_PR_BGD));  // Filled mask
                    inpMask.setTo(0, maskProbFg);  // Exclude foreground
                    // Erode convex to extract Probable background
                    cv::erode(inpMask, inpMask, Mat(), Point(-1, -1), 2);  // Iterations: ~= 1 .. 2
                    inpMask.setTo(cv::GC_PR_FGD, maskProbFg);
                    inpMask.setTo(cv::GC_FGD, maskFg);

                    showGrabCutMask(inpMask, "1.InpMasks", cvWnds);
                }
            }

            // Form the mask
            Mat maskRoi(fgrect.size(), CV_8UC1, Scalar(cv::GC_PR_BGD));  // Resulting mask;  GC_PR_BGD, GC_BGD
            Mat maskTmp(fgrect.size(), CV_8UC1);  // Temporary mask
            // Add CLAHE/OTSU based probable foreground
            cv::Ptr<CLAHE> clahe = createCLAHE();
            //const int  grain = 1 + matchStat.distAvg / 2.f;  // Square 3 or 4
            //clahe->setTilesGridSize(cv::Size(grain, grain));
            clahe->setClipLimit(8);  // 40; 2,4, 32; 16; 8
            const Mat imgRoi = img(fgrect);  // Image with the corrected contrast
            if(extraVis) {
                showCvWnd("OrigROI", imgRoi, cvWnds);

                //// Find and show contours
                //cv::Canny(imgRoi, maskTmp, grayThresh, min<double>(grayThresh * 1.618f + 1, min(grayThresh + 32, 255)));
                //showCvWnd("0.OrigRoiEdges", maskTmp, cvWnds);
            }
            clahe->apply(imgRoi, claheRoi);
            // Remove noise if any
            cv::fastNlMeansDenoising(claheRoi, claheRoi, 6);  // Note: h = 3 (default), 5 (larger denoising with a minor loss in details), or 7 (works the best for the CLAHE-processed images of narrow contrast bandwidth)
            if(extraVis) {
                const char*  wname = "2.ClaheROI Denoised";
                //fprintf(stderr, "DBG: %s\n", wname);
                showCvWnd(wname, claheRoi, cvWnds);
            }

            // Erode excessive probable foreground
            const unsigned  opClaheIters = 1 + round(matchStat.distAvg / 20.f);  // Operation iterations; ~= 2 for vid_!; 24 -> 12 for FG; 16 -> 8  // 12..16 for probable foreground; 6 .. 8 for the foreground

            // Adaptive optimal thresholds: THRESH_OTSU, THRESH_TRIANGLE
            // Identifies true background (THRESH_BINARY_INV | THRESH_TRIANGLE), and propable foreground (THRESH_BINARY | THRESH_OTSU)!
            // Note: reuse maskProbFg for the true foreground
            //const int  thrOtsu = round(cv::threshold(claheRoi, maskProbFg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU));  // THRESH_TRIANGLE, THRESH_OTSU; 0 o 8; 255 or 196
            cv::threshold(claheRoi, maskProbFg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  // THRESH_TRIANGLE, THRESH_OTSU; 0 o 8; 255 or 196
            //if(extraVis)
            //    showCvWnd("2.0.1.ProbFgClahe", maskProbFg, cvWnds);

            const unsigned  probFgxArea = cv::countNonZero(maskProbFg);
            if(probFgxArea > opClaheIters * opClaheIters && probFgxArea < maskProbFg.rows * maskProbFg.cols - opClaheIters * opClaheIters) {
                // Set CLAHE/TRIANGLE-based true background
                Mat maskClaheBg;
                //const int  thrTriag =
                cv::threshold(claheRoi, maskClaheBg, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_TRIANGLE);  // THRESH_TRIANGLE, THRESH_OTSU;  0 o 8; 255 or 196
                // Identify CLAHE-based contours to separate nearby foreground regions (touching larvae) in the GrabCut masks
                Mat maskClaheAth;
                contours_t  contours;
                constexpr int  MORPH_SIZE = 1;  // round(opClaheIters * 1.5f);  // 1-3
                const Mat erdKern = getStructuringElement(cv::MORPH_CROSS, Size(2*MORPH_SIZE + 1, 2*MORPH_SIZE + 1));  // MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE; kernel size = 2*MORPH_SIZE + 1; Point(morph_size, morph_size); Note: MORPH_ELLIPSE == MORPH_CROSS for the kernel size 3x3
                {
                    //constexpr int  blockSize = 13;
                    int blockSize = max<int>(1 + round(matchStat.distAvg / 2.f), 5);  // The recommended blockSize is around 13, a half of a larva size
                    if (blockSize % 2 == 0)
                        ++blockSize;
                    cv::adaptiveThreshold(claheRoi, maskClaheAth, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, 0);  // ADAPTIVE_THRESH_MEAN_C;; 1 + matchStat.distAvg / 4.f; 2; NOTE: positive C=2 requires inverted thresholding to have black BG
                    // Refine maskClaheAth, filtering out the background
                    maskClaheAth.setTo(0, maskClaheBg);
                    //if(extraVis)
                    //    showCvWnd("2.0.AthClaheProbFgRoi", maskClaheAth, cvWnds);

                    cv::erode(maskClaheAth, maskClaheAth, erdKern, Point(-1, -1), 1);  // 8 .. 12 .. 16; Scalar(cv::GC_PR_FGD); 1 or opClaheIters
                    cv::dilate(maskClaheAth, maskClaheAth, erdKern, Point(-1, -1), 1);
                    cv::findContours(maskClaheAth, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                    //cv::findContours(imgRoiFg, contours, topology, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);  // cv::CHAIN_APPROX_TC89_L1 or cv::CHAIN_APPROX_SIMPLE for the approximate compressed contours; CHAIN_APPROX_NONE to retain all points as they are;  RETR_EXTERNAL, RETR_LIST to retrieve all countours without any order
                    //printf("%s> external contours 2.1.CntRfnAthClaheProbFgRoi: %lu, blockSize: %d\n", __FUNCTION__, contours.size(), blockSize);
                    maskClaheAth.setTo(0);
                    cv::drawContours(maskClaheAth, contours, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
                    //cv::drawContours(maskClaheAth, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
                    //cv::drawContours(maskClaheAth, contours, -1, 0xFF, 1);  // cv::FILLED, 1
                }
                if(extraVis) {
                    showCvWnd("2.1.CntRfnAthClaheProbFgRoi", maskClaheAth, cvWnds);
                    maskTmp.setTo(CLR_BG_PROB);
                    maskTmp.setTo(CLR_FG_PROB, maskProbFg);
                    maskTmp.setTo(CLR_BG, maskClaheBg);
                    showCvWnd("2.2.ProbFgClahe", maskTmp, cvWnds);
                    //printf("%s thresholds> triag: %d, otsu: %d\n", __FUNCTION__, thrTriag, thrOtsu);
                    //
                    //// Find and show edges (more accurate than direct contours evaluation)
                    //cv::Canny(claheRoi, maskTmp, thrTriag, thrOtsu);
                    //showCvWnd("2.3.ClaheROI maskTmp", edges, cvWnds);
                }

                // Refine Probable Foreground mask considering both OTSU and adaptive thresholding of the CLAHE results
                //cv::bitwise_or(maskProbFg, maskClaheAth, maskProbFg);
                maskRoi.setTo(cv::GC_PR_FGD, maskProbFg);  // GC_PR_FGD, GC_FGD
                // Note: considering a large grain size in CLAHE, it makes sence to reduce a bit the foreground, also denoising the mask
                cv::erode(maskProbFg, maskProbFg, erdKern, Point(-1, -1), 1);  // 4..6, 8..12; Iterations: ~= Ravg / 8 + 1
                // Note: reuse maskProbFg for the merge of the reduced maskProbFg and maskTmp as true foreground
                //printf("%s> fgdCLAHE: %d, fgdMask: %d\n", __FUNCTION__, cv::countNonZero(maskProbFg), cv::countNonZero(maskTmp));
                //cv::bitwise_or(maskProbFg, maskTmp, maskProbFg);  // Note: maskTmp operates with GC_FGD, but maskProbFg operated with 0xFF
                cv::bitwise_and(maskProbFg, maskClaheAth, maskProbFg);
                maskRoi.setTo(cv::GC_FGD, maskProbFg);  // Foreground; GC_PR_FGD, GC_FGD
                //maskRoi.setTo(cv::GC_FGD, maskTmp);  // Foreground based on DLC
                maskRoi.setTo(cv::GC_BGD, maskClaheBg);
                if(extraVis) {
                    showCvWnd("2.4.RfnProbFgRoi", maskProbFg, cvWnds);
                    //showCvWnd("2.5.ClaheProbFgBg", maskClaheBg, cvWnds);
                }

                // Apply GrabCut to segment larva foreground vs background using hints
                {
                    Mat bgdModel, fgdModel;
                    Mat imgClr;
                    cv::cvtColor(imgRoi, imgClr, cv::COLOR_GRAY2BGR);  // CV_8UC3; imgCorr
                    if(extraVis)
                         showGrabCutMask(maskRoi, "2.5.PreGcMask", cvWnds);

                    //try {
                        cv::grabCut(imgClr, maskRoi, fgrect, bgdModel, fgdModel, 2, cv::GC_INIT_WITH_MASK);  // GC_INIT_WITH_RECT |
                    //} catch(cv::Exception& err) {
                    //    printf("WARNING %s> OpenCV exception in grabCut 1: %s\n", __FUNCTION__, err.msg.c_str());
                    //}
                    if(extraVis)
                         showGrabCutMask(maskRoi, "3.1.GcMask", cvWnds);

                    // Note: maskProbFg should be strict without the probable background to evaluate brightness threshold accurately
                    // Note: reuse maskProbFg for the foreground mask extended with the probable foreground withot the backround
                    cv::threshold(maskRoi, maskProbFg, cv::GC_FGD, 0xFF, cv::THRESH_TOZERO_INV);  // thresh, maxval, type;
                    updateConditional2(maskRoi, maskProbFg, cv::GC_PR_FGD, 0, 0xFF);  // Reset non-foreground regions of maskClaheAth (set to 0)
                    //maskProbFg.setTo(0, maskClaheBg);
                    if(extraVis)
                        showCvWnd("3.2.ProbFgRoiMask", maskProbFg, cvWnds);

                    // Reduce maskClaheAth with the Probable Background (it never includes true Background but could contain the Probable Background)
                    updateConditional2(maskProbFg, maskClaheAth, 0, 0xFF, 0);  // Reset non-foreground regions of maskClaheAth (set to 0)
                    // Extend true Background with 9.0.b2.mc.CntMorphOnAthProbFgRoi (based on adaptive thresholding of the foreground)
//                    // Note: reuse maskClaheAth for the morphology results; MORPH_OPEN is harder than erode (cross) + delate
//                    cv::morphologyEx(maskClaheAth, maskClaheAth, cv::MORPH_OPEN, erdKern, Point(-1, -1), 1);  // MORPH_OPEN, MORPH_CLOSE
//                    //showCvWnd("4.0m" + to_string(MORPH_SIZE) + ".MopnAthProbFgRoi", imgFgXm, cvWnds);

                    // Reduce holes in maskClaheAth (or eroded maskClaheAth) by extending it with:
                    // either erode the ProbFgRoiMask (works the best)
                    // or overlap the ProbFg of GrabCut with the eroded original ProbFg (without the true Foreground)
                    cv::erode(maskClaheAth, maskClaheAth, erdKern, Point(-1, -1), 1);
                    //cv::erode(maskProbFg, maskTmp, Mat(), Point(-1, -1), 2);  // 2, 1; erdKern or Mat(); (1 or opClaheIters=2) + 1; Note:
                    cv::erode(maskProbFg, maskProbFg, erdKern, Point(-1, -1), 3);  // 3, 2,[4]; erdKern (Cross = Ellipse 3x3) with 4 iters or Mat() (Rect) with 1 iter, the latter is harder even with a single iteration; Iterations: 3-4 ~= blockSize / 4
                    if(extraVis) {
                        showCvWnd("4.1.ErdCntAthProbFgRoi", maskClaheAth, cvWnds);

                        //cv::findContours(maskClaheAth, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                        //printf("%s> external contours 4.2.CntErdCntAthProbFgRoi: %lu\n", __FUNCTION__, contours.size());
                        //maskTmp.setTo(0);
                        //cv::drawContours(maskTmp, contours, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
                        ////cv::drawContours(maskTmp, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
                        ////cv::drawContours(maskTmp, contours, -1, 0xFF, 1);  // cv::FILLED, 1
                        //if(extraVis)
                        //    showCvWnd("4.2.CntErdCntAthProbFgRoi", maskTmp, cvWnds);

                        //showCvWnd("4.3r.ErdRcProbFgRoi", maskTmp, cvWnds);
                        showCvWnd("4.3+.ErdCrProbFgRoi", maskProbFg, cvWnds);
                    }
                    // Form the compound mask
                    //Mat maskClaheAthR;
                    //maskClaheAth.copyTo(maskClaheAthR);
                    //maskClaheAthR += maskTmp;
                    maskClaheAth += maskProbFg;
                    showCvWnd("5.1+.RfnMopnCrProbFgRoi", maskClaheAth, cvWnds);  // Works better than 8+.RfnErdCiProbFgRoi
                    //showCvWnd("5.1r.RfnMopnRcProbFgRoi", maskClaheAthR, cvWnds);

                    cv::findContours(maskClaheAth, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                    printf("%s> external contours 5.2.CntRfnMopnCrProbFgRoi: %lu\n", __FUNCTION__, contours.size());
                    maskTmp.setTo(0);
                    cv::drawContours(maskTmp, contours, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
                    //cv::drawContours(maskTmp, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
                    //cv::drawContours(maskTmp, contours, -1, 0xFF, 1);  // cv::FILLED, 1
                    if(extraVis)
                        showCvWnd("5.2.CntRfnMopnCrProbFgRoi", maskTmp, cvWnds);

                    //cv::morphologyEx(maskClaheAth, maskClaheAth, cv::MORPH_OPEN, erdKern, Point(-1, -1), 1);  // MORPH_OPEN, MORPH_CLOSE
                    cv::erode(maskClaheAth, maskClaheAth, erdKern, Point(-1, -1), 1);  // Less destroying than MORPH_OPEN
                    showCvWnd("6.1.ErdRfnMopnProbFgRoi", maskClaheAth, cvWnds);
                    cv::dilate(maskClaheAth, maskClaheAth, erdKern, Point(-1, -1), 1);
                    showCvWnd("6.2.MclRfnMopnProbFgRoi", maskClaheAth, cvWnds);

                    // Refine maskRoi from ProbFg to ProbBG [/FG] based on maskClaheAthXXX
                    updateConditional2(maskClaheAth, maskRoi, 0, cv::GC_PR_FGD, cv::GC_PR_BGD);
                    if(extraVis)
                         showGrabCutMask(maskRoi, "7.UpdGcMask", cvWnds);

                    // Refine the segmentation using the updated mask
                    cv::grabCut(imgClr, maskRoi, fgrect, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_MASK);  // GC_INIT_WITH_RECT | ; or cv::GC_EVAL
                    if(extraVis)
                         showGrabCutMask(maskRoi, "8.ResGcMask", cvWnds);

                    // Note: maskProbFg shuld be strict without probable background to evaluate brightness threshold accurately
                    // Note: reuse maskProbFg for the foreground mask extended with the probable foreground withot the backround
                    cv::threshold(maskRoi, maskProbFg, cv::GC_FGD, 0xFF, cv::THRESH_TOZERO_INV);  // thresh, maxval, type;
                    updateConditional2(maskRoi, maskProbFg, cv::GC_PR_FGD, 0, 0xFF);  // Reset non-foreground regions of maskClaheAth (set to 0)

                    // Note: reuse imgRoiFg for the resulting image
                    imgFgOut = Mat(img.size(), img.type(), Scalar(0));  // cv::GC_BGD
                    imgRoiFg = imgFgOut(fgrect);  // Produce the final probable foreground
                    imgRoi.copyTo(imgRoiFg, maskProbFg);
                    //imgRoiFg.copyTo(imgFgOut(fgrect));

                    if(wndName) {
                        Mat imgFgVis;  // Visualizing foreground image
                        imgFgOut.copyTo(imgFgVis);
                        cv::rectangle(imgFgVis, fgrect, cv::Scalar(CLR_FG), 1);
                        cv::drawContours(imgFgVis, hulls, -1, 0xFF, 2);  // index, color; v or Scalar(v), cv::GC_FGD, GC_PR_FGD
            //            cv::drawContours(maskProbFg, hulls, -1, cv::Scalar(255, 0, 0), 2, cv::LINE_8, cv::noArray(), INT_MAX, Point(-fgrect.x, -fgrect.y));  // index, color; v or Scalar(v), cv::GC_FGD, GC_PR_FGD
                        showCvWnd(wndName, imgFgVis, cvWnds);
                        //// Set image to black outside the mask
                        //bitwise_not(mask, mask);  // Invert the mask
                        //img.setTo(0, mask);  // Zeroize image by mask (outside the ROI)
                    }
                }
            } else printf("WARNING %s> the foreground (or background) is empty\n", __FUNCTION__);
        }

        // Evaluate brightness
        //fprintf(stderr, "%s> Brightness ROI (%d, %d; %d, %d) content:\n", __FUNCTION__, brect.x, brect.y, brect.width, brect.height);
        //if(extraVis)
        //     showCvWnd("HistOrigin", imgRoiFg, cvWnds);  // Same as ProbFgRoi
        //unsigned  brightness = 0;
        for(int y = 0; y < imgRoiFg.rows; ++y) {
            uint8_t* bval = imgRoiFg.ptr<uint8_t>(y);
            for(int x = 0; x < imgRoiFg.cols; ++x, ++bval) {
                //fprintf(stderr, "%u ", img.at<uchar>(y, x));
                //uint8_t  bval = imgRoiFg.at<uint8_t>(y, x);  // img
                // Omit zero mask
                if(*bval) {
                    ++larvaHist[*bval];  // brightness
                    //brightness += *bval;
                }
            }
            //fprintf(stderr, "\n");
        }

        ////  Check the histogram change (manually on the same frame) compared to the previous run
        //static Mat imgRoiFgPrev;
        //{
        //    if(imgRoiFg.size() != imgRoiFgPrev.size())
        //        imgRoiFg.copyTo(imgRoiFgPrev);
        //    Mat imgRoiRes;
        //    cv::absdiff(imgRoiFg, imgRoiFgPrev, imgRoiRes);
        //    showCvWnd("FrameDiff: ", imgRoiRes, cvWnds);
        //    printf("%s> acc bright: %u, diff pix: %d / %d, diff: %f (res: %f)\n", __FUNCTION__,
        //        brightness, countNonZero(imgRoiRes), imgRoiRes.rows * imgRoiRes.cols, cv::sum(imgRoiFg) - cv::sum(imgRoiFgPrev), cv::sum(imgRoiRes));
        //    cv::threshold(imgRoiRes, imgRoiRes, 0, 255, cv::THRESH_BINARY);
        //    showCvWnd("MaskFrameDiff: ", imgRoiRes, cvWnds);
        //}
        //imgRoiFg.copyTo(imgRoiFgPrev);

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
        // Note: foreground cutting shuold not be made when larvae foreground is separated well
        binsFixed = false;
        //// Cut the background candidates if applicable
        //const uint8_t  bgSoft = histSize / 6;  // 42
        //printf("%s> foreground reduction: binMax(%d [%d], %d), binMin1(%d, %d), hvMax: %d\n", __FUNCTION__
        //    , binMax1.x, bgSoft, binMax1.y, binMin1.x, binMin1.y, hvMax);
        vector<cv::Point>  histCutPts;
        //if(binsFixed && binMin1.x <= binMax1.x + binsz * 5 && binMin1.y * (1 + 2.f / binsz) < binMax1.y  // 1 + 2.25 / binsz -> 1.75;  + 2 -> 1.68
        //&& binMin1.x >= binMax1.x + binsz / 2) {  //  && binMax1.x <= bgSoft
        //    if(binMax1.x > bgSoft)
        //        printf("WARNING %s> foreground reduction might affect larva: binMax(%d > %d, %d), binMin1(%d, %d), hvMax: %d\n", __FUNCTION__
        //            , binMax1.x, bgSoft, binMax1.y, binMin1.x, binMin1.y, hvMax);
        //    // Correct bin margins to represent approximate centers of the bins
        //    binMax1.x -= binsz / 4;
        //    binMin1.x = max(binMin1.x - binsz * 3 / 4, binMax1.x + 1);
        //    unsigned  bgVal = 0;
        //    for(unsigned i = 0; i <= binMin1.x; ++i) {
        //        const auto& val = larvaHist[i];
        //        histCutPts.push_back(cv::Point(i, val));
        //        bgVal += val;
        //    }
        //    // Ensure that BG candidate is not a dark part of larva
        //    if(bgVal * 2 < count) {
        //        count -= bgVal;
        //        for(unsigned i = 0; i <= binMin1.x; ++i)
        //            larvaHist[i] = 0;
        //    } else binsFixed = false;
        //} else binsFixed = false;

        // Show larva histogram
        const uint16_t  rsz = 2;
        Mat imgHistFg(cv::Size(rsz*histSize + 2, histSize + 2), CV_8UC1, Scalar(cv::GC_BGD));
        vector<cv::Point>  histPts;
        histPts.reserve(histSize);
        for(uint16_t i = 0; i < histSize; ++i) {
            histPts.push_back(cv::Point(1 + rsz * i, 1 + histSize - round(float(larvaHist[i]) * histSize / hvMax)));
            if(i % 10 == 0)
                cv::putText(imgHistFg, to_string(i), cv::Point(1 + rsz * i, 1 + histSize), cv::FONT_HERSHEY_PLAIN, 1, 0x77);
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
        if(extraVis)
            showCvWnd("FG Hist", imgHistFg, cvWnds);

        // Cut foreground to ~<70% of the brightest values considering that the hull includes some background.
        // Note that the convex hull cause inclusion of some background pixel in the larva area.
        unsigned  ifgmin = 0;
        if(!binsFixed) {
            // Note: grabcut may catch some baground, so basic filtering is required (~ .02-.04)
            count *= 0.005f;  // 0.06f; 0.04 .. 0.08; 0.2f
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

//        // Tracing visualization for the inspection of per-larva separation
//        if(extraVis && !imgRoiFg.empty()) {
//            // Apply Final foreground as a  mask for the CLAHE results, and then fetch edges
//            Mat imgFgX(imgRoiFg.size(), CV_8UC1, Scalar(0xFF));
//            imgFgX.setTo(0, imgRoiFg);
//            claheRoi.setTo(0, imgFgX);
//            showCvWnd("7.ClaheProbFgRoi", claheRoi, cvWnds);

//            // Fetch edges for the CLAHE Probable Foreground ROI
//            const double  grayThreshSoft = min<double>(grayThresh * 1.618f + 1, min(grayThresh + 32, 0xFF));
//            Mat edges(imgRoiFg.size(), CV_8UC1);
//            //cv::Canny(claheRoi, edges, grayThresh, grayThreshSoft);  // Note: apertureSize for the Sobel operator larger than 3 causes too much noise, still without the larvae separation: 3 (defauly=t), 5, 7
//            //showCvWnd("8.EdgesClaheProbFgRoi", edges, cvWnds);

//            // Perform adaptive threshold for the CLAHE Probable Foreground ROI
//            int blockSize = max<int>(1 + round(matchStat.distAvg / 4.f), 3);
//            if (blockSize % 2 == 0)
//                ++blockSize;
//            const int  MORPH_SIZE = 1;  // round(opClaheIters * 1.5f);  // 1-3
//            const Mat morphKern = getStructuringElement(cv::MORPH_ELLIPSE, Size(2*MORPH_SIZE + 1, 2*MORPH_SIZE + 1));  // MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE; kernel size = 2*MORPH_SIZE + 1; Point(morph_size, morph_size); Note: MORPH_ELLIPSE == MORPH_CROSS for the kernel size 3x3
//            const Mat morphOpnKern = getStructuringElement(cv::MORPH_CROSS, Size(2*MORPH_SIZE + 1, 2*MORPH_SIZE + 1));  // MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE; kernel size = 2*MORPH_SIZE + 1; Point(morph_size, morph_size)
//            const unsigned mopIters = 1;  // 1..3
//            const unsigned opClaheIters = 3;
//            Mat imgFgXm;
//            contours_t contours;

//            // Note:
//            // Adaptive threshold with positive constant (requires BIN INV) produces too large contours (wrapping outside)
//            // Const less than -2 causes holes in a larva contour
//            cv::adaptiveThreshold(claheRoi, imgFgX, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, 0);  // ADAPTIVE_THRESH_MEAN_C;; 1 + matchStat.distAvg / 4.f; 2; NOTE: positive C=2 requires inverted thresholding to have black BG
//            showCvWnd("9.0.AthClaheProbFgRoi", imgFgX, cvWnds);
//            cv::morphologyEx(imgFgX, imgFgXm, cv::MORPH_OPEN, morphOpnKern, Point(-1, -1), mopIters);  // MORPH_OPEN, MORPH_CLOSE
//            showCvWnd("9.0.m" + to_string(MORPH_SIZE) + "." + to_string(mopIters) + ".MorphOnAthProbFgRoi", imgFgXm, cvWnds);
//            cv::findContours(imgFgXm, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//            printf("%s> external contours 9.0.m.CntMorphOnAthProbFgRoi: %lu\n", __FUNCTION__, contours.size());
//            //edges.copyTo(imgFgX);
//            imgFgXm.setTo(0);
//            cv::drawContours(imgFgXm, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
//            cv::drawContours(imgFgXm, contours, -1, 0xFF, 1);  // cv::FILLED, 1
//            showCvWnd("9.0.m1.1.CntMorphOnAthProbFgRoi", imgFgXm, cvWnds);

//            //cv::adaptiveThreshold(claheRoi, imgFgX, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, -1);  // ADAPTIVE_THRESH_MEAN_C;; 1 + matchStat.distAvg / 4.f; 2; NOTE: positive C=2 requires inverted thresholding to have black BG
//            //showCvWnd("9.1n.AthClaheProbFgRoi", imgFgX, cvWnds);
//            //cv::morphologyEx(imgFgX, imgFgXm, cv::MORPH_OPEN, morphOpnKern, Point(-1, -1), mopIters);  // MORPH_OPEN, MORPH_CLOSE
//            //showCvWnd("9.1n.m" + to_string(MORPH_SIZE) + "." + to_string(mopIters) + ".MorphOnAthProbFgRoi", imgFgXm, cvWnds);
//            //cv::findContours(imgFgXm, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//            //printf("%s> external contours 9.1n.m.CntMorphOnAthProbFgRoi: %lu\n", __FUNCTION__, contours.size());
//            ////edges.copyTo(imgFgX);
//            //imgFgXm.setTo(0);
//            //cv::drawContours(imgFgXm, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
//            //cv::drawContours(imgFgXm, contours, -1, 0xFF, 1);  // cv::FILLED, 1
//            //showCvWnd("9.1n.m1.1.CntMorphOnAthProbFgRoi", imgFgXm, cvWnds);

//            //cv::adaptiveThreshold(claheRoi, imgFgX, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 13, 0);  // ADAPTIVE_THRESH_MEAN_C;; 1 + matchStat.distAvg / 4.f; 2; NOTE: positive C=2 requires inverted thresholding to have black BG
//            //showCvWnd("9.0.13.AthClaheProbFgRoi", imgFgX, cvWnds);
//            cv::adaptiveThreshold(claheRoi, imgFgX, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize*2-1, 0);  // ADAPTIVE_THRESH_MEAN_C;; 1 + matchStat.distAvg / 4.f; 2; NOTE: positive C=2 requires inverted thresholding to have black BG
//            showCvWnd("9.0.b2.AthClaheProbFgRoi", imgFgX, cvWnds);
//            cv::erode(imgFgX, imgFgXm, morphOpnKern, Point(-1, -1), 1);  // 8 .. 12 .. 16; Scalar(cv::GC_PR_FGD)
//            showCvWnd("9.0.b2.e.ErdAthProbFgRoi", imgFgXm, cvWnds);
//            cv::findContours(imgFgXm, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//            //cv::findContours(imgRoiFg, contours, topology, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);  // cv::CHAIN_APPROX_TC89_L1 or cv::CHAIN_APPROX_SIMPLE for the approximate compressed contours; CHAIN_APPROX_NONE to retain all points as they are;  RETR_EXTERNAL, RETR_LIST to retrieve all countours without any order
//            printf("%s> external contours 9.0.b2.ec.CntErdAthProbFgRoi: %lu\n", __FUNCTION__, contours.size());
//            //edges.copyTo(imgFgX);
//            imgFgXm.setTo(0);
//            cv::drawContours(imgFgXm, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
//            cv::drawContours(imgFgXm, contours, -1, 0xFF, 1);  // cv::FILLED, 1
//            showCvWnd("9.0.b2.ec.CntErdAthProbFgRoi", imgFgXm, cvWnds);

//            cv::morphologyEx(imgFgX, imgFgXm, cv::MORPH_OPEN, morphOpnKern, Point(-1, -1), mopIters);  // MORPH_OPEN, MORPH_CLOSE
//            showCvWnd("9.0.b2.m" + to_string(MORPH_SIZE) + "." + to_string(mopIters) + ".MorphOnAthProbFgRoi", imgFgXm, cvWnds);
//            cv::findContours(imgFgXm, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//            printf("%s> external contours 9.0.b2.mc.CntMorphOnAthProbFgRoi: %lu\n", __FUNCTION__, contours.size());
//            //edges.copyTo(imgFgX);
//            imgFgXm.setTo(0);
//            cv::drawContours(imgFgXm, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
//            cv::drawContours(imgFgXm, contours, -1, 0xFF, 1);  // cv::FILLED, 1
//            showCvWnd("9.0.b2.mc.CntMorphOnAthProbFgRoi", imgFgXm, cvWnds);

//            // Find and show contours
//            cv::Canny(imgRoiFg, edges, grayThresh, grayThreshSoft);  // Note: apertureSize for the Sobel operator larger than 3 causes too much noise, still without the larvae separation: 3 (defauly=t), 5, 7
//            showCvWnd("10e.EdgesProbFgRoi", edges, cvWnds);
//            //Mat imgFgX(imgRoiFg.size(), CV_8UC1, Scalar(0));
//            cv::findContours(imgRoiFg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//            //cv::findContours(imgRoiFg, contours, topology, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);  // cv::CHAIN_APPROX_TC89_L1 or cv::CHAIN_APPROX_SIMPLE for the approximate compressed contours; CHAIN_APPROX_NONE to retain all points as they are;  RETR_EXTERNAL, RETR_LIST to retrieve all countours without any order
//            printf("%s> external contours 10c.ContsProbFgRoi: %lu\n", __FUNCTION__, contours.size());
//            //edges.copyTo(imgFgX);
//            imgFgX.setTo(0);
//            cv::drawContours(imgFgX, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
//            cv::drawContours(imgFgX, contours, -1, 0xFF, 1);  // cv::FILLED, 1
//            showCvWnd("10c.ContsProbFgRoi", imgFgX, cvWnds);

//            //cv::threshold(imgRoiFg, imgFgX, grayThreshSoft, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  // THRESH_TRIANGLE, THRESH_OTSU;  0 o 8; 255 or 196
//            //if(extraVis)
//            //    showCvWnd("11.o.OtsuProbFgMask", imgFgX, cvWnds);
//            //cv::threshold(imgRoiFg, imgFgX, grayThreshSoft, 255, cv::THRESH_BINARY | cv::THRESH_TRIANGLE);  // THRESH_TRIANGLE, THRESH_OTSU;  0 o 8; 255 or 196
//            //if(extraVis)
//            //    showCvWnd("11.t.TriagProbFgMask", imgFgX, cvWnds);


//            //imgRoiFg.copyTo(imgFgXm);
//            //cv::drawContours(imgFgXm, contours, -1, 0xFF, 1);
//            //showCvWnd("14.1n.m.ContsMorphAthProbFgRoi", imgFgXm, cvWnds);

//            //// Morphology operations over the Probable Foreground
//            //for(const unsigned opClaheIters: {1,3})
//            //{
//            //    //const unsigned  opClaheIters = 1 + round(matchStat.distAvg / 20.f);
//            //    for(const unsigned MORPH_SIZE: {2,3})
//            //    {
//            //        //const int  MORPH_SIZE = round(opClaheIters * 1.5f);  // 1-3
//            //        const Mat morphKern = getStructuringElement(cv::MORPH_ELLIPSE, Size(2*MORPH_SIZE + 1, 2*MORPH_SIZE + 1));  // MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE; kernel size = 2*MORPH_SIZE + 1; Point(morph_size, morph_size); Note: MORPH_ELLIPSE == MORPH_CROSS for the kernel size 3x3
//            //        //cv::morphologyEx(maskProbFg, maskProbFg, cv::MORPH_OPEN, kernel, Point(-1, -1), opClaheIters);  // Iterations: 1, 2, opClaheIters
//            //        cv::morphologyEx(imgRoiFg, imgFgX, cv::MORPH_OPEN, morphKern, Point(-1, -1), opClaheIters);  // MORPH_OPEN, MORPH_CLOSE
//            //        showCvWnd("7.3." + to_string(MORPH_SIZE) + "." + to_string(opClaheIters) + ".MorphProbFgRoiPFg", imgFgX, cvWnds);
//            //
//            //        //cv::morphologyEx(edges, imgFgX, cv::MORPH_CLOSE, morphKern, Point(-1, -1), opClaheIters);  // MORPH_OPEN
//            //        //showCvWnd("7.4." + to_string(MORPH_SIZE) + "." + to_string(opClaheIters) + ".MorphProbFgRoiEdges", imgFgX, cvWnds);
//            //    }
//            //}

//            //vector<vector<cv::Point>>  conts;
//            contours_t conts;
//            std::vector<cv::Vec4i>  topology;
////            // Filter contours by topology
////            cv::findContours(edges, contours, topology, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);  // cv::CHAIN_APPROX_TC89_L1 or cv::CHAIN_APPROX_SIMPLE for the approximate compressed contours; CHAIN_APPROX_NONE to retain all points as they are;  RETR_EXTERNAL, RETR_LIST to retrieve all countours without any order
////            enum ContourTopology {
////                CNT_NEXT,
////                CNT_PREV,
////                CNT_CHILD,
////                CNT_PARENT,
////            };
//////            for(unsigned i = 0; i < contours.size(); ++i) {
////////                const int  ipar = topology[i][CNT_PARENT];
////////                if(ipar >= 0 && topology[ipar][CNT_PARENT] == -1)  // Fetch 2nd level contours only
//////                if(topology[i][CNT_PARENT] == -1) {
//////                    conts.push_back(contours[i]);
//////                    Mat imgCont(imgRoiFg.size(), CV_8UC1, Scalar(0));
//////                    cv::drawContours(imgCont, conts, conts.size() - 1, 0xFF, 1);
//////                    showCvWnd(("ParContour " + to_string(conts.size() - 1)).c_str(), imgCont, cvWnds);
//////                }
//////            }
//////            printf("%s> contours ProbFg: %lu, parent contours: %lu\n", __FUNCTION__, contours.size(), conts.size());

////            imgRoiFg.copyTo(imgFgX);
////            cv::drawContours(imgFgX, conts, -1, 0xFF, 1);
////            showCvWnd("7.1.ProbFgRoiCmb", imgFgX, cvWnds);

//            //// Show probable background
//            //Mat imgBgRoi = imgRoiFg;
//            //imgBgRoi.setTo(CLR_BG);
//            //imgBgRoi.setTo(CLR_BG_PROB, maskClaheBg);
//            //showCvWnd("ProbBgRoi", imgBgRoi, cvWnds);
//        }
    } else imgFgOut = img;

    // Evaluate brightness histograms and larva area
    // Refine the foreground an approximaiton (convexHull) of the DLC-tracked larva contours
    vector<float>  areas;
    areas.reserve(larvae.size());
    for(const auto& hull: hulls) {
        // Note: OpenCV expects points to be ordered in contours, so convexHull() is used
        areas.push_back(cv::contourArea(hull));

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
    }
    cv::Scalar mean, stddev;
    cv::meanStdDev(areas, mean, stddev);
    std::sort(areas.begin(), areas.end());
    //const folat  rstd = 4;  // (3 .. 5+);  Note: 3 convers ~ 96% of results, but we should extend those margins
    //const int  minSizeThreshLim = 0.56f * areas[0]
    const float  expPerimMin = matchStat.distAvg - 0.36f * matchStat.distStd;  // 0.3 .. 0.4
    //minSizeThresh = (max<int>(min<int>(mean[0] - 4 * stddev[0], 0.56f * areas[0]), areas[0] * 0.36f) + expPerimMin * expPerimMin) / 2;  // 0.56 area ~= 0.75 perimiter; 0.36 a ~= 0.6 p
    // Note: only DLC-tracked larvae are reqrured to be detected, so the thresholds can be strict
    minSizeThresh = (max<int>(min<int>(mean[0] - stddev[0], 0.92f * areas[0]), areas[0] * 0.82f) + expPerimMin * expPerimMin) / 2;  // 0.56 area ~= 0.75 perimiter; 0.36 a ~= 0.6 p
    const float  expPerimMax = matchStat.distAvg + 3.8f * matchStat.distStd;  // 3.6 .. 3.8
////    maxSizeThresh = (min<int>(max<int>(mean[0] + 5 * stddev[0], 2.5f * areas[areas.size() - 1]), areas[areas.size() - 1] * 3.24f) + expPerimMax * expPerimMax) / 2;  // 2.5 area ~= 1.58 perimiter; 3.24 a ~= 1.8 p
    maxSizeThresh = (min<int>(max<int>(mean[0] + 2 * stddev[0], 2.5f * areas[areas.size() - 1]), areas[areas.size() - 1] * 3.24f) + expPerimMax * expPerimMax) / 2;  // 2.5 area ~= 1.58 perimiter; 3.24 a ~= 1.8 p
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
