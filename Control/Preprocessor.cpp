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

// Accessory routines ----------------------------------------------------------
void showCvWnd(const String& wname, InputArray mat, unordered_set<String>& cvWnds) {
    if(!cvWnds.count(wname)) {
        cvWnds.insert(wname);  // A new window is being created
        printf("%s> %s, cvWnds: %lu\n", __FUNCTION__, wname.c_str(), cvWnds.size());
    }
    cv::imshow(wname, mat);
}

void showGrabCutMask(const char* imgName, const Mat& mask, unordered_set<String>& cvWnds) {
    constexpr uint8_t  CLR_BG = 0;
    constexpr uint8_t  CLR_BG_PROB = 0x44;
    constexpr uint8_t  CLR_FG_PROB = 0xAA;
    constexpr uint8_t  CLR_FG = 0xFF;

    Mat img(mask.size(), CV_8UC1, Scalar(CLR_BG));
    for(int y = 0; y < mask.rows; ++y) {
        uint8_t* bval = img.ptr<uint8_t>(y);
        const uint8_t* clr = mask.ptr<uint8_t>(y);
        for(int x = 0; x < mask.cols; ++x, ++bval, ++clr) {
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
        cv::imshow(wnd, Mat::zeros(1, 0xFF, CV_8UC1));  // cv::Mat(1, 0xFF, CV_8UC1, Scalar(0)));
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
    for(int y = 0; y < upd.rows; ++y) {
        const uint8_t* sval = src.ptr<uint8_t>(y);  // Note: other type for non CV_8UC1
        uint8_t* uval = upd.ptr<uint8_t>(y);  // Note: other type for non CV_8UC1
        for(int x = 0; x < upd.cols; ++x, ++sval, ++uval)
            if(*uval == clrUpd && *sval == clrSrc)  // Note: typically, uval matches less frequently than sval
                *uval = clrRes;
    }
}

void Preprocessor::estimateThresholds(int& grayThresh, int& minSizeThresh, int& maxSizeThresh, Mat& imgFgOut,
                                      const Mat& img, const dlc::Larvae& larvae, const dlc::MatchStat& matchStat,
                                      bool smooth, const char* wndName, bool extraVis)
{
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

    vector<dlc::Larva::Points> hulls;
    Rect  fgrect = dlc::getLarvaeRoi(larvae, img.size(), matchStat.distAvg + matchStat.distStd, &hulls);

    constexpr  unsigned  histSize = 256;
    std::array<unsigned, histSize>  larvaHist = {0};
    std::array<unsigned, histSize>  bgHist = {0};
    // Note: grayThresh retains the former value if can't be defined
    //if(grayThresh > GeneralParameters::iGrayThreshold * 1.5f || grayThresh * 1.5f < GeneralParameters::iGrayThreshold)
    //    grayThresh = GeneralParameters::iGrayThreshold;
    int thrBrightRaw = 0;  // Initial brightness threshold
    if(!fgrect.empty()) {
        Mat imgRoiFg;  // Foreground image for the brightness evaluation
        Mat claheRoi;  // Image with the corrected contrast
        {
            constexpr uint8_t  CLR_BG = 0;
            constexpr uint8_t  CLR_BG_PROB = 0x44;
            constexpr uint8_t  CLR_FG_PROB = 0xAA;
            constexpr uint8_t  CLR_FG = 0xFF;

            Mat maskTmp(fgrect.size(), CV_8UC1);  // Temporary mask

            // Form Foreground and Background masks from the (DLC-exported) input larva contours
            //Mat maskProbFg;  // Probable Foreground mask
            {
                Mat maskProbFgOrig(fgrect.size(), CV_8UC1, Scalar(0));  // Empty mask;  Mat::zeros(fgrect.size(), CV_8UC1)
                //cv::drawContours(maskProbFgOrig, hulls, -1, Scalar(cv::GC_PR_FGD), cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, Point(-fgrect.x, -fgrect.y));  // index, color; v or Scalar(v), cv::GC_FGD, GC_PR_FGD
                // Note: the color of nested (overlaping) contours is inverted, so each hull should be drawn separately
                for(const auto& hull: hulls)
                    cv::drawContours(maskProbFgOrig, vector<dlc::Larva::Points>(1, hull), 0, cv::GC_PR_FGD, cv::FILLED, cv::LINE_8, cv::noArray(), 0, Point(-fgrect.x, -fgrect.y)); // Scalar(cv::GC_PR_FGD)
                // Dilate convex to extract probable foreground
                const unsigned  opIters = round(1 + matchStat.distAvg / 4.f);  // Operation iterations
                cv::dilate(maskProbFgOrig, maskTmp, Mat(), Point(-1, -1), opIters, cv::BORDER_CONSTANT, Scalar(cv::GC_PR_FGD));  // 2.5 .. 4; Iterations: ~= Ravg / 8 + 1
                //if(extraVis) {
                //    // Erode excluded convex hulls in the original foreground mask to produce the actual Foreground
                //    Mat maskFg;  // Foreground mask
                //    cv::erode(maskProbFgOrig, maskFg, Mat(), Point(-1, -1), opIters);  // 4..6..8; Iterations: ~= Ravg / 8 + 1
                //    // Use stricter processing for more accurate results if possible (almost always)
                //    if(cv::countNonZero(maskFg) <= opIters * opIters)
                //        maskProbFgOrig.copyTo(maskFg);
                //
                //    // Produce the Probable Background from the extended inversion of the probable foreground
                //    Mat inpMask(fgrect.size(), CV_8UC1, Scalar(GC_PR_BGD));  // Filled mask
                //    inpMask.setTo(0, maskTmp);  // Exclude foreground
                //    // Erode convex to extract Probable background
                //    cv::erode(inpMask, inpMask, Mat(), Point(-1, -1), 2);  // Iterations: ~= 1 .. 2
                //    inpMask.setTo(cv::GC_PR_FGD, maskTmp);
                //    inpMask.setTo(cv::GC_FGD, maskFg);
                //
                //    showGrabCutMask("0.InpMasks", inpMask, cvWnds);
                //}
            }


            // Form Foreground and Background masks from the original ROI
            const Mat imgRoi = img(fgrect);  // Image with the corrected contrast
            Mat maskRoi(imgRoi.size(), CV_8UC1, Scalar(cv::GC_PR_BGD));  // Original mask for the ROI
            Mat maskFg;  // Mask for the pure Foreground (both by OTSU andTriangle) of the original ROI
            Mat maskBg;  // Mask for the pure Background (both by OTSU andTriangle) of the original ROI

            //! \brief Compose mask from the probable foreground and probable background components
            //! \param maskFg  - probable foreground mask, which is reduced to the pure foreground
            //! \param maskBg  - probable background mask, which is reduced to the pure background
            //! \param[out] maskFg  - resulting compound mask with GrabCut labels
            auto composeMask = [&maskTmp](Mat& maskFg, Mat& maskBg, Mat& maskCompound)
            {
                // Identify the disaggrement of masks
                cv::bitwise_and(maskFg, maskBg, maskTmp);
                maskCompound.setTo(cv::GC_PR_FGD, maskTmp);
                // Reduce Probable Foreground and Background to the pure ones
                maskFg.setTo(0, maskTmp);
                maskBg.setTo(0, maskTmp);
                maskCompound.setTo(cv::GC_FGD, maskFg);
                maskCompound.setTo(cv::GC_BGD, maskBg);
            };


            if(extraVis)
                showCvWnd("OrigROI", imgRoi, cvWnds);

            // Apply thresholds to the original image to identify Foregrounds and Backgrounds (probable and pure/clear)
            // Identify the probable Foreground
            const int  thrOtsu = cv::threshold(imgRoi, maskFg, 0, 0xFF, cv::THRESH_BINARY | cv::THRESH_OTSU);  // THRESH_TRIANGLE, THRESH_OTSU; 0 o 8; 255 or 196

            const unsigned  probFgArea = cv::countNonZero(maskFg);
            const unsigned  opClaheIters = 1 + round(matchStat.distAvg / 20.f);  // Operation iterations; ~= 2 for vid_!; 24 -> 12 for FG; 16 -> 8  // 12..16 for probable foreground; 6 .. 8 for the foreground
            if(probFgArea > opClaheIters * opClaheIters && probFgArea < maskFg.rows * maskFg.cols - opClaheIters * opClaheIters) {
                // Identify the probable Background
                const int  thrTriag = cv::threshold(imgRoi, maskBg, 0, 0xFF, cv::THRESH_BINARY_INV | cv::THRESH_TRIANGLE);  // THRESH_TRIANGLE, THRESH_OTSU;  0 o 8; 255 or
                thrBrightRaw = thrTriag;
                //if(extraVis) {
                //    showCvWnd("1.1.ProbFg", maskFg, cvWnds);
                //    showCvWnd("1.2.ProbBg", maskBg, cvWnds);
                //}

                //// Erode the original background mask to prevent loss of values (see vid1-031)
                constexpr int  MORPH_SIZE = 1;  // round(opClaheIters * 1.5f);  // 1-3
                const Mat erdKern = getStructuringElement(cv::MORPH_CROSS, Size(2*MORPH_SIZE + 1, 2*MORPH_SIZE + 1));  // MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE; kernel size = 2*MORPH_SIZE + 1; Point(morph_size, morph_size); Note: MORPH_ELLIPSE == MORPH_CROSS for the kernel size 3x3
                //////cv::erode(maskBg, maskTmp, Mat(), Point(-1, -1), 1);  // 1-2
                ////cv::erode(maskBg, maskBg, erdKern, Point(-1, -1), 2);  // 3, 1, 2-4; Yields more accurate contours compared to a rect mask
                cv::erode(maskBg, maskBg, erdKern, Point(-1, -1), 3);

                composeMask(maskFg, maskBg, maskRoi);
                //printf("%s> 1.3.MaskOrigROI thresholds (triag: %d, otsu: %d)\n", __FUNCTION__, thrTriag, thrOtsu);
                if(extraVis) {
                     showGrabCutMask("1.3.MaskOrigROI", maskRoi, cvWnds);
                     ////showCvWnd("1.3r.ErdBgOrig", maskTmp, cvWnds);
                     //showCvWnd("1.3+.ErdBgOrig", maskBg, cvWnds);
                }

                // Apply adaptive thresholding to the original ROI
                constexpr int  blockSize = 13;  // >= round(ClaheClipLimit * 1.5f)
                //int blockSize = max<int>(1 + round(matchStat.distAvg / 2.f), 5);  // The recommended blockSize is around 13, a half of a larva size
                //if (blockSize % 2 == 0)
                //    ++blockSize;
                Mat maskAth;
                contours_t  contours;
                cv::adaptiveThreshold(imgRoi, maskAth, 0xFF, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, 1);  // 0 or 1;  ADAPTIVE_THRESH_MEAN_C;; 1 + matchStat.distAvg / 4.f; 2; NOTE: positive C=2 requires inverted thresholding to have black BG
                // Refine maskAth, filtering out the background, reducing the noise
                maskAth.setTo(0, maskBg);
                if(extraVis)
                    showCvWnd("2.1.AthProbFgRoi", maskAth, cvWnds);

                cv::erode(maskAth, maskAth, erdKern, Point(-1, -1), 1);  // 8 .. 12 .. 16; Scalar(cv::GC_PR_FGD); 1 or opClaheIters
                cv::dilate(maskAth, maskAth, erdKern, Point(-1, -1), 1);
                //if(extraVis)
                //    showCvWnd("2.2.RfnAthProbFgRoi", maskAth, cvWnds);
                cv::findContours(maskAth, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                //printf("%s> external contours 2.3.CntRfnAthProbFgRoi: %lu, blockSize: %d\n", __FUNCTION__, contours.size(), blockSize);
                maskAth.setTo(0);
                cv::drawContours(maskAth, contours, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
                //cv::drawContours(maskAth, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
                //cv::drawContours(maskAth, contours, -1, 0xFF, 1);  // cv::FILLED, 1
                if(extraVis)
                    showCvWnd("2.3.CntRfnAthProbFgRoi", maskAth, cvWnds);

//				//// Note: adaptiveThreshold with a small block size (3-7) results in too noisy contours and lots of noise;
//				//// A positive const larger than 2 causes loss of separation regions, a negative const causes too many holes
//				cv::adaptiveThreshold(imgRoi, maskTmp, 0xFF, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, 0);  // ADAPTIVE_THRESH_MEAN_C;; 1 +
//				maskTmp.setTo(0, maskBg);
//				if(extraVis)
//					showCvWnd("2.1.b0.AthProbFgRoi", maskTmp, cvWnds);
//				cv::erode(maskTmp, maskTmp, erdKern, Point(-1, -1), 1);  // 8 .. 12 .. 16; Scalar(cv::GC_PR_FGD); 1 or opClaheIters
//				cv::dilate(maskTmp, maskTmp, erdKern, Point(-1, -1), 1);
//				//if(extraVis)
//				//	showCvWnd("2.2.b0.RfnAthProbFgRoi", maskTmp, cvWnds);
//				cv::findContours(maskTmp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//				maskTmp.setTo(0);
//				cv::drawContours(maskTmp, contours, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
//				if(extraVis)
//					showCvWnd("2.3.b0.CntRfnAthProbFgRoi", maskTmp, cvWnds);

                // Extend GrabCut mask with maskAth
                // Erode the adaptive thresholding of Clahe for more strict foreground (otherwise more touching larae can't be separated properly, see vid_1-026, 040, 051, 060).
                // Late erosion reduces noise caused by 3.2.ProbFgRoiMask on merging it later to expand the connected components (see vid_1-040, 042)
                cv::erode(maskAth, maskAth, erdKern, Point(-1, -1), 1);
                if(extraVis)
                    showCvWnd("2.4.ErdCntRfnAthProbFgRoi", maskAth, cvWnds);


                // Find edges (more accurate than direct contours evaluation)
                Mat edgesOrig;  // Orifinal edges to separate touching larvae
                cv::Canny(imgRoi, edgesOrig, max(max<float>(thrTriag * 0.86f, thrTriag-8), 0.f), thrOtsu);
                // Note: maskFg and even non-reduced maskAth are not sufficient to filter inner and noisy edgesOrig, where some small contours might be remained
                if(extraVis)
                    showCvWnd("3.1.EdgesOrigRoi", edgesOrig, cvWnds);

                //// Refine the edges, removing inner elements and noise by substructing the eroded closed contours
                //cv::findContours(edgesOrig, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                //
                //// Conver contours to convex hulls to fill the internal area
                //vector<contour_t>  hulls;
                //hulls.reserve(contours.size());
                //contour_t  hull;
                //for(const auto& cnt: contours) {
                //    cv::convexHull(cnt, hull);
                //	hulls.push_back(hull);
                //}
                //
                //Mat maskEdges = Mat::zeros(edgesOrig.size(), edgesOrig.type());  // (edgesOrig.size(), edgesOrig.type(), Scalar(0))
                //cv::drawContours(maskEdges, hulls, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
                //showCvWnd("3.2.FilledCntOrigRoi", maskEdges, cvWnds);
                //cv::erode(maskEdges, maskEdges, erdKern, Point(-1, -1), 2);
                //

                // Refine the edges, removing inner elements and noise by substructing 2.6.RfnProbFgRoi
                //Mat maskFlt;
                ////cv::erode(maskAth, maskFlt, Math(), Point(-1, -1), 1);  // 8 .. 12 .. 16; Scalar(cv::GC_PR_FGD); 1 or
                //cv::erode(maskAth, maskFlt, erdKern, Point(-1, -1), 2);  // 8 .. 12 .. 16; Scalar(cv::GC_PR_FGD); 1 or
                //edgesOrig -= maskFlt;
//                edgesOrig -= maskAth;
//                if(extraVis)
//                    showCvWnd("3.3.RfnEdgesOrigRoi", edgesOrig, cvWnds);


                //// Reduce maskAth with the Probable Background (it never includes true Background but might contain the Probable Background)
                // Note: for the original masks, the Probable Background in many cases is true Foreground, so that reduction might be harmful
                //updateConditional2(maskRoiOrig, maskAth, cv::GC_PR_BGD, 0xFF, 0);  // Reset non-foreground regions of maskAth (set to 0)
                //if(extraVis)
                //	showCvWnd("4.1.RfnErdCntRfnAthProbFgRoi", maskAth, cvWnds);

                // Extend GrabCut mask with the refined maskAth as probable foreground
                updateConditional2(maskAth, maskRoi, 0xFF, cv::GC_PR_BGD, cv::GC_PR_FGD);
                if(extraVis)
                     showGrabCutMask("4.2.PreGcMaskOrig", maskRoi, cvWnds);

                //// Note: GrabCut can be iteratively executed until the convergence of the output mask with the input one, where the agreed Probable
                //// Foreground is set to the pure Foreground in the input mask. However, a more efficient way is to define more foreground at once based on CLAHE
                //// Apply GrabCut to segment larva foreground vs background using hints
                //{
                //	Mat bgdModel, fgdModel;
                //	Mat imgClr;
                //	cv::cvtColor(imgRoi, imgClr, cv::COLOR_GRAY2BGR);  // CV_8UC3; imgCorr
                //
                //	//try {
                //		cv::grabCut(imgClr, maskRoiOrig, fgrect, bgdModel, fgdModel, 2, cv::GC_INIT_WITH_MASK);  // GC_INIT_WITH_RECT |
                //	//} catch(cv::Exception& err) {
                //	//    printf("WARNING %s> OpenCV exception in grabCut 1: %s\n", __FUNCTION__, err.msg.c_str());
                //	//}
                //	if(extraVis)
                //		 showGrabCutMask("4.3.GcMaskOrig", maskRoiOrig, cvWnds);
                //}


                // Form the ROI mask for GrabCut segmentation base on CLAHE, where the foreground is iderntified well
                {
                    cv::Ptr<CLAHE> clahe = createCLAHE();
                    //const int  grain = 1 + matchStat.distAvg / 2.f;  // ~Square 3
                    //clahe->setTilesGridSize(cv::Size(grain, grain));
                    clahe->setClipLimit(8);  // 40; 2,4, 32; 16; 8;  ~blockSize / 1.5, but also may not have any relation to the block size
                    clahe->apply(imgRoi, claheRoi);
                    //if(extraVis)
                    //    showCvWnd("5.0.ClaheRoi", claheRoi, cvWnds);
                    //{
                    //    const Size claheCell = clahe->getTilesGridSize();
                    //    printf("%s> 5.0.ClaheRoi clip: %f, grid: (%d, %d)\n", __FUNCTION__
                    //        , contours.size(), clahe->getClipLimit(), claheCell.width, claheCell.height);
                    //}
                }
                // Remove noise if any
                cv::fastNlMeansDenoising(claheRoi, claheRoi, 6);  // Note: h = 3 (default), 5 (larger denoising with a minor loss in details), or 7 (works the best for the CLAHE-processed images of narrow contrast bandwidth); ~= round(ClaheClipLimit / 1.5f)
                if(extraVis) {
                    const char*  wname = "5.1.DnsClaheRoi";
                    //fprintf(stderr, "DBG: %s\n", wname);
                    showCvWnd(wname, claheRoi, cvWnds);
                }


                // Form Foreground and Background masks from the CLAHE ROI
                Mat maskClaheRoi(imgRoi.size(), CV_8UC1, Scalar(cv::GC_PR_BGD));  // Clahe mask for the ROI
                Mat maskClaheFg;  // Mask for the pure Foreground (both by OTSU andTriangle) of the Clahe ROI
                Mat maskClaheBg;  // Mask for the pure Background (both by OTSU andTriangle) of the Clahe ROI

                // Adaptive optimal thresholds: THRESH_OTSU, THRESH_TRIANGLE
                // Identify true background (THRESH_BINARY_INV | THRESH_TRIANGLE), and propable foreground (THRESH_BINARY | THRESH_OTSU)!
                const int  thrClhOtsu = cv::threshold(claheRoi, maskClaheFg, 0, 0xFF, cv::THRESH_BINARY | cv::THRESH_OTSU);  // THRESH_TRIANGLE, THRESH_OTSU; 0 o 8; 255 or 196
                // Identify CLAHE/TRIANGLE-based true background
                const int  thrClhTriag = cv::threshold(claheRoi, maskClaheBg, 0, 0xFF, cv::THRESH_BINARY_INV | cv::THRESH_TRIANGLE);  // THRESH_TRIANGLE, THRESH_OTSU;  0 o 8; 255 or 196
                //if(extraVis) {
                //    showCvWnd("6.1.ProbFg", maskClaheFg, cvWnds);
                //    showCvWnd("6.2.ProbBg", maskClaheBg, cvWnds);
                //}

                // Refine the pure CLAHE background as either of the both former pure backgrounds
                // Note: that is not applicable for the Foreground, which is excessive in CLAHE
                cv::bitwise_or(maskBg, maskClaheBg, maskClaheBg);

                //Mat maskProbFg;  // Probable Foreground mask
                //maskClaheFg.copyTo(maskProbFg);
                composeMask(maskClaheFg, maskClaheBg, maskClaheRoi);
                //printf("%s> 6.3.MaskClaheRoi thresholds (triag: %d, otsu: %d)\n", __FUNCTION__, thrClhTriag, thrClhOtsu);
                if(extraVis)
                     showGrabCutMask("6.3.MaskClaheRoi", maskClaheRoi, cvWnds);

                // Update the Clahe mask, respecting the pure background
                maskClaheRoi.setTo(cv::GC_BGD, maskClaheBg);
                if(extraVis)
                     showGrabCutMask("6.4.RfnMaskClaheRoi", maskClaheRoi, cvWnds);


                // Identify CLAHE-based contours to separate nearby foreground regions (touching larvae) in the GrabCut masks
                Mat maskClaheAth;

                cv::adaptiveThreshold(claheRoi, maskClaheAth, 0xFF, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, 1);  // 0 or 1;  ADAPTIVE_THRESH_MEAN_C;; 1 + matchStat.distAvg / 4.f; 2; NOTE: positive C=2 requires inverted thresholding to have black BG
                // Refine maskClaheAth, filtering out the background, reducing the noise
                maskClaheAth.setTo(0, maskClaheBg);
                if(extraVis)
                    showCvWnd("7.1.AthProbFgClaheRoi", maskClaheAth, cvWnds);

                cv::erode(maskClaheAth, maskClaheAth, erdKern, Point(-1, -1), 1);  // 8 .. 12 .. 16; Scalar(cv::GC_PR_FGD); 1 or opClaheIters
                cv::dilate(maskClaheAth, maskClaheAth, erdKern, Point(-1, -1), 1);
                //if(extraVis)
                //    showCvWnd("7.2.RfnAthProbFgRoi", maskClaheAth, cvWnds);
                cv::findContours(maskClaheAth, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); // cv::CHAIN_APPROX_TC89_L1 or cv::CHAIN_APPROX_SIMPLE for the approximate compressed contours; CHAIN_APPROX_NONE to retain all points as they are;  RETR_EXTERNAL, RETR_LIST to retrieve all countours without any order
                //printf("%s> external contours 7.3.CntRfnAthProbFgClaheRoi: %lu, blockSize: %d\n", __FUNCTION__, contours.size(), blockSize);
                maskClaheAth.setTo(0);
                cv::drawContours(maskClaheAth, contours, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
                //cv::drawContours(maskClaheAth, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
                //cv::drawContours(maskClaheAth, contours, -1, 0xFF, 1);  // cv::FILLED, 1
                if(extraVis)
                    showCvWnd("7.3.CntRfnAthProbFgClaheRoi", maskClaheAth, cvWnds);

//				// Note: adaptiveThreshold with a small block size (3-7) results in too noisy contours and lots of noise;
//				// A positive const larger than 2 causes loss of separation regions, a negative const causes too many holes
//				cv::adaptiveThreshold(claheRoi, maskTmp, 0xFF, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, 0);  // ADAPTIVE_THRESH_MEAN_C;; 1 +
//				maskTmp.setTo(0, maskBg);
//				if(extraVis)
//					showCvWnd("7.1.b0.AthProbFgClaheRoi", maskTmp, cvWnds);
//				cv::erode(maskTmp, maskTmp, erdKern, Point(-1, -1), 1);  // 8 .. 12 .. 16; Scalar(cv::GC_PR_FGD); 1 or opClaheIters
//				cv::dilate(maskTmp, maskTmp, erdKern, Point(-1, -1), 1);
//				//if(extraVis)
//				//	showCvWnd("7.2.b0.RfnAthProbFgClaheRoi", maskTmp, cvWnds);
//				cv::findContours(maskTmp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//				maskTmp.setTo(0);
//				cv::drawContours(maskTmp, contours, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
//				if(extraVis)
//					showCvWnd("7.3.b0.CntRfnAthProbFgClaheRoi", maskTmp, cvWnds);


                //// Note: Clahe results are pretty noisy, which makes them useless for the edge detection
                //// Find edges (more accurate than direct contours evaluation)
                //Mat edgesClahe;  // Orifinal edges to separate touching larvae
                //cv::Canny(claheRoi, edgesClahe, max(max<float>(thrTriag * 0.86f, thrTriag-8), 0.f), thrOtsu);
                //// Note: maskFg and even non-reduced maskClaheAth are not sufficient to filter inner and noisy edgesClahe, where some small contours might be remained
                //if(extraVis) {
                //	showCvWnd("8.1.EdgesClaheRoi", edgesClahe, cvWnds);
                //	edgesClahe.setTo(0, maskClaheFg);  // maskClaheBg, maskProbFg
                //	showCvWnd("8.2.NfgEdgesClaheRoi", maskTmp, cvWnds);
                //}


                // Extend GrabCut mask with maskClaheAth
                // Erode the adaptive thresholding of Clahe for more strict foreground (otherwise more touching larae can't be separated properly, see vid_1-026, 040, 051, 060).
                // Note: Late erosion reduces noise caused by 3.2.ProbFgRoiMask on merging it later to expand the connected components (see vid_1-040, 042)
                cv::erode(maskClaheAth, maskClaheAth, erdKern, Point(-1, -1), 1);
                if(extraVis)
                    showCvWnd("9.1.ErdCntRfnAthProbFgClaheRoi", maskClaheAth, cvWnds);
                // Reduce maskClaheAth with the Probable Background (it never includes true Background but might contain the Probable Background)
                updateConditional2(maskClaheRoi, maskClaheAth, cv::GC_PR_BGD, 0xFF, 0);  // Reset non-foreground regions of maskClaheAth (set to 0)
                if(extraVis)
                    showCvWnd("9.2.RfnErdCntRfnAthProbFgRoi", maskClaheAth, cvWnds);
                //// Refine edges, substructing maskClaheAth to eliminate noisy contours, which do not bound larvae
                //cv::dilate(edgesOrig, edgesOrig, erdKern, Point(-1, -1), 1);
                //edgesOrig -= maskClaheAth;
                //if(extraVis)
                //    showCvWnd("9.3.RfnEdgesClaheRoi", edgesOrig, cvWnds);
                // Refine GrabCut mask with the refined maskClaheAth as the only foreground, the remained former foreground shuold become possible foreground
                updateConditional2(maskClaheAth, maskClaheRoi, 0, cv::GC_FGD, cv::GC_PR_FGD);
                // Degrade pure Foreground to the one present in maskAth
                updateConditional2(maskAth, maskClaheRoi, 0, cv::GC_FGD, cv::GC_PR_FGD);
                // Ensure that GrabCut mask does not mark as pure background regions that belong to maskAth, promote those regions to the probable background
                updateConditional2(maskAth, maskClaheRoi, 0xFF, cv::GC_BGD, cv::GC_PR_BGD);
                if(extraVis)
                     showGrabCutMask("9.4.PreGcMaskClahe", maskClaheRoi, cvWnds);

                //// Refine Probable Foreground mask considering both OTSU and adaptive thresholding of the CLAHE results
                ////cv::bitwise_or(maskProbFg, maskClaheAth, maskProbFg);
                //maskClaheRoi.setTo(cv::GC_PR_FGD, maskProbFg);  // GC_PR_FGD, GC_FGD
                //// Note: considering a large grain size in CLAHE, it makes sence to reduce a bit the foreground, also denoising the mask
                //cv::erode(maskProbFg, maskProbFg, erdKern, Point(-1, -1), 1);  // 4..6, 8..12; Iterations: ~= Ravg / 8 + 1
                //// Note: reuse maskProbFg for the merge of the reduced maskProbFg and maskTmp as true foreground
                ////printf("%s> fgdCLAHE: %d, fgdMask: %d\n", __FUNCTION__, cv::countNonZero(maskProbFg), cv::countNonZero(maskTmp));
                ////cv::bitwise_or(maskProbFg, maskTmp, maskProbFg);  // Note: maskTmp operates with GC_FGD, but maskProbFg operated with 0xFF
                //cv::bitwise_and(maskProbFg, maskClaheAth, maskProbFg);
                //maskClaheRoi.setTo(cv::GC_FGD, maskProbFg);  // Foreground; GC_PR_FGD, GC_FGD
                ////maskClaheRoi.setTo(cv::GC_FGD, maskTmp);  // Foreground based on DLC
                //maskClaheRoi.setTo(cv::GC_BGD, maskClaheBg);
                //if(extraVis) {
                //	showCvWnd("9.1.RfnProbFgClaheRoi", maskProbFg, cvWnds);
                //    //showCvWnd("9.2.ClaheProbFgBg", maskClaheBg, cvWnds);
                //}


                // Apply GrabCut to segment larva foreground vs background using hints
                {
                    Mat bgdModel, fgdModel;
                    Mat imgClr;
                    cv::cvtColor(imgRoi, imgClr, cv::COLOR_GRAY2BGR);  // CV_8UC3; imgCorr

                    //try {
                        cv::grabCut(imgClr, maskClaheRoi, fgrect, bgdModel, fgdModel, 2, cv::GC_INIT_WITH_MASK);  // GC_INIT_WITH_RECT |
                    //} catch(cv::Exception& err) {
                    //    printf("WARNING %s> OpenCV exception in grabCut 1: %s\n", __FUNCTION__, err.msg.c_str());
                    //}
                    if(extraVis)
                         showGrabCutMask("9.5.GcMaskClahe", maskClaheRoi, cvWnds);

                    // Feth exclusively probable foreground without the probable background to evaluate brightness threshold accurately
                    Mat maskGcProbFg(maskClaheRoi.size(), maskClaheRoi.type(), Scalar(0));  // Mat::zeros(maskClaheRoi.size(), maskClaheRoi.type())
                    //cv::threshold(maskClaheRoi, maskGcProbFg, cv::GC_PR_FGD-1, 0xFF, cv::THRESH_BINARY);  // thresh, maxval, type;
                    updateConditional2(maskClaheRoi, maskGcProbFg, cv::GC_PR_FGD, 0, 0xFF);  // Reset non-foreground regions of maskClaheAth (set to 0)
                    //maskGcProbFg.setTo(0, maskClaheBg);  // Note: this is redundant because ProbFg never contains the true Backgound
                    if(extraVis)
                        showCvWnd("9.6.ProbFgRoiMask", maskGcProbFg, cvWnds);

                    // Refine the edges
                    cv::dilate(edgesOrig, edgesOrig, erdKern, Point(-1, -1), 1);
                    // Extend with the reduced maskGcProbFg
                    // Extensin Option 1
//                    edgesOrig += maskGcProbFg;
//                    // Reduce edges by the eroded maskGcProbFg to retain internal foreground
//                    {
//                        cv::erode(maskGcProbFg, maskTmp, erdKern, Point(-1, -1), 2);  // 8 .. 12 .. 16;
//                        //cv::erode(maskGcProbFg, maskTmp, Mat(), Point(-1, -1), 1);  // 8 .. 12 .. 16;
//                        edgesOrig -= maskTmp;
//                    }
                    // Extensin Option 2
                    cv::Canny(maskGcProbFg, maskTmp, 0, 255);
                    cv::dilate(maskTmp, maskTmp, erdKern, Point(-1, -1), 1);
                    cv::erode(maskTmp, maskTmp, erdKern, Point(-1, -1), 1);
                    //showCvWnd("9.6.2.EdgProbFgRoiMask", maskTmp, cvWnds);
                    edgesOrig += maskTmp;  // 9.6.ProbFgRoiMask
//                    edgesOrig += maskGcProbFg;  // 9.6.ProbFgRoiMask
                    // Add reduced maskGcProbFg
                    cv::subtract(maskGcProbFg, edgesOrig, maskTmp);
                    cv::erode(maskTmp, maskTmp, erdKern, Point(-1, -1), 1);  // 8 .. 12 .. 16;
                    if(extraVis)
                        showCvWnd("9.6.-1.ErdRemMaskGcProbFg", maskTmp, cvWnds);
                    edgesOrig += maskTmp;

                    // Substract reduced CLAHE foreground 6.4
                    cv::erode(maskClaheFg, maskTmp, erdKern, Point(-1, -1), 6);
                    if(extraVis)
                        showCvWnd("9.6.0.ErdMaskClaheFg", maskTmp, cvWnds);
                    edgesOrig -= maskTmp;

                    edgesOrig -= maskAth;  // 2.4.ErdCntRfnAthProbFgRoi

                    edgesOrig -= maskClaheAth;  // 9.2.RfnErdCntRfnAthProbFgRoi
                    //// Reduce edges by the convex hulls of maskClaheAth
                    //Mat maskHullClaheAth = Mat::zeros(maskClaheAth.size(), maskClaheAth.type());
                    //{
                    //    cv::findContours(maskClaheAth, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                    //    vector<contour_t>  hulls;
                    //    hulls.reserve(contours.size());
                    //    contour_t  hull;
                    //    for(const auto& cnt: contours) {
                    //        cv::convexHull(cnt, hull);
                    //        hulls.push_back(hull);
                    //    }
                    //    cv::drawContours(maskHullClaheAth, contours, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
                    //    showCvWnd("9.6.5.CntRfnEdgesOrigRoi", maskHullClaheAth, cvWnds);
                    //}
                    //edgesOrig -= maskHullClaheAth;  // 9.2.RfnErdCntRfnAthProbFgRoi

                    if(extraVis) {
                        showCvWnd("9.6.1.RfnEdgesOrigRoi", edgesOrig, cvWnds);
                        //showCvWnd("9.6.2.maskTmp", maskTmp, cvWnds);
                        showCvWnd("9.6.3.maskAth", maskAth, cvWnds);
                        showCvWnd("9.6.4.maskClaheAth", maskClaheAth, cvWnds);
                        //showCvWnd("9.6.5.maskClaheAthHulls", maskHullClaheAth, cvWnds);
                        //showCvWnd("9.6.6.maskClaheFg", maskClaheFg, cvWnds);

                        //cv::findContours(edgesOrig, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                        ////printf("%s> external contours 9.6.5.CntRfnEdgesOrigRoi: %lu\n", __FUNCTION__, contours.size());
                        //maskTmp.setTo(0);
                        //cv::drawContours(edgesOrig, contours, -1, 0xFF, 1);  // cv::FILLED, 1
                        //showCvWnd("9.6.6.CntRfnEdgesOrigRoi", edgesOrig, cvWnds);

                        //cv::erode(edgesOrig, maskTmp, erdKern, Point(-1, -1), 1);
                        //showCvWnd("9.6.7.ErdCntRfnEdgesOrigRoi", maskTmp, cvWnds);
                    }
                    //cv::erode(edgesOrig, edgesOrig, erdKern, Point(-1, -1), 1);
                    cv::dilate(edgesOrig, edgesOrig, erdKern, Point(-1, -1), 1);
                    cv::erode(edgesOrig, edgesOrig, erdKern, Point(-1, -1), 1);
                    if(extraVis)
                        showCvWnd("9.6.8.Rfn2EdgesOrigRoi", edgesOrig, cvWnds);
                    //edgesOrig -= maskTmp;
                    //showCvWnd("9.6.9.ResEdgesOrigRoi", edgesOrig, cvWnds);

                    // Reduce holes in maskClaheAth (or eroded maskClaheAth) by extending it with:
                    // either erode the ProbFgRoiMask (works the best)
                    // or overlap the ProbFg of GrabCut with the [eroded] original ProbFg (without the true Foreground)
                    //cv::erode(maskGcProbFg, maskTmp, Mat(), Point(-1, -1), 2);  // 2, 1; erdKern or Mat(); (1 or opClaheIters=2) + 1; Note:
                    cv::erode(maskGcProbFg, maskGcProbFg, erdKern, Point(-1, -1), 3);  // 3, 2,[4]; erdKern (Cross = Ellipse 3x3) with 4 iters or Mat() (Rect) with 1 iter, the latter is harder even with a single iteration; Iterations: 3-4 ~= blockSize / 4
                    if(extraVis) {
                        //cv::findContours(maskClaheAth, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                        //printf("%s> external contours 9.6.CntErdCntAthProbFgRoi: %lu\n", __FUNCTION__, contours.size());
                        //maskTmp.setTo(0);
                        //cv::drawContours(maskTmp, contours, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
                        ////cv::drawContours(maskTmp, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
                        ////cv::drawContours(maskTmp, contours, -1, 0xFF, 1);  // cv::FILLED, 1
                        //if(extraVis)
                        //    showCvWnd("9.7.CntErdCntAthProbFgRoi", maskTmp, cvWnds);

                        //showCvWnd("9.7r.ErdRcProbFgRoi", maskTmp, cvWnds);
                        showCvWnd("9.7+.ErdCrProbFgRoi", maskGcProbFg, cvWnds);
                    }

                    // Erode the adaptive thresholding of Clahe for more strict foreground (otherwise more touching larae can't be separated properly, see vid_1-026, 040, 051, 060).
                    // Late erosion reduces noise caused by 3.2.ProbFgRoiMask on merging it later to expand the connected components (see vid_1-040, 042)
                    cv::erode(maskClaheAth, maskClaheAth, erdKern, Point(-1, -1), 1);
                    // Reduce maskClaheAth with the Probable Background (it never includes true Background but might contain the Probable Background)
                    updateConditional2(maskClaheRoi, maskClaheAth, cv::GC_PR_BGD, 0xFF, 0);  // Reset non-foreground regions of maskClaheAth (set to 0)
                    if(extraVis)
                        showCvWnd("9.8.ErdCntAthProbFgRoi", maskClaheAth, cvWnds);

                    // Form the compound mask
                    //Mat maskClaheAthR;
                    //maskClaheAth.copyTo(maskClaheAthR);
                    //maskClaheAthR += maskTmp;
                    maskClaheAth += maskGcProbFg;
                    if(extraVis)
                        showCvWnd("9.9+.RfnMopnCrProbFgRoi", maskClaheAth, cvWnds);  // Works better than 8+.RfnErdCiProbFgRoi
                    //showCvWnd("9.9r.RfnMopnRcProbFgRoi", maskClaheAthR, cvWnds);

                    //cv::findContours(maskClaheAth, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                    //printf("%s> external contours 9.10.CntRfnMopnCrProbFgRoi: %lu\n", __FUNCTION__, contours.size());
                    //maskTmp.setTo(0);
                    //cv::drawContours(maskTmp, contours, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
                    ////cv::drawContours(maskTmp, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
                    ////cv::drawContours(maskTmp, contours, -1, 0xFF, 1);  // cv::FILLED, 1
                    //if(extraVis)
                    //    showCvWnd("9.10.CntRfnMopnCrProbFgRoi", maskTmp, cvWnds);

//                    // Apply morphological refinement to reduce possible noise further
//                    //cv::morphologyEx(maskClaheAth, maskClaheAth, cv::MORPH_OPEN, erdKern, Point(-1, -1), 1);  // MORPH_OPEN, MORPH_CLOSE
//                    cv::erode(maskClaheAth, maskClaheAth, erdKern, Point(-1, -1), 1);  // Less destroying than MORPH_OPEN
//                    //if(extraVis)
//                    //    showCvWnd("10.1.ErdRfnMopnProbFgRoi", maskClaheAth, cvWnds);

//                    //cv::findContours(maskClaheAth, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//                    //printf("%s> external contours 10.1c.CntMclRfnMopnProbFgRoi: %lu\n", __FUNCTION__, contours.size());
//                    //maskTmp.setTo(0);
//                    //cv::drawContours(maskTmp, contours, -1, 0xFF, cv::FILLED);  // cv::FILLED, 1
//                    ////cv::drawContours(maskTmp, contours, -1, 0x77, cv::FILLED);  // cv::FILLED, 1
//                    ////cv::drawContours(maskTmp, contours, -1, 0xFF, 1);  // cv::FILLED, 1
//                    //if(extraVis)
//                    //    showCvWnd("10.1c.CntMclRfnMopnProbFgRoi", maskTmp, cvWnds);

//                    cv::dilate(maskClaheAth, maskClaheAth, erdKern, Point(-1, -1), 1);
//                    if(extraVis)
//                        showCvWnd("10.2.MclRfnMopnProbFgRoi", maskClaheAth, cvWnds);

                    // Refine maskClaheRoi from ProbFg to ProbBG [/FG] based on maskClaheAthXXX
                    //updateConditional2(maskClaheAth, maskClaheRoi, 0, cv::GC_PR_FGD, cv::GC_PR_BGD);
                    updateConditional2(maskClaheAth, maskClaheRoi, 0, cv::GC_FGD, cv::GC_PR_FGD);
                    updateConditional2(maskAth, maskClaheRoi, 0, cv::GC_PR_FGD, cv::GC_PR_BGD);
                    if(extraVis)
                         showGrabCutMask("11.RfnPreGcMask", maskClaheRoi, cvWnds);

                    // Refine the segmentation using the updated mask
                    cv::grabCut(imgClr, maskClaheRoi, fgrect, bgdModel, fgdModel, 3, cv::GC_INIT_WITH_MASK);  // GC_INIT_WITH_RECT | ; or cv::GC_EVAL
                    if(extraVis)
                         showGrabCutMask("12.ResGcMask", maskClaheRoi, cvWnds);

                    // Extract the foreground mask extended with the probable foreground withot the backround to evaluate brightness threshold accurately
                    cv::threshold(maskClaheRoi, maskTmp, cv::GC_FGD, 0xFF, cv::THRESH_TOZERO_INV);  // thresh, maxval, type;
                    updateConditional2(maskClaheRoi, maskTmp, cv::GC_PR_FGD, 0, 0xFF);  // Reset non-foreground regions of maskClaheAth (set to 0)

                    // Note: reuse imgRoiFg for the resulting image
                    imgFgOut = Mat::zeros(img.size(), img.type());  // Mat(img.size(), img.type(), Scalar(0));  // cv::GC_BGD
                    imgRoiFg = imgFgOut(fgrect);  // Produce the final probable foreground
                    imgRoi.copyTo(imgRoiFg, maskTmp);
                    // Ensure that edges separating larava are present
                    imgRoiFg.setTo(0, edgesOrig);
                    //imgRoiFg.copyTo(imgFgOut(fgrect));
                    if(extraVis) {
                        showCvWnd("13.Foreground", imgRoiFg, cvWnds);  // Resulting foreground

                        cv::findContours(imgRoiFg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                        maskTmp.setTo(0);
                        cv::drawContours(maskTmp, contours, -1, 0xFF, 1);  // cv::FILLED, 1
                        showCvWnd("14.CntForeground", maskTmp, cvWnds);

                        const int nblobs = cv::connectedComponents(imgRoiFg, maskTmp);
                        printf("%s> external contours 14.CntForeground: %lu, connected components: %d\n", __FUNCTION__, contours.size(), nblobs);
                    }

                    //cv::findContours(imgRoiFg, contours, topology, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);  // cv::CHAIN_APPROX_TC89_L1 or cv::CHAIN_APPROX_SIMPLE for the approximate compressed contours; CHAIN_APPROX_NONE to retain all points as they are;  RETR_EXTERNAL, RETR_LIST to retrieve all countours without any order

                    if(wndName) {
                        Mat imgFgVis;  // Visualizing foreground image
                        imgFgOut.copyTo(imgFgVis);
                        cv::rectangle(imgFgVis, fgrect, cv::Scalar(CLR_FG), 1);
                        cv::drawContours(imgFgVis, hulls, -1, 0xFF, 2);  // index, color; v or Scalar(v), cv::GC_FGD, GC_PR_FGD
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
        //    cv::threshold(imgRoiRes, imgRoiRes, 0, 0xFF, cv::THRESH_BINARY);
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
        // Note: thrBrightness shuold not cut larvae in most cases
        if(thrBrightRaw && thrBrightRaw > ifgmin)
            ifgmin = (ifgmin + thrBrightRaw) / 2;

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
        printf("%s> grayThresh: %d (from %d), binsFixed: %d, binMin1.x: %d, binMax1.x: %d, thrBrightRaw: %d\n", __FUNCTION__,
            grayThresh, ifgmin, binsFixed, binMin1.x, binMax1.x, thrBrightRaw);

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
    cv::Scalar mean{0}, stddev{0};
    cv::meanStdDev(areas, mean, stddev);
    std::sort(areas.begin(), areas.end());
    //const folat  rstd = 4;  // (3 .. 5+);  Note: 3 convers ~ 96% of results, but we should extend those margins
    //const int  minSizeThreshLim = 0.56f * areas[0]
    const float  expPerimMin = matchStat.distAvg - 0.36f * matchStat.distStd;  // 0.3 .. 0.4
    //minSizeThresh = (max<int>(min<int>(mean[0] - 4 * stddev[0], 0.56f * areas[0]), areas[0] * 0.36f) + expPerimMin * expPerimMin) / 2;  // 0.56 area ~= 0.75 perimiter; 0.36 a ~= 0.6 p
    // Note: only DLC-tracked larvae are reqrured to be detected, so the thresholds can be strict
    //const auto  stdBounded = max(mean[0], stddev[0]);  // ATTENTION: for the (almost) empty images, std might be negative int (uninitialized)
    minSizeThresh = (max<int>(min<int>(max<int>(mean[0] - stddev[0], 0), 0.92f * areas[0]), areas[0] * 0.82f) + expPerimMin * expPerimMin) / 2;  // 0.56 area ~= 0.75 perimiter; 0.36 a ~= 0.6 p
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

    assert(grayThresh >= 0 && minSizeThresh >= 0 && maxSizeThresh >= 0 && "Unexpected resulting values");
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
                                      bool checkRoiBorders)
{
    // generate a scratch image
    Mat tmpImg = Mat::zeros(src.size(), src.type());
    
    // perform gray threshold
    Preprocessor::graythresh(src,grayThresh,tmpImg);
    
    // generate a contours container scratch
    contours_t contours;

    // calculate the contours
    Preprocessor::calcContours(tmpImg,contours);
    
    // check if contours overrun image borders (as well as ROI-borders, if ROI selected)
    Preprocessor::borderRestriction(contours, src, checkRoiBorders);
    
    // filter the contours
    Preprocessor::sizethreshold(contours, minSizeThresh, maxSizeThresh, acceptedContoursDst, biggerContoursDst);
}
