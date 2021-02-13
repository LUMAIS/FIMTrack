//! \brief DLC Tracking data being imported
//! It includes DLC-tracked trajectories for named points on several larvae
//!
//! Copyright (c)
//! \authr Artem Lutov <lua@lutan.ch>

//#include <algorithm>
#include <limits>
#include <fstream>  // CSV reading
#include <sstream>  // CSV reading
#include <unordered_set>  // CSV reading
#include <cmath>  // round()
#include <iostream>
#include <string>
//#include <h5cpp/all>
#include <H5Cpp.h>
//// Note: OpenCV HDF5 works only with the root attributes and datasets representable as Mat
//#include <opencv2/hdf/hdf5.hpp>  // HDF5 format
//#include <opencv2/imgproc.hpp>  // convexHull() to order point properly for the contour formation
//#include <iostream>
#include <cstdio>  // printf()
#include "DlcTrack.hpp"

//using std::binary_search;
//using cv::String;
//using std::cout;
//using std::endl;
//using namespace dlc;
using cv::Mat;
using cv::Rect;
using cv::norm;
using namespace H5;


namespace dlc {

bool importVideo(const string& vidName, const string& outpDir, const string& frameBaseName, const string& format)
{
    cv::VideoCapture cap(vidName);
    if(!cap.isOpened()) {
        fprintf(stderr, "ERROR %s> video file can not be opened: %s\n", __FUNCTION__, vidName.c_str());
        return false;
    }
    Mat frame;
    unsigned  nframes = cap.get(cv::CAP_PROP_FRAME_COUNT);
    uint8_t digs = 0;
    while(nframes) {
        ++digs;
        nframes /= 10;
    }
    for(unsigned i = 1; ; ++i) {
        cap.read(frame);  // >>
        if(frame.empty())
            break;
        string frname = std::to_string(i);
        if(frname.size() < digs)
            frname = string(digs - frname.size(), '0') + frname;
        frname = outpDir + '/' + frameBaseName + '-' + frname + '.' + format;
        if(!cv::imwrite(frname, frame)) {
            fprintf(stderr, "ERROR %s> can not save the output image: %s\n", __FUNCTION__, frname.c_str());
            return false;
        }
    }
    printf("%s> %s imported as '%s' images to the dir: %s\n", __FUNCTION__, vidName.c_str(), format.c_str(), outpDir.c_str());
    return true;
}

//bool cmpPoint (const point& a, const Point& b)
//{
//    a.y < b.y || (a.y == b.y && a.x < b.x);
//}

//bool cmpLarva (const Larva& larva, const Point& center)
//{
//    return cmpPoint(larva.center, center);
//}

const Rect Tracker::DFL_ROI = Rect(0, 0, USHRT_MAX, USHRT_MAX);

bool Tracker::loadHDF5(const string& filename, const Rect& roi)
{
    struct DataPoint {
        int64_t frame;
        vector<double> values;
    };

    H5File file(filename, H5F_ACC_RDONLY);
    const string  dsname("/df_with_missing/table");
    DataSet ds = file.openDataSet(dsname);

    // Read attributes
    Attribute hdr = ds.openAttribute("values_block_0_kind");
    string  hdrVal;
    hdr.read(StrType(PredType::C_S1), hdrVal);
    if (hdrVal.empty()) {
        fprintf(stderr, "ERROR %s> Larvae dataset does not descibe larvae in the attributes\n", __FUNCTION__);
        return false;
    }
    // Check the number of "\nVlarva1\n"-like values to indentify the numbe of tracking larvae
    unsigned nlvs = 0;  // The number of larvaes
    size_t  pos = 0;  // Char index in the header
    while(true) {
        pos = hdrVal.find("\nVlarva", pos);
        if(pos != string::npos) {
            ++nlvs;
            continue;
        }
        break;
    }

//    // Read Data
//    H5T_class_t type_class = dataset.getTypeClass();
//    DataSpace dataspace = dataset.getSpace();
//    int rank = dataspace.getSimpleExtentNdims();


    return false;

//    CompType hdf5DataPointType( sizeof(dataPoint) );
//    hdf5DataPointType.insertMember(MEMBER_TIME, 0, PredType::NATIVE_DOUBLE);
//    hdf5DataPointType.insertMember(MEMBER_SIGN, sizeof(double), PredType::NATIVE_DOUBLE);

//    // Retain old data when loading is failed
//    cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(filename);
//    const string  dsname("/df_with_missing/table");
//    if(!h5io->hlexists(dsname)) {
//        fprintf(stderr, "ERROR %s> Larvae dataset does not exist in the HDF5 file\n", __FUNCTION__);
//        h5io->close();
//        return false;
//    }
//    cv::String attr_str_name = "/df_with_missing/table/values_block_0_kind";
//    if (!h5io->atexists(attr_str_name)) {
//        fprintf(stderr, "ERROR %s> Larvae dataset does not descibe larvae in the attributes\n", __FUNCTION__);
//        h5io->close();
//        return false;
//    }
//    cv::String hdr;  // Table headers annotation
//    // Note: OpenCV HDF5 works only with the root attributes and datasets representable as Mat
//    h5io->atread(&hdr, attr_str_name);  // The attribute MUST exist, otherwise CV_Error() is called. Use atexists() to check if it exists beforehand.
////    printf("%s> Opening the HDF5 dataset...\n", __FUNCTION__);
//    // Check the number of "\nVlarva1\n"-like values to indentify the numbe of tracking larvae
//    unsigned nlvs = 0;  // The number of larvaes
//    size_t  pos = 0;  // Char index in the header
//    while(true) {
//        pos = hdr.find("\nVlarva", pos);
//        if(pos != string::npos) {
//            ++nlvs;
//            continue;
//        }
//        break;
//    }
//
//    Mat lsvs;  // Larvaes data
//    // Read larvaes dataset, which has a compound format
//    // int offset[2] = { 1, 2 };
//    h5io->dsread(lsvs, dsname, {0, 1});  // Read from the second element, because the first on is the frame number
//    h5io->close();
//    //    cout << "DS row 0:" << lsvs.at<float>(0) << endl;
//    //    cout << "DS row 1:" << lsvs.at<float>(1) << endl;
//
//    return loadTrajects(lsvs, nlvs);
}

bool Tracker::loadCSV(const string& filename, const Rect& roi)
{
    std::ifstream finp(filename);
    if(!finp.is_open()) {
        fprintf(stderr, "ERROR %s> Larvae CSV file can not be opened\n", __FUNCTION__);
        return false;
    }
    using val_t = float;
    constexpr static val_t  val_nan = std::numeric_limits<val_t>::quiet_NaN();
    using Vals = vector<val_t>;
    Vals  vals;
    vector<Vals>  lsvs;  // Larvaes data
    string  ln;
    string  val;
    unsigned  nhdr = 4;  // Lines in the header
    size_t iframe = 0;
    size_t rowVals = 0;  // The number of values in each row
    unsigned nlvs = 0;  // The number of larvaes
    while(getline(finp, ln)) {
        // Process CSV header
        if(nhdr) {
            if(nhdr == 3) {
                // Get the number of distinct larvae
                std::stringstream  sln(ln);
                std::unordered_set<string>  larvaNames;
                // Skip first columnt
                getline(sln, val, ',');
                while(getline(sln, val, ',')) {
                    ++rowVals;
                    larvaNames.emplace_hint(larvaNames.end(), val);
                 }
                nlvs = larvaNames.size();
            }
            --nhdr;
            continue;
        }
        // Process CSV body
        std::stringstream  sln(ln);
        vals.clear();
        // First column contains the frame id
        getline(sln, val, ',');
        const size_t nfr = std::stoul(val);
        // Adjust frame counter
        while(iframe < nfr) {
            lsvs.push_back(Vals(rowVals, val_nan));
            ++iframe;
        }
        // Load raw larvae data
        while(getline(sln, val, ',')) {
            try {
                vals.push_back(std::stof(val));
            } catch(std::invalid_argument& err) {
                vals.push_back(val_nan);
            }
        }
        // Consider that the last empty value is omitted on reading
        if(vals.size() == rowVals - 1)
            vals.push_back(val_nan);
        lsvs.push_back(vals);
        if(rowVals != vals.size()) {
            fprintf(stderr, "ERROR %s> Inconsistent size of rows for #%lu (nfr: %lu): %lu != %lu\n"
                , __FUNCTION__, iframe, nfr, vals.size(), rowVals);
            return false;
        }
        ++iframe;
    }

    if(lsvs.empty()) {
        fprintf(stderr, "ERROR %s> Larvae trajectories are not specified\n", __FUNCTION__);
        return false;
    }

    Mat  rawVals(0, lsvs[0].size(), cv::DataType<val_t>::type);
    for (auto& vals: lsvs)
    {
        // Make a temporary Mat row and add to lsvs _without_ data copy
        Mat valsView(1, lsvs[0].size(), cv::DataType<val_t>::type, vals.data());
        rawVals.push_back(valsView);
    }
    return loadTrajects(rawVals, nlvs, roi);
}

//bool Tracker::loadJSON(const string& filename)
//{
//    // JSON loading
//    FileStorage fs = FileStorage(_dlcTrackFile.toStdString(), FileStorage::READ, StringConstats::textFileCoding);
//    if (fs.isOpened()) {
//        _dlcTrack.clear();
//
//        _dlcTrack.loadHDF5();
//        FileNode meta = fs["metadata"];
//        // Read point names
//        PointNames pointNames;
//        meta["all_joints_names"] >> pointNames;
//        // Read the number of frames
//        int nframes;
//        meta["nframes"] >> nframes;
//        unsigned char frameDigs = 0;  // The number of digits in nframes to consider '0' padding
//        int i = nframes;
//        while(i > 0) {
//            frameDigs += 1;
//            i /= 10;
//        }
//
//        // Read points
//        LarvaeTrajectories  trajects;
//        trajects.resize(nframes);
//        for(unsigned ifr = 0; ifr < nframes; ++ifr) {
//            string siframe = std::to_string(ifr);
//            FileNode frame = fs[string("frame").append(string(frameDigs - siframe.length(), '0')).append(siframe)];
//            NamedLarvaePoints& nlpts = trajects[ifr];
//            //frame["coordinates"][0] >> nlpts;
//            nlpts.resize(pointNames.size());
//            for(unsigned ipt = 0; ipt < nlpts.size(); ++ipt)
//                frame["coordinates"][0][ipt]>>nlpts[ipt];
//        }
//
//        _dlcTrack.initialize(std::move(pointNames), std::move(trajects));
//    }
//}

bool Tracker::loadTrajects(const Mat& rawVals, unsigned nlarvae, const Rect& roi, float confmin)
{
    if(rawVals.empty() || rawVals.cols % (nlarvae * _larvaPtCols)) {  // nlarvae * (x, y, likelihood)
        fprintf(stderr, "ERROR %s> Invalid size of rawVals\n", __FUNCTION__);
        return false;
    }

    const int type = rawVals.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    if(chans != 1) {
        fprintf(stderr, "ERROR %s> Unexpected number of channels (dimensions) in rawVals\n", __FUNCTION__);
        return false;
    }

    switch (depth) {
    case CV_32F:
        break;  // OK
    case CV_64F:
        fprintf(stderr, "ERROR %s> float32 values are expected in rawVals instead of CV_64F\n", __FUNCTION__);
        return false;
        break;
    default:
        fprintf(stderr, "ERROR %s> Unexpected type of values in rawVals\n", __FUNCTION__);
        return false;
    }

    clear();

    // Historical Larvae to recover larva id swapped by DLC
    struct HistLarva {
        const Larva*  larva;
        unsigned  nframe;  //!< Frame number when the larva was present at the last time

        HistLarva(): larva(nullptr), nframe(-1)  {}
    };
    std::vector<HistLarva>  histLarvae;  // Historical Larvae
    const unsigned  histDistMax = 15;  // Max historical distance in frames to recover id: 10 .. 20
    const MatchParams  matchParams{1.5, 0.4};  // Defult matching parameters
    histLarvae.resize(nlarvae);

    // Load valid (non-nan) values that have reuqired confidance, rounding to Point<int>
    using ValT = float;
    const unsigned  larvaCols = rawVals.cols / nlarvae;
    const unsigned  lvPtsMin = _matchParams.rLarvaPtsMin * (larvaCols / _larvaPtCols);  // Min larva points to accept the larva
    Larva  lv;
    vector<float>  distances;
    distances.reserve(rawVals.rows * rawVals.cols / 2);  // Some raw larva are likely to be filtered out
    unsigned  nfopts = 0;  // The number of filtered out points of larva contours (invalid coordinates only, excluding the dependent points)
    unsigned  nfolvs = 0;  // The number of filtered out larvae

    _trajects.reserve(rawVals.rows);
    for(int i = 0; i < rawVals.rows; ++i) {
        Larvae  larvae;
        const ValT* rvRow = rawVals.ptr<ValT>(i);
        Larva::Id  uid = 0;
        for(int j = 0; j < rawVals.cols;) {
            // Look-ahead reading of larva
            // Each 3rd value is likelihood
            const auto lkh = rvRow[j+2];
            // Note: DLC sometimes estimate point coordinates out of the processing view,
            // resulting in either negative coordinates or ones exceeding the view frame.
            // Negative values can be filtered out at once, but the out of range positive values can be identified and handled only after loading the actual frame.
            // So, it makes sence
            //  || rvRow[j] < 0 rvRow[j+1] < 0
            const auto x = rvRow[j];
            const auto y = rvRow[j+1];
            // Validate larva points
            // Note: sometimes x or y is NaN for valid likelihood (e.g., vid_2 #4[0])
            if(lkh < confmin || std::isnan(lkh) || std::isnan(x) || std::isnan(y)
            || x < roi.x || x >= roi.x + roi.width || y < roi.y || y >= roi.y + roi.height)
                ++nfopts;
            else lv.points.emplace_back(round(x), round(y));
            j += _larvaPtCols;

            // Insert a larva to the container when all its points are handled
            if(j % larvaCols == 0) {
                // ATTENTION: ids are assigned to the original larvae even if they are empty and omitted, otherwise ids would be invalid
                lv.id = ++uid;  // Note: ids should start from 1, 0 indicates non-tracable larva
                if(lv.points.size() >= lvPtsMin) {
                    // Fetch the convex hull for the larva
                    lv.hull.reserve(lv.points.size() / 2);
                    cv::convexHull(lv.points, lv.hull);

                    lv.center = toPoint(cv::mean(lv.points));
                    // Ensure that ids are not swapped when the current larva is absent in the previous frame and at least one larva is missed.
                    Larva::Id  lid = 0;  // Recovered larva id, 0 means none

                    const auto&  hlvCur = histLarvae[lv.id - 1];
                    if(i - hlvCur.nframe > histDistMax || !hlvCur.larva) {
                        Larvae  tls;
                        tls.reserve(histLarvae.size() / 2);
                        for(auto& hl: histLarvae)
                            if(i - hl.nframe <= histDistMax && hl.larva)
                                tls.push_back(*hl.larva);
                        if(!tls.empty()) {
                            lid = matchedLarva(lv.hull, tls, matchParams, 0, true);
                            printf("%s> frame: %d, candidate larva for the id swapping correction: %u -> %u\n", __FUNCTION__
                                , i, lv.id, lid);
////#if defined(QT_DEBUG) || defined(_QT_DEBUG)
//                        // Trace id swapping-related data
//                        if(appeared) {
//                            printf("%s> larva #%u, %lu missedIds, %lu pastLarvae\n", __FUNCTION__
//                                , lv.id, missedIds.size(), pastLarvae.size());
//                            printf("  missedIds: ");
//                            for(auto mid: missedIds)
//                                printf(" %u", mid);
//                            puts("");  // Newline
//                            printf("  prevLarvae: ");
//                            for(const auto& ltr: _trajects.back())
//                                printf(" %u", ltr.id);
//                            puts("");  // Newline
//                        }
////#endif  // Detailed Tracing
                        }
                    }

                    //                    if(!missedIds.empty() && !_trajects.empty() && !_trajects.back().empty()) {
                    //                        // Fetch larvae that are missed now and existed in the previous frame, and check whether the current larva existed.
                    //                        pastLarvae.clear();
                    //                        auto ipl = _trajects.back().cbegin();
                    //                        for(auto mid: missedIds) {
                    //                            while(ipl != _trajects.back().cend() && ipl->id < mid)
                    //                                ++ipl;
                    //                            if(ipl == _trajects.back().cend())
                    //                                break;
                    //                            if(ipl->id == mid)
                    //                                pastLarvae.push_back(&*ipl++);
                    //                        }
                    //                        // Check whether the current larva was absent in the previous frame and appeared now
                    //                        bool appeared = false;
                    //                        while(ipl != _trajects.back().cend() && ipl->id < lv.id)
                    //                            ++ipl;
                    //                        if(ipl == _trajects.back().cend() || ipl->id != lv.id)
                    //                            appeared = true;

                    //                        // Ensure that id is not swapped for a larva the appearing larva
                    //                        if(appeared && pastLarvae.size()) {
                    //                            // Match with larvae that are missed but existed in the previous frame
                    //                            Larvae  tls;
                    //                            tls.reserve(pastLarvae.size());
                    //                            for(auto plv: pastLarvae)
                    //                                tls.push_back(*plv);
                    //                            lid = matchedLarva(lv.hull, tls, matchParams);
                    //                            printf("%s> candidate larva for the id swapping correction: %u -> %u\n", __FUNCTION__, lv.id, lid);
                    //                        }
                    //                    }

                    // Note: lavra points validation is already performed
                    if(lid) {
                        printf("WARNING %s> frame: %d, corrected larva id: %u -> %u\n", __FUNCTION__, i, lv.id, lid);
                        lv.id = lid;
                        // Insert the larva in a proper place
                        auto ilv = larvae.begin();
                        while(ilv != larvae.end() && ilv->id < lv.id)
                            ++ilv;
                        larvae.insert(ilv, lv);
                    } else larvae.push_back(lv);

                    // Store distances from the center
                    for(const auto& pt: lv.points)
                        distances.push_back(norm(lv.center - pt));
                } else ++nfolvs;
                lv.clear();
            }
        }
        // Update histLarvae
        for(const auto& clv: larvae) {
            auto&  hlv = histLarvae[clv.id - 1];
            hlv.larva = &clv;
            hlv.nframe = i;
        }

        _trajects.push_back(move(larvae));
    }
    assert(_trajects.size() == rawVals.rows && "Unexpected number of the loaded trajectories");
    if(!distances.empty()) {
        cv::Scalar  mean, stddev;
        cv::meanStdDev(distances, mean, stddev);
        _matchStat.distAvg = mean[0];
        _matchStat.distStd = stddev[0];
    }

    if(nfopts || nfolvs)
        printf("WARNING %s> filtered out on %d frames: %u larvae, %u larva points\n", __FUNCTION__
            , rawVals.rows, nfolvs, nfopts);

    printf("%s> larvaCols: %u, lvPtsMin: %u, trajects: %lu, confmin: %f, roi: (%d, %d, %d, %d)\n", __FUNCTION__
           , larvaCols, lvPtsMin, _trajects.size(), confmin, roi.x, roi.y, roi.width, roi.height);
    if(!_trajects.empty()) {
        unsigned  iframe = 0;  // Frame index
        for(const auto& tr: _trajects) {
            ++iframe;
            if(tr.empty())
                continue;
            const Larva&  lv = tr[0];
            printf("%s> #%u.center[t=%u]: (%d, %d); avg global dist: %f (SD: %f)\n", __FUNCTION__
                , lv.id, iframe-1, lv.center.x, lv.center.y, _matchStat.distAvg, _matchStat.distStd);
            break;
        }
        if(iframe >= _trajects.size())
            printf("WARNING %s> trajectories are empty: no valid larvae exist\n", __FUNCTION__);
    }
    return true;
}

void Tracker::filter(const Rect& roi)
{
    if(roi.width <= 0 || roi.height <= 0)
        return;

    size_t  npts = 0;  // The number of removed points
    const Point  marg{roi.x + roi.width, roi.y + roi.height};
    // Note: empty items of trajectories should not be erased  because they represent time points.
    // Each trajectory contains few larvae, so there is no sense to release memory for the empty items
    unsigned  ntr = 0;  // Trajectory item number
    for(auto& tr: _trajects) {
        for(auto ilv = tr.begin(); ilv != tr.end();) {
            // Check location of each larva center and remove each larva out of the ROI
            if(ilv->center.x < roi.x || ilv->center.x > marg.x
            || ilv->center.y < roi.y || ilv->center.y > marg.y) {
                npts += ilv->points.size();
                printf("%s> timePoint: %u, removing %lu points with the center (%d, %d) for the ROI: (%d + %d, %d + %d)\n", __FUNCTION__
                    , ntr, npts, ilv->center.x, ilv->center.y, roi.x, roi.width, roi.y, roi.height);
                ilv = tr.erase(ilv);
                continue;
            }

            // Remove all larva points out of the ROI
            for(auto ipt = ilv->points.begin(); ipt != ilv->points.end();) {
                if(ipt->x < roi.x || ipt->x > marg.x
                || ipt->y < roi.y || ipt->y > marg.y) {
                    printf("%s> removing (%d, %d)\n", __FUNCTION__, ipt->x, ipt->y);
                    ipt = ilv->points.erase(ipt);
                    ++npts;
                } else ++ipt;
            }
            if(ilv->points.empty())  // lv.points.size() < lvPtsMin
                ilv = tr.erase(ilv);
            else ++ilv;
            //assert(!lv.points.empty() && "All larva points can not be located ourside a frame");
        }
        ++ntr;
    }

    if(npts)
        printf("%s> removed %lu points for the ROI: (%d + %d, %d + %d)\n", __FUNCTION__, npts, roi.x, roi.width, roi.y, roi.height);
}

void Tracker::clear()
{
    _matchStat = {0};
    _trajects.clear();
}

// Function definitions
//namespace dlc {

cv::Point toPoint(const cv::Scalar& sv)
{
    return Point(round(sv[0]), round(sv[1]));
}

Rect getLarvaeRoi(const Larvae& larvae, const cv::Size& area, int span)
{
    // Identify an approximate foregroud ROI
    // Note: initially, store top-left and bottom-right points in the rect, and ther change the latter to height-width
    Rect  fgrect = {area.width, area.height, 0, 0};

    // Identify the foreground ROI as an approximaiton (convexHull) of the DLC-tracked larva contours
    for(const auto& lv: larvae) {
        for(const auto& pt: lv.hull) {
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
    if(!fgrect.empty()) {
        // Expand the foreground ROI with the statistical span
        //const int  span = matchStat.distAvg + matchStat.distStd;  // Note: we expend from the border points rather thatn from the center => *1..2 rather than *3
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
        if(fgrect.x + fgrect.width >= area.width)
            fgrect.width = area.width - fgrect.x;
        fgrect.height += dy + span;
        if(fgrect.y + fgrect.height >= area.height)
            fgrect.height = area.height - fgrect.y;
        //printf("%s> fgrect: (%d + %d of %d, %d + %d of %d), span: %d\n", __FUNCTION__, fgrect.x, fgrect.width, img.cols, fgrect.y, fgrect.height, img.rows, span);
    }

    return fgrect;
}

unsigned matchedLarva(const Larva::Points& contour, const Larvae& larvae, const MatchParams& mp, unsigned idHint, bool visInsp)
{
//    static const auto  wnds = {"matchedLarva> OvpCnt_OmitCand", "matchedLarva> Cnt_OmitCand"};   // Inspection Windows
//    static std::vector<bool>  wndActFlags(wnds.size(), false);
//    static auto showCvWnd = [](const cv::String& wnd, const Mat& img) {
//        auto iw = std::find(wnds.begin(), wnds.end(), wnd);
//        if(iw != wnds.end()) {
//            cv::imshow(wnd, img);
//            wndActFlags[std::distance(wnds.begin(), iw)] = true;
//        }
//    };
//    static auto safeDestroyCvWnd = [](const cv::String& wnd) {
//        auto iw = std::find(wnds.begin(), wnds.end(), wnd);
//        const unsigned  iwaf = std::distance(wnds.begin(), iw);
//        if(iwaf < wnds.size() && wndActFlags[iwaf]) {
//            cv::destroyWindow(wnd);
//            wndActFlags[iwaf] = false;
//        }
//    };

//#if !(defined(QT_DEBUG) || defined(_QT_DEBUG))
//    visInsp = false;  // Note: should be activated only for debugging to reduce cluttering
//#endif  // Release Build

    cv::Scalar mean, stddev;
    cv::meanStdDev(contour, mean, stddev);
    auto larva = matchedLarva(toPoint(mean), toPoint(stddev), larvae, mp, idHint, visInsp);
    if(larva)
        printf("WARNING %s> candidate #%u\n", __FUNCTION__, larva->id);
    // Ensure that more than a half of DLC contour is covered with the matching contour
    if(larva) {
        // Evaluate area of the DLC contour
        // ATTENTION: take convex hull from the larva points because they contain points inside the larva and those points are not ordered.
        // Note: OpenCV expects points to be ordered in contours, so convexHull() is used
        const Larva::Points&  larvaHull = larva->hull;
        const float areaLarv = cv::contourArea(larvaHull);  // cv::countNonZero(maskLarv);
        // Use preliminary soft validation of the size of the matched contour
        const float areaCont = cv::contourArea(contour); // cv::countNonZero(maskCont);  // cv::arcLength(contour, true)
        constexpr float  rLarvContMax = 1.8;  // 1.5 .. 2.2
        if(areaCont < rLarvContMax * areaLarv && areaLarv < rLarvContMax * areaCont) { // 3 .. 5
            // Identify the ROI that covers both contours
            Larva::Points  pts(contour);
            pts.insert(pts.end(), larvaHull.begin(), larvaHull.end());
            const Rect  roi = cv::boundingRect(pts);

            Mat  maskLarv = Mat::zeros(roi.size(), CV_8UC1);
            cv::drawContours(maskLarv, vector<dlc::Larva::Points>(1, larvaHull), 0, 0xFF, cv::FILLED, cv::LINE_8, cv::noArray(), 0, Point(-roi.x, -roi.y));
            // Intersect with the target contour
            Mat  maskCont = Mat::zeros(roi.size(), CV_8UC1);
            cv::drawContours(maskCont, vector<dlc::Larva::Points>(1, contour), 0, 0xFF, cv::FILLED, cv::LINE_8, cv::noArray(), 0, Point(-roi.x, -roi.y));
            cv::bitwise_and(maskLarv, maskCont, maskCont);
            const auto areaOvp = cv::countNonZero(maskCont);
            // Filter-out the match if the (DLC) larva is covered up to a half by the contour
            if(areaLarv >= 2.5 * areaOvp || areaLarv + areaCont >= 3 * areaOvp) {
               printf("WARNING %s> candidate #%u is omitted. areaLarv / areaOvp: %f \n", __FUNCTION__, larva->id, areaLarv / areaOvp);
               if(visInsp) {
                   cv::drawContours(maskCont, vector<dlc::Larva::Points>(1, contour), 0, 0xAA, 1, cv::LINE_8, cv::noArray(), 0, Point(-roi.x, -roi.y));
                   cv::drawContours(maskCont, vector<dlc::Larva::Points>(1, larvaHull), 0, 0x44, 1, cv::LINE_8, cv::noArray(), 0, Point(-roi.x, -roi.y));
                   cv::imshow("matchedLarva> OvpCnt_OmitCand#" + std::to_string(larva->id), maskCont);
                   //auto iwnd = wnds.begin();
                   //showCvWnd(*iwnd, maskCont);
                   //safeDestroyCvWnd(*++iwnd);
               }
               larva = nullptr;
            }
        } else {
            printf("WARNING %s> candidate #%u is omitted. areaLarv / areaCont: %f \n", __FUNCTION__, larva->id, areaLarv / areaCont);
            if(visInsp) {
                // Identify the ROI that covers both contours
                Larva::Points  pts(contour);
                pts.insert(pts.end(), larvaHull.begin(), larvaHull.end());
                const Rect  roi = cv::boundingRect(pts);

                Mat  maskLarv = Mat::zeros(roi.size(), CV_8UC1);
                cv::drawContours(maskLarv, vector<dlc::Larva::Points>(1, larvaHull), 0, 0xFF, cv::FILLED, cv::LINE_8, cv::noArray(), 0, Point(-roi.x, -roi.y));
                //cv::imshow("matchedLarva> Larva DLC", maskLarv);
                Mat  maskCont = Mat::zeros(roi.size(), CV_8UC1);
                cv::drawContours(maskCont, vector<dlc::Larva::Points>(1, contour), 0, 0xFF, cv::FILLED, cv::LINE_8, cv::noArray(), 0, Point(-roi.x, -roi.y));
                cv::bitwise_and(maskLarv, maskCont, maskCont);
                cv::drawContours(maskCont, vector<dlc::Larva::Points>(1, contour), 0, 0xAA, 1, cv::LINE_8, cv::noArray(), 0, Point(-roi.x, -roi.y));
                cv::drawContours(maskCont, vector<dlc::Larva::Points>(1, larvaHull), 0, 0x44, 1, cv::LINE_8, cv::noArray(), 0, Point(-roi.x, -roi.y));
                cv::imshow("matchedLarva> Cnt_OmitCand#" + std::to_string(larva->id), maskCont);
                //auto iwnd = wnds.begin();
                //safeDestroyCvWnd(*iwnd);
                //showCvWnd(*++iwnd, maskCont);
            }
            larva = nullptr;
        }
    }
    //else for(auto wnd: wnds)
    //    safeDestroyCvWnd(wnd);

    return larva ? larva->id : 0;
}

const Larva* matchedLarva(const Point& center, const Point& stddev, const Larvae& larvae, const MatchParams& mp, unsigned idHint, bool visInsp)
{
    if(larvae.empty()) {
        //printf("%s> empty\n", __FUNCTION__);
        return nullptr;
    }
    const Larva  *res = nullptr;  // Closest larva
    double  dmin = std::numeric_limits<double>::max();
    // 1..3 * stddev
    const double  dmax = mp.rLarvaStdMax * norm(stddev);  // Maximal allowed distance beween the centers; Note: rLarvaStdMax = 1.5 is the default value
    if(idHint && idHint <= larvae.size() && norm(larvae[idHint - 1].center - center) <= dmax) {
        res = &larvae[idHint - 1];
    } else {
        for(const auto& lv: larvae) {
            if(lv.points.empty())
                continue;
            double dist = norm(lv.center - center);
            if(dist < dmin) {
                dmin = dist;
                res = &lv;
            }
        }
        if(dmin > dmax) {
            printf("WARNING %s> candidate #%u is omitted: dmin = %f > dmax = %f\n", __FUNCTION__, res ? res->id : 0, dmin, dmax);
            return 0;
        }
    }

    //printf("%s> %u\n", __FUNCTION__, res->id);
    return res;
}

} // dlc
