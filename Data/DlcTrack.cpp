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

bool Tracker::loadHDF5(const string& filename)
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

bool Tracker::loadCSV(const string& filename)
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
        // Process SCV header
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
            fprintf(stderr, "ERROR %s> Inconsistent size of rows for #%ul (nfr: %ul): %ul != %ul\n"
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
    return loadTrajects(rawVals, nlvs);
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

bool Tracker::loadTrajects(const Mat& rawVals, unsigned nlarvae, float confmin)
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

    // Load valid (non-nan) values that have reuqired confidance, rounding to Point<int>
    vector<float>  distances;
    using ValT = float;
    const unsigned  larvaCols = rawVals.cols / nlarvae;
    const unsigned  lvPtsMin = _matchParams.rLarvaPtsMin * (larvaCols / _larvaPtCols);  // Min larva points to accept the larva
    Larva  lv;
    for(int i = 0; i < rawVals.rows; ++i) {
        Larvae  larvae;
        const ValT* rvRow = rawVals.ptr<ValT>(i);
        Larva::Id  uid = 0;
        for(int j = 0; j < rawVals.cols; j += _larvaPtCols) {
            // Look-ahead reading of larva
            if(j % larvaCols == 0) {
                //if(!lv.points.empty())
                if(lv.points.size() >= lvPtsMin) {
                   lv.id = ++uid;  // Note: ids should start from 1, 0 indicates non-tracable larva
                   //cv::Scalar  center = cv::mean(lv);
                   //lv.center = Point(round(center.x), round(center.y))
                   lv.center = toPoint(cv::mean(lv.points));
                   larvae.push_back(lv);

                   // Store distances from the center
                   for(const auto& pt: lv.points)
                       distances.push_back(norm(lv.center - pt));
                }
                lv.clear();
            }
            // Each 3rd value is likelihood
            const auto lkh = rvRow[j+2];
            if(lkh < confmin || std::isnan(lkh))
                continue;
            lv.points.emplace_back(rvRow[j], rvRow[j+1]);
        }
        _trajects.push_back(larvae);
    }
    if(!distances.empty()) {
        cv::Scalar  mean, stddev;
        cv::meanStdDev(distances, mean, stddev);
        _matchStat.distAvg = mean[0];
        _matchStat.distStd = stddev[0];
    }

    printf("%s> larvaCols: %u, lvPtsMin: %u, _trajects: %ul\n", __FUNCTION__, larvaCols, lvPtsMin, _trajects.size());
    if(!_trajects.empty()) {
        const Larva&  lv = _trajects[0][0];
        printf("%s> #%u.center[t=0]: %d, %d; avg global dist: %f (SD: %f)\n", __FUNCTION__
            , lv.id, lv.center.x, lv.center.y, _matchStat.distAvg, _matchStat.distStd);
    }
    return true;
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

unsigned matchedLarva(const Larva::Points& contour, const Larvae& larvae, const MatchParams& mp, unsigned idHint)
{
    cv::Scalar mean, stddev;
    cv::meanStdDev(contour, mean, stddev);
    return matchedLarva(toPoint(mean), toPoint(stddev), larvae, mp);
}

unsigned matchedLarva(const Point& center, const Point& stddev, const Larvae& larvae, const MatchParams& mp, unsigned idHint)
{
    if(larvae.empty()) {
        //printf("%s> empty\n", __FUNCTION__);
        return 0;
    }
    const Larva  *res = nullptr;  // Closest larva
    double  dmin = std::numeric_limits<double>::max();
    // 1..3 * stddev
    const double  dmax = mp.rLarvaStdMax * norm(stddev);  // Maximal allowed distance beween the centers
    if(idHint && idHint <= larvae.size() && norm(larvae[idHint - 1].center - center) <= dmax) {
        res = &larvae[idHint - 1];
    } else {
        for(const auto& lv: larvae) {
            double dist = norm(lv.center - center);
            if(dist < dmin) {
                dmin = dist;
                res = &lv;
            }
        }
        if(dmin > dmax) {
            printf("WARNING %s> candidate #%u is omitted: dmin = %f > dmax = %f\n", __FUNCTION__, res->id, dmin, dmax);
            return 0;
        }
    }

    //printf("%s> %u\n", __FUNCTION__, res->id);
    return res->id;
}

} // dlc
