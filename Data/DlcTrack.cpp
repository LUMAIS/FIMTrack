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
//#include <h5cpp/all>
#include <H5Cpp.h>
//// Note: OpenCV HDF5 works only with the root attributes and datasets representable as Mat
//#include <opencv2/hdf/hdf5.hpp>  // HDF5 format
//#include <opencv2/imgproc.hpp>  // convexHull() to order point properly for the contour formation
//#include <iostream>
#include <cstdio>  // printf()
#include "DlcTrack.hpp"

using std::string;
using std::vector;
//using std::binary_search;
//using cv::String;
//using std::cout;
//using std::endl;
using cv::Point;
//using namespace dlc;
using namespace H5;


namespace dlc {

//bool cmpPoint (const point& a, const Point& b)
//{
//    a.y < b.y || (a.y == b.y && a.x < b.x);
//}

//bool cmpLarva (const Larva& larva, const Point& center)
//{
//    return cmpPoint(larva.center, center);
//}

bool Tracker::loadHDF5(const std::string& filename)
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
        fprintf(stderr, "ERROR loadHDF5: Larvae dataset does not descibe larvae in the attributes\n");
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
//        fprintf(stderr, "ERROR loadHDF5: Larvae dataset does not exist in the HDF5 file\n");
//        h5io->close();
//        return false;
//    }
//    cv::String attr_str_name = "/df_with_missing/table/values_block_0_kind";
//    if (!h5io->atexists(attr_str_name)) {
//        fprintf(stderr, "ERROR loadHDF5: Larvae dataset does not descibe larvae in the attributes\n");
//        h5io->close();
//        return false;
//    }
//    cv::String hdr;  // Table headers annotation
//    // Note: OpenCV HDF5 works only with the root attributes and datasets representable as Mat
//    h5io->atread(&hdr, attr_str_name);  // The attribute MUST exist, otherwise CV_Error() is called. Use atexists() to check if it exists beforehand.
////    printf("Opening the HDF5 dataset...\n");
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
//    cv::Mat lsvs;  // Larvaes data
//    // Read larvaes dataset, which has a compound format
//    // int offset[2] = { 1, 2 };
//    h5io->dsread(lsvs, dsname, {0, 1});  // Read from the second element, because the first on is the frame number
//    h5io->close();
//    //    cout << "DS row 0:" << lsvs.at<float>(0) << endl;
//    //    cout << "DS row 1:" << lsvs.at<float>(1) << endl;
//
//    return loadTrajects(lsvs, nlvs);
}

bool Tracker::loadCSV(const std::string& filename)
{
    std::ifstream finp(filename);
    if(!finp.is_open()) {
        fprintf(stderr, "ERROR loadCSV: Larvae CSV file can not be opened\n");
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
        const auto nfr = std::stol(val);
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
            fprintf(stderr, "ERROR loadCSV: Inconsistent size of rows for #%d (nfr: %d): %d != %d\n", iframe, nfr, vals.size(), rowVals);
            return false;
        }
        ++iframe;
    }

    if(lsvs.empty()) {
        fprintf(stderr, "ERROR loadCSV: Larvae trajectories are not specified\n");
        return false;
    }

    cv::Mat  rawVals(0, lsvs[0].size(), cv::DataType<val_t>::type);
    for (auto& vals: lsvs)
    {
        // Make a temporary cv::Mat row and add to lsvs _without_ data copy
        cv::Mat valsView(1, lsvs[0].size(), cv::DataType<val_t>::type, vals.data());
        rawVals.push_back(valsView);
    }
    return loadTrajects(rawVals, nlvs);
}

bool Tracker::loadTrajects(const cv::Mat& rawVals, unsigned nlarvae, float confmin)
{
    if(rawVals.empty() || rawVals.cols % (nlarvae * _larvaPtCols)) {  // nlarvae * (x, y, likelihood)
        fprintf(stderr, "ERROR loadTrajects: Invalid size of rawVals\n");
        return false;
    }

    const int type = rawVals.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    if(chans != 1) {
        fprintf(stderr, "ERROR loadTrajects: Unexpected number of channels (dimensions) in rawVals\n");
        return false;
    }

    switch (depth) {
    case CV_32F:
        break;  // OK
    case CV_64F:
        fprintf(stderr, "ERROR loadTrajects: float32 values are expected in rawVals instead of CV_64F\n");
        return false;
        break;
    default:
        fprintf(stderr, "ERROR loadTrajects: Unexpected type of values in rawVals\n");
        return false;
    }

    // Load valid (non-nan) values that have reuqired confidance, rounding to Point<int>
    using ValT = float;
    const unsigned  larvaCols = rawVals.cols / nlarvae;
    const unsigned  lvPtsMin = _matchParams.rLarvaPtsMin * (larvaCols / _larvaPtCols);  // Min larva points to accept the larva
    Larva  lv;
    Larva::Id  uid = 0;
    for(int i = 0; i < rawVals.rows; ++i) {
        Larvae  larvae;
        const ValT* rvRow = rawVals.ptr<ValT>(i);
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

    printf("loadTrajects: larvaCols: %d, lvPtsMin: %d, _trajects: %d\n", larvaCols, lvPtsMin, _trajects.size());
    if(!_trajects.empty()) {
        const Larva&  lv = _trajects[0][0];
        printf("loadTrajects: center[0][0] #%d: %d, %d\n", lv.id, lv.center.x, lv.center.y);
    }
    return true;
}

// Function definitions
//namespace dlc {

cv::Point toPoint(const cv::Scalar& sv)
{
    return Point(round(sv[0]), round(sv[1]));
}

unsigned matchedLarva(const Larva::Points& contour, const Larvae& larvae, const MatchParams& mp)
{
    cv::Scalar mean, stddev;
    cv::meanStdDev(contour, mean, stddev);
    return matchedLarva(toPoint(mean), toPoint(stddev), larvae, mp);
}

unsigned matchedLarva(const Point& center, const Point& stddev, const Larvae& larvae, const MatchParams& mp)
{
    if(larvae.empty())
        return 0;
    const Larva  *res = nullptr;  // Closest larva
    double  dmin = std::numeric_limits<double>::max();
    for(const auto& lv: larvae) {
        double dist = cv::norm(lv.center - center);
        if(dist < dmin) {
            dmin = dist;
            res = &lv;
        }
    }
    // 1..3 * stddev
    const double  dmax = mp.rLarvaStdMax * cv::norm(stddev);  // Maximal allowed distance beween the centers
    if(dmin > dmax) {
        printf("WARNING matchedLarva: candidate id_dlc = %d is omitted: dmin = %f > dmax = %f\n", res->id, dmin, dmax);
        return 0;
    }

    return res->id;
}

} // dlc
