//! \brief DLC Tracking data being imported
//! It includes DLC-tracked trajectories for named points on several larvae
//!
//! Copyright (c)
//! \authr Artem Lutov <lua@lutan.ch>

#ifndef DLCTRACK_HPP
#define DLCTRACK_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


namespace dlc {

using std::string;
using std::vector;
using cv::Point;

//! \brief Import video into frame images
//! \param vidName  - videofile path
//! \param outpDir  - output directory
//! \param frameBaseName  - base name of the outputting frames
//! \param format  - format of the output images: "png", "tiff"
//! \return The video is successfully imported
bool importVideo(const string& vidName, const string& outpDir, const string& frameBaseName="frame", const string& format="png");

struct Larva {
    using Points = std::vector<Point>;
    using Id = unsigned;

    Id id;
    Point  center;  //!< Center point
    Points  points;  //!< Filtered larva points
    //LarvaPoints  hull;  //!< Convex hull

    Larva(): id(0), center(), points()  {}

    void clear()
    {
        id = 0;
        center = Point();
        points.clear();
    }
};
using Larvae = vector<Larva>;
using LarvaeTrajects = vector<Larvae>;

Point toPoint(const cv::Scalar& sv);

struct MatchParams {
    float  rLarvaStdMax;  //!< max ratio of the Standard Deviation of the larva center on matching, ~ 1..3
    float  rLarvaPtsMin;  //!< min ratio of the larva points to acccept the DLC-tracked larva, (0, 1]
};

struct MatchStat {
    float  distAvg;  //!< average distance from the larva center to the DLC-tracked points
    float  distStd;  //!< sandard deviation of the distance distance
};

//! \brief Evaluate id >= 1 of the matched larva if any, otherwise return 0
//! \param contour  - contour to be matched
//! \param larvae  - ordered larvae to be matched
//! \param idHint  - hinted id for the fast matching, 0 means disabled
//! \param mp  - parameters for the larvae matching
unsigned matchedLarva(const Larva::Points& contour, const Larvae& larvae, const MatchParams& mp, unsigned idHint=0);

//! \brief Evaluate id >= 1 of the matched larva if any, otherwise return 0
//! \param center  - center point
//! \param stddev  - standard deviation of the points
//! \param larvae  - ordered larvae to be matched
//! \param idHint  - hinted id for the fast matching, 0 means disabled
//! \param mp  - parameters for the larvae matching
unsigned matchedLarva(const Point& center, const Point& stddev, const Larvae& larvae, const MatchParams& mp, unsigned idHint=0);

class Tracker {
    constexpr static  unsigned  _larvaPtCols = 3;  //!< The number of columns (fields) in each larva point
    const MatchParams  _matchParams;
    MatchStat   _matchStat;
    //! Tracking frames that include trajectories; ordered as by the larvae centers OpenCV contours
    LarvaeTrajects  _trajects;
//    LarvaeTrajectories  _trajects;
//    PointNames  _pointNames;
protected:
    //! \brief Load filtered larvae trajectories from the raw values
    //! \param rawVals  - raw values in the form (x, y, likelihood) * larvae_points * nlarvae
    //! \param nlarvae  - the number of larvae
    //! \param confmin  - minimal confidance of coordinates to be accepted
    //! \return whether the trajectories loaded successfully
    bool loadTrajects(const cv::Mat& rawVals, unsigned nlarvae, float confmin=0.5);
public:
    bool active;  //!< Whether the stored data should be used or omitted by the external client
    bool autoThreshold;  //!< Whether to evaluate the brightness (gray) and larvae area thresholds or use manual values

    //! \brief Tracker constructor
    //! \param rLarvaStdMax  - max ratio of the Standard Deviation of the larva center on matching, ~ 1..3
    //! \param rLarvaPtsMin  - min ratio of the larva points to acccept the DLC-tracked larva, (0, 1]
    Tracker(float rLarvaStdMax=1.5, float rLarvaPtsMin=0.4): _matchParams{rLarvaStdMax, rLarvaPtsMin}, _matchStat{0}, _trajects(), active(true)  {}

    //! \brief Load larvae trajectories
    //! \param filename  - hdf5 file name, containing larvae trajectories
    //! \return Whether the file is loaded successfully
    bool loadHDF5(const string& filename);

    //! \brief Load larvae trajectories
    //! \param filename  - csv file name, containing larvae trajectories
    //! \return Whether the file is loaded successfully
    bool loadCSV(const string& filename);

    //! \brief Clear stored data
    void clear();

//    void initialize(PointNames&& names, LarvaeTrajectories&& trajects)
//    {
//        _trajects = std::move(trajects);
//        _pointNames = std::move(names);
//    }

//    const PointNames& pointNames() const  { return _pointNames; }
//    const LarvaeTrajectories& trajects() const  { return _trajects; }

//    bool empty() const  { return _trajects.empty(); }
//    void clear()
//    {
//        _trajects.clear();
//        _pointNames.clear();
//    }

    //! DLC-tracked larvae at the specified frame (time point)
    const Larvae& larvae(unsigned frameNum) const  { return _trajects.at(frameNum); }

    //! \brief The number of frames in the tracking
    unsigned size() const  { return _trajects.size(); }

    const MatchParams& matchParams() const  { return _matchParams; }
    const MatchStat& matchStat() const  { return _matchStat; }

//    float rStdMax() const  { return _matchParams.rLarvaStdMax; }

//    float rPtsMin() const  { return _matchParams.rLarvaPtsMin; }
};

}  // dlc

#endif // DLCTRACK_HPP
