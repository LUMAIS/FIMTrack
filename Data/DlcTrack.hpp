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

using std::vector;
using PointNames = vector<std::string>;
using LarvaePoints = vector<cv::Point>;
using NamedLarvaePoints = vector<LarvaePoints>;
using LarvaeTrajectories = vector<NamedLarvaePoints>;

class DlcTrack
{
    LarvaeTrajectories  _trajects;
    PointNames  _pointNames;
public:
    bool active;  //! \brief Whether the stored data should be used or omitted by the external client

    DlcTrack(): active(true)  {}

    void initialize(PointNames&& names, LarvaeTrajectories&& trajects)
    {
        _trajects = std::move(trajects);
        _pointNames = std::move(names);
    }

    const PointNames& pointNames() const  { return _pointNames; }
    const LarvaeTrajectories& trajects() const  { return _trajects; }

    bool empty() const  { return _trajects.empty(); }
    void clear()
    {
        _trajects.clear();
        _pointNames.clear();
    }
};

#endif // DLCTRACK_HPP
