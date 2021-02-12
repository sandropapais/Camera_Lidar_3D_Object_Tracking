
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator itVal = boundingBoxes.begin(); itVal != boundingBoxes.end(); ++itVal)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*itVal).roi.x + shrinkFactor * (*itVal).roi.width / 2.0;
            smallerBox.y = (*itVal).roi.y + shrinkFactor * (*itVal).roi.height / 2.0;
            smallerBox.width = (*itVal).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*itVal).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(itVal);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto itVal = it1->lidarPoints.begin(); itVal != it1->lidarPoints.end(); ++itVal)
        {
            // world coordinates
            float xw = (*itVal).x; // world position in m with x facing forward from sensor
            float yw = (*itVal).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL); // WINDOW_NORMAL, WINDOW_AUTOSIZE, WINDOW_FULLSCREEN
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1/frameRate; // time between two measurements in seconds
    // double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        //if (it->y < laneWidth/2 && it->y > -laneWidth/2)
        //{
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
        //}
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        //if (it->y < laneWidth/2 && it->y > -laneWidth/2)
        //{
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
        //}
    }

    // compute TTC from both measurements
    double relVel = (minXPrev - minXCurr) / dT;
    if (relVel > 0)
    {
        TTC = minXCurr / relVel;
    }
    else // no collision case
    {
        TTC = -1;
    }
    
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::multimap<int, int> bbMatches;

    double shrinkFactor = 0.10; // bounding box size reduction factor
    int featureMatchesThresh = 10; // minimum number of feature matches inside previous and current box
    int verboseFlg = 0; // flag to debug info

    // loop over all keypoint matches
    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        int kptInBoxPrevCount = 0;
        int kptInBoxCurrCount = 0;
        int boxIDPrev, boxIDCurr;

        // loop over previous bounding boxed
        for (vector<BoundingBox>::iterator itVal = prevFrame.boundingBoxes.begin(); itVal != prevFrame.boundingBoxes.end(); ++itVal)
        {
            // shrink bounding box slightly to avoid outliers
            cv::Rect smallerRoi;
            smallerRoi.x = (*itVal).roi.x + shrinkFactor * (*itVal).roi.width / 2.0;
            smallerRoi.y = (*itVal).roi.y + shrinkFactor * (*itVal).roi.height / 2.0;
            smallerRoi.width = (*itVal).roi.width * (1 - shrinkFactor);
            smallerRoi.height = (*itVal).roi.height * (1 - shrinkFactor);

            // check if keypoint is inside bounding box, if so add to box hit counter
            cv::Point2f posPrevKpt = prevFrame.keypoints[it->queryIdx].pt;
            if (smallerRoi.contains(posPrevKpt))
            {
                boxIDPrev = (*itVal).boxID;
                kptInBoxPrevCount += 1;
            }
        }        
        // loop over current bounding boxes
        for (vector<BoundingBox>::iterator itVal = currFrame.boundingBoxes.begin(); itVal != currFrame.boundingBoxes.end(); ++itVal)
        {
            // shrink bounding box slightly to avoid outliers
            cv::Rect smallerRoi;
            smallerRoi.x = (*itVal).roi.x + shrinkFactor * (*itVal).roi.width / 2.0;
            smallerRoi.y = (*itVal).roi.y + shrinkFactor * (*itVal).roi.height / 2.0;
            smallerRoi.width = (*itVal).roi.width * (1 - shrinkFactor);
            smallerRoi.height = (*itVal).roi.height * (1 - shrinkFactor);

            // check if keypoint is inside bounding box, if so add to box hit counter
            cv::Point2f posCurrKpt = currFrame.keypoints[it->trainIdx].pt;
            if (smallerRoi.contains(posCurrKpt))
            {
                boxIDCurr = (*itVal).boxID;
                kptInBoxCurrCount += 1;
            }
        }
        
        // If both matched kpts are inside exactly one box, then store box IDs in multimap
        if(kptInBoxPrevCount == 1 && kptInBoxCurrCount == 1)
        {
            bbMatches.insert({boxIDPrev,boxIDCurr});
        }   
    }

    // Cout number of map pairs & sort in order of best to worst & remove outliers

    // find set of unique keys (prev box ID) in multimap 
    std::set<int> uniqueKeys;
    for (auto const& pair: bbMatches) 
    {
        uniqueKeys.insert(pair.first);
    }
    
    // debug printout for box match candidates
    if (verboseFlg == 1)
    {
        std::cout << "[DEBUG] Bounding box match candidates: {BoxIDPrev,BoxIDCurr}, count: # features matched \n";
    }

    // loop through unique keys (prev box ID)
    for (auto itKey : uniqueKeys)
    {
        // loop through values (curr box ID) for each unique key (prev box ID)
        std::multiset<int> keyValues;
        std::set<int> keyValuesUnique;
        for (auto itVal = bbMatches.equal_range(itKey).first ; itVal != bbMatches.equal_range(itKey).second ; ++itVal)
        {
            // do stuff with each value: (*itVal).second;
            keyValues.insert((*itVal).second);
            keyValuesUnique.insert((*itVal).second);
        }

        int count_max = 0;
        int value_max = 0;
        for (auto itValUniq : keyValuesUnique)
        {
            int count = keyValues.count(itValUniq);
            if (count > count_max)
            {
                count_max = count;
                value_max = itValUniq;
            }
            if (verboseFlg == 1)
            {
                std::cout << "{" << itKey << "," << itValUniq << "}, count:" << count << "\n";
            }
        }
        if (count_max > featureMatchesThresh)
        {
            bbBestMatches.insert({itKey,value_max});
        }
    }

    // debug printout for successful box matches
    if (verboseFlg == 1)
    {
        std::cout << "[DEBUG] Bounding box matches selected: {BoxIDPrev,BoxIDCurr}\n";
        for (auto const& pair: bbBestMatches) 
        {
            std::cout << "{" << pair.first << "," << pair.second << "}\n";
        }
    }

}
