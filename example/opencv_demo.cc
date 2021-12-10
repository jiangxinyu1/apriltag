/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.
This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#include <iostream>
#include <string>
#include "/home/xinyu/cleanRobot/thirdparty/include/opencv2/core/core.hpp"
#include "/home/xinyu/cleanRobot/thirdparty/include/opencv2/imgproc/imgproc.hpp"
#include "/home/xinyu/cleanRobot/thirdparty/include/opencv2/imgcodecs/imgcodecs.hpp"
#include "/home/xinyu/cleanRobot/thirdparty/include/opencv2/highgui/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ceres/ceres.h"
#include "ceres/local_parameterization.h"
#include "ceres/autodiff_cost_function.h"

extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagCustom16h5.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
#include "common/getopt.h"
#include "common/getopt.h"
#include "common/image_u8.h"
#include "common/image_u8x4.h"
#include "common/pjpeg.h"
#include "apriltag_pose.h"
}


using namespace std;
using namespace cv;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

#define TIME_STATISTICS
#define USE_TAG_BOARD
// #define USE_CHARGE_TAG


#ifdef USE_TAG_BOARD
const double tagCenterDistance = 0.2015;
const double tagCenterHeight = 0.051;
const double pointZ = 0.25;
#endif // USE_TAG_BOARD

#ifdef USE_CHARGE_TAG
const double tagCenterDistance = 0.204;
const double tagCenterHeight = 0.065;
const double pointZ = 0.25;
#endif // USE_CHARGE_TAG

// const double k1 =-0.338011, k2 = 0.130450, p1 = 0.000287, p2 =0.000001 ,k3=  -0.024906;
// const double fx = 934.166126, fy = 935.122766, cx = 960.504061-300, cy =  562.707915-200;

const double k1 =-0.337591, k2 = 0.125105, p1 = 0.000421416, p2 =-0.000218,k3=  -0.022025;
const double fx = 933.213000, fy = 934.140000, cx = 967.549-300, cy =  562.393-200;

// const int leftTagId = 6;
// const int rightTagId = 3;

const int leftTagId = 19;
const int rightTagId = 2;
const std::string path_prefix = "/data/test_1208_new/";

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ReprojectCostFunctor{
public:
    ReprojectCostFunctor(const Eigen::Vector3d& tag_point,const Eigen::Vector3d& real_point, const Eigen::Matrix3d& KMat) : 
                                          tag_point_(tag_point),real_point_(real_point),KMat_(KMat){}

    template <typename T>
    bool operator() (const T* const q,const T* const t, T* residual) const 
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_(t);
        Eigen::Map<const Eigen::Quaternion<T>> quaternion1(q);
        Eigen::Matrix<T,3,3> R_ = quaternion1.toRotationMatrix();
        Eigen::Matrix<T,3,1> tmp1;
        tmp1= KMat_*(R_* real_point_ + t_);
        residual[0] = tag_point_[0]  - tmp1(0,0)/tmp1(2,0);
        residual[1]= tag_point_[1] - tmp1(1,0)/tmp1(2,0);
        return true;
    } // operator ()

private:
    Eigen::Vector3d tag_point_;
    Eigen::Vector3d real_point_;
    Eigen::Matrix3d KMat_;
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class distortedCostFunctor{
public:
    distortedCostFunctor(cv::Point2d &point_distortion,const Eigen::Matrix3d& KMat) : point_distortion_(point_distortion),KMat_(KMat){}

    template <typename T>
    bool operator() (const T* const point, T* residual) const 
    {
        T x1 = point[0];
        T y1 = point[1];
        T r2 =x1*x1+y1*y1;
        T r4 =r2*r2;
        T r6 =r2*r2*r2;
        T res1 = point_distortion_.x - x1*(1.0+(T)k1*r2+(T)k2*r4+(T)k3*r6)+2.0*(T)p1*x1*y1+(T)p2*(r2+2.0*x1*x1);
        T res2 = point_distortion_.y - y1*(1.0+(T)k1*r2+(T)k2*r4+(T)k3*r6)+(T)p1*(r2+2.0*y1*y1)+2.0*(T)p2*x1*y1;
        residual[0] = res1;
        residual[1] = res2;
        return true;
    } // operator ()

private:
    const cv::Point2d &point_distortion_;
    Eigen::Matrix3d KMat_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CostFunctor{
public:
    // 构造函数
    CostFunctor(const Eigen::Vector3d& tag_point,  const  Eigen::Vector3d& tag_center_1_point, const  Eigen::Vector3d& tag_center_2_point,
                          const Eigen::Vector3d& real_point, const Eigen::Matrix3d& KMat, const int tagFlag) :
        tag_point_(tag_point), tag_center_1_point_(tag_center_1_point), tag_center_2_point_(tag_center_2_point),real_point_(real_point),KMat_(KMat) ,tagFlag_(tagFlag){}

    // 定义残差项计算方法
    template <typename T>
    bool operator() (const T* const q_1,const T* const t_1, const T* const q_2,const T* const t_2, T* residual) const 
    {
        // 将数据组织旋转矩阵
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t1(t_1);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t2(t_2);
        Eigen::Map<const Eigen::Quaternion<T>> quaternion1(q_1);
        Eigen::Map<const Eigen::Quaternion<T>> quaternion2(q_2);
        Eigen::Matrix<T,3,3> R1 = quaternion1.toRotationMatrix();
        Eigen::Matrix<T,3,3> R2 = quaternion2.toRotationMatrix();
        /////////////////////////////////////////////////////////////////////////////////
        //  1 计算tag1和tag2的重投影残差
        Eigen::Matrix<T,3,1> tmp1;
        if ( tagFlag_ == 1 )
        {
            tmp1= KMat_*(R1* real_point_ + t1);
        }
        else if ( tagFlag_ == 2 )
        {
            tmp1= KMat_*(R2* real_point_ + t2);
        }
        T r1[2];
        r1[0] = tag_point_[0]  - tmp1(0,0)/tmp1(2,0);
        r1[1]= tag_point_[1] - tmp1(1,0)/tmp1(2,0);
        /////////////////////////////////////////////////////////////////////////////////
        //  2 两个tag系轴之间的夹角应为0
        T r2[3];
        Eigen::Matrix<double,3,1> ttz (0,0,1);
        Eigen::Matrix<double,3,1> ttx (1,0,0);
        Eigen::Matrix<double,3,1> tty (0,1,0);
        r2[0] = ((R1*ttz).cross(R2*ttz)).norm();
        r2[1] = ((R1*tty).cross(R2*tty)).norm();
        r2[2] = ((R1*ttx).cross(R2*ttx)).norm();
        /////////////////////////////////////////////////////////////////////////////////
        //  3 Camera系下，tag1 的每个点（角点与中心点）与 tag2 的中心点组成的向量与 tag2的pose垂直
        Eigen::Matrix<T,3,1> TagPoint_camera;
        Eigen::Matrix<T,3,1> TagCenterPoint_camera;
        Eigen::Matrix<T,3,1> vec_;
        T r3;
        if ( tagFlag_ == 1 )
        {
            TagPoint_camera = R1*real_point_+t1;
            TagCenterPoint_camera = R2*Eigen::Vector3d(0,0,0)+t2;
            vec_ = TagPoint_camera - TagCenterPoint_camera;
            r3 = (vec_.transpose()*(R2*ttz));
        }
        if ( tagFlag_ == 2 )
        {
            // 使用对应的tag系下的点，计算当前点的深度值
            TagPoint_camera = R2*real_point_+t2;
            TagCenterPoint_camera = R1*Eigen::Vector3d(0,0,0)+t1;
            vec_ = TagPoint_camera - TagCenterPoint_camera;
            r3 = (vec_.transpose()*(R1*ttz));
        }
        /////////////////////////////////////////////////////////////////////////////////
        //  4  两个tag系中心之间的距离
        T r4;
        r4 = (t2-t1).norm() - tagCenterDistance;
        // T r5;
        // Eigen::Matrix<T,1,1> e_y = (t1-t2).transpose()*Eigen::Vector3d(0,1,0);
        // r5 = e_y.transpose()*e_y;

        residual[0] = r1[0] ;// 重投影误差第一项
        residual[1] = r1[1]; // 重投影误差第二项
        residual[2] = r2[0]*100000.0; // z轴
        residual[3] = r2[1]*10000.0; // y轴
        residual[4] = r2[2]*10000.0; // x轴
        residual[5] = r3*100000.0; // 10点共面
        residual[6] = r4*1000.0; // tag中心距离
        // residual[7] = r5*1000.0; // 高度一致
        return true;
    }

private:
    const Eigen::Vector3d tag_point_;
    const Eigen::Vector3d tag_center_1_point_; // 图像系下中心点（u,v,1）
    const Eigen::Vector3d tag_center_2_point_;
    const Eigen::Vector3d real_point_;
    Eigen::Matrix3d KMat_;
    int tagFlag_;
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct distortion_uv_4{
    int u_;
    int v_;
    uint32_t weight_11;
    uint32_t weight_12;
    uint32_t weight_21;
    uint32_t weight_22;
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct distortion_uv{
    double u_;
    double v_;
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void myImageDistorted(cv::Mat &src , cv::Mat &image_undistort);
bool SaveImage(const cv::Mat& img, std::string& absolutePath);
void YU12toRGB( std::string &yuv_file_path,cv::Mat &rgb_Img,const int w ,const int h ,bool blSave);
int64_t getTime();
void YUV4202GRAY_CV_SAVE(std::string &inputPath ,cv::Mat&grayImg,int width,int height);
void preBuildDistortedLookupTable(std::vector<std::vector<distortion_uv >> &lookupTable,const int width, const int height);
void myImageDistorted(cv::Mat &src , cv::Mat &image_undistort,const std::vector<std::vector<distortion_uv>> &lookupTable);
cv::Mat ComputeH(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
void homography_compute3(double c[4][4] , Eigen::Matrix3d &H);
void poseOptimization(const std::vector<Eigen::Vector3d>& tag1_points, const std::vector<Eigen::Vector3d>& tag2_points,
                                         const Eigen::Matrix3d &K, Eigen::Matrix3d & R1, Eigen::Vector3d & t1, Eigen::Matrix3d & R2, Eigen::Vector3d & t2 );
void singleTagPoseOptimization(const std::vector<Eigen::Vector3d>& tag1_points,  const Eigen::Matrix3d &K, Eigen::Matrix3d & R1, Eigen::Vector3d & t1);                      
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void homography_compute3(double c[4][4] , Eigen::Matrix3d &H) 
{
    double A[] =  {
            c[0][0], c[0][1], 1,       0,       0, 0, -c[0][0]*c[0][2], -c[0][1]*c[0][2], c[0][2],
                  0,       0, 0, c[0][0], c[0][1], 1, -c[0][0]*c[0][3], -c[0][1]*c[0][3], c[0][3],
            c[1][0], c[1][1], 1,       0,       0, 0, -c[1][0]*c[1][2], -c[1][1]*c[1][2], c[1][2],
                  0,       0, 0, c[1][0], c[1][1], 1, -c[1][0]*c[1][3], -c[1][1]*c[1][3], c[1][3],
            c[2][0], c[2][1], 1,       0,       0, 0, -c[2][0]*c[2][2], -c[2][1]*c[2][2], c[2][2],
                  0,       0, 0, c[2][0], c[2][1], 1, -c[2][0]*c[2][3], -c[2][1]*c[2][3], c[2][3],
            c[3][0], c[3][1], 1,       0,       0, 0, -c[3][0]*c[3][2], -c[3][1]*c[3][2], c[3][2],
                  0,       0, 0, c[3][0], c[3][1], 1, -c[3][0]*c[3][3], -c[3][1]*c[3][3], c[3][3],
    };

    double epsilon = 1e-10;

    // Eliminate.
    for (int col = 0; col < 8; col++) {
        // Find best row to swap with.
        double max_val = 0;
        int max_val_idx = -1;
        for (int row = col; row < 8; row++) {
            double val = fabs(A[row*9 + col]);
            if (val > max_val) {
                max_val = val;
                max_val_idx = row;
            }
        }

        if (max_val < epsilon) {
            fprintf(stderr, "WRN: Matrix is singular.\n");
        }

        // Swap to get best row.
        if (max_val_idx != col) {
            for (int i = col; i < 9; i++) {
                double tmp = A[col*9 + i];
                A[col*9 + i] = A[max_val_idx*9 + i];
                A[max_val_idx*9 + i] = tmp;
            }
        }
        // Do eliminate.
        for (int i = col + 1; i < 8; i++) {
            double f = A[i*9 + col]/A[col*9 + col];
            A[i*9 + col] = 0;
            for (int j = col + 1; j < 9; j++) {
                A[i*9 + j] -= f*A[col*9 + j];
            }
        }
    }
    // Back solve.
    for (int col = 7; col >=0; col--) {
        double sum = 0;
        for (int i = col + 1; i < 8; i++) {
            sum += A[col*9 + i]*A[i*9 + 8];
        }
        A[col*9 + 8] = (A[col*9 + 8] - sum)/A[col*9 + col];
    }
    H << A[8], A[17], A[26], A[35], A[44], A[53], A[62], A[71], 1;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void preBuildDistortedLookupTable(std::vector<std::vector<distortion_uv >> &lookupTable,const int width, const int height)
{
    // 畸变参数
    // double k1 =-0.338011, k2 = 0.130450, p1 = 0.000287, p2 =0.000001 ,k3=  -0.024906;
    // // 内参 
    // double fx = 934.166126/2, fy = 935.122766/2, cx = 1360/2/2, cy =  680/2/2;
    // 初始化
    std::vector<distortion_uv> rows_(width);
    lookupTable.resize(height,rows_);
    for (int v = 0 ; v < height; v++)
    {
        for (int u = 0 ; u < width; u++)
        {
            double u_distorted = 0, v_distorted = 0;
            double x1,y1,x_distorted,y_distorted;
            // 1 计算空白图像的每个点在归一化平面上的坐标（x1,y1）
            x1 = (u-cx)/fx;
            y1 = (v-cy)/fy;
            double r2;
            //  2 由畸变参数计算每个点发生畸变后在归一化平面的对应坐标 (x_distorted,y_distorted)
            r2 = pow(x1,2)+pow(y1,2);
            x_distorted  = x1*(1+k1*r2+k2*pow(r2,2)+k3*pow(r2,3))+2*p1*x1*y1+p2*(r2+2*x1*x1);
            y_distorted = y1*(1+k1*r2+k2*pow(r2,2)+k3*pow(r2,3))+p1*(r2+2*y1*y1)+2*p2*x1*y1;
            //  3 将畸变后的点由内参矩阵投影到像素平面,得到该点在输入的带有畸变图像上的位置 
            u_distorted = fx*x_distorted+cx;
            v_distorted = fy*y_distorted+cy;
            distortion_uv tmp;
            tmp.u_ = u_distorted;
            tmp.v_ = v_distorted;
            lookupTable[v][u] = tmp;
        }// for 
    }// for 
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void preBuildDistortedLookupTable(std::vector<std::vector<distortion_uv_4>> &lookupTable,const int width, const int height)
{
    std::vector<distortion_uv_4> rows_(width);
    lookupTable.resize(height,rows_);
    for (int v = 0 ; v < height; v++)
    {
        for (int u = 0 ; u < width; u++)
        {
            double u_distorted = 0, v_distorted = 0;
            double x1,y1,x_distorted,y_distorted;
            // 1 计算空白图像的每个点在归一化平面上的坐标（x1,y1）
            x1 = (u-cx)/fx;
            y1 = (v-cy)/fy;
            double r2;
            //  2 由畸变参数计算每个点发生畸变后在归一化平面的对应坐标 (x_distorted,y_distorted)
            r2 = pow(x1,2)+pow(y1,2);
            x_distorted  = x1*(1+k1*r2+k2*pow(r2,2)+k3*pow(r2,3))+2*p1*x1*y1+p2*(r2+2*x1*x1);
            y_distorted = y1*(1+k1*r2+k2*pow(r2,2)+k3*pow(r2,3))+p1*(r2+2*y1*y1)+2*p2*x1*y1;
            //  3 将畸变后的点由内参矩阵投影到像素平面,得到该点在输入的带有畸变图像上的位置 
            u_distorted = fx*x_distorted+cx;
            v_distorted = fy*y_distorted+cy;
            distortion_uv_4 tmp;
            tmp.u_ = (int)(u_distorted);
            tmp.v_ = (int)(v_distorted);
            tmp.weight_11 = (int)((tmp.u_+1 - u_distorted)*2048)* (int)((tmp.v_ +1 - v_distorted)*2048);
            tmp.weight_21 = (int)((tmp.u_+1 - u_distorted)*2048)* (int)((v_distorted-tmp.v_)*2048);
            tmp.weight_12 = (int)((u_distorted-tmp.u_)*2048)* (int)((tmp.v_ +1 - v_distorted)*2048);
            tmp.weight_22 = (int)((u_distorted - tmp.u_)*2048)*(int)((v_distorted-tmp.v_)*2048);
            lookupTable[v][u] = tmp;
        }// for 
    }// for 
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int64_t getTime()
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec*1000.0+round(t.tv_nsec/1000000.0);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void YUV4202GRAY_CV_SAVE(std::string &inputPath ,cv::Mat&grayImg,int width,int height)
{
    FILE* pFileIn = fopen(inputPath.data(),"rb+");
    // FILE* pFileOut = fopen(outputPath.data(),"rb+");
    unsigned char* data = (unsigned char*) malloc(width*height*3/2);
    auto size_ = fread(data,height*width*sizeof(unsigned char),1,pFileIn);
	grayImg.create(height,width, CV_8UC1);
    memcpy(grayImg.data, data, height*width*sizeof(unsigned char));
    // fwrite(data.c_str(), 1, data.size(), pFileOut);
    free(data);
    fclose(pFileIn);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void myImageDistorted(cv::Mat &src , cv::Mat &image_undistort,const std::vector<std::vector<distortion_uv>> &lookupTable)
{
    double u_distorted,v_distorted;
    for (int v = 0; v < src.rows; v++)
    {
        for (int u = 0; u < src.cols; u++) 
        {
            u_distorted = lookupTable[v][u].u_;
            v_distorted = lookupTable[v][u].v_;
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < src.cols && v_distorted < src.rows) 
            {
                image_undistort.at<uint8_t>(v, u) = src.at<uint8_t>(int(std::round(v_distorted)), (int)std::round(u_distorted));
            }else 
            {
                image_undistort.at<uint8_t>(v, u)= 0;
            }
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void myImageDistorted(cv::Mat &src , cv::Mat &image_undistort,const std::vector<std::vector<distortion_uv_4>> &lookupTable)
{
    int u_1,v_1;
    uint32_t w_11,w_12,w_21,w_22;
    for (int v = 0; v < src.rows; v++)
    {
        for (int u = 0; u < src.cols; u++) 
        {
            u_1 = lookupTable[v][u].u_;
            v_1 = lookupTable[v][u].v_;
            w_11 = lookupTable[v][u].weight_11;
            w_12 = lookupTable[v][u].weight_12;
            w_21 = lookupTable[v][u].weight_21;
            w_22 = lookupTable[v][u].weight_22;
            if (u_1 >= 0 && v_1 >= 0 && u_1+1 < src.cols && v_1+1 < src.rows) 
            {
                auto value  =(w_11*(uint32_t)src.at<uint8_t>(v_1,u_1) + w_12*(uint32_t)src.at<uint8_t>(v_1,u_1+1) +  w_21*(uint32_t)src.at<uint8_t>(v_1+1,u_1) +  w_22*(uint32_t)src.at<uint8_t>(v_1 +1,u_1+1) );
                image_undistort.at<uint8_t>(v, u) = value/4194304;
            }else 
            {
                image_undistort.at<uint8_t>(v, u)= 0;
            }
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void myImageDistorted(cv::Mat &src , cv::Mat &image_undistort)
{
    cv::Mat image = src;
    int rows = src.rows, cols = src.cols;
    /*
     *  准备一张空白图像，由畸变模型和相机模型计算发生畸变后每个像素应在的位置(u,v)（也就是在原始输入图像上的位置）
     *  将该位置对应原始图像中的值，赋给空白图像就可以
    */

    for (int v = 0; v < src.rows; v++)
    {
        for (int u = 0; u < src.cols; u++) 
        {
            double u_distorted = 0, v_distorted = 0;
            double x1,y1,x_distorted,y_distorted;
            // 1 计算空白图像的每个点在归一化平面上的坐标（x1,y1）
            x1 = (u-cx)/fx;
            y1 = (v-cy)/fy;
            double r2;
            //  2 由畸变参数计算每个点发生畸变后在归一化平面的对应坐标 (x_distorted,y_distorted)
            r2 = pow(x1,2)+pow(y1,2);
            x_distorted  = x1*(1+k1*r2+k2*pow(r2,2)+k3*pow(r2,3))+2*p1*x1*y1+p2*(r2+2*x1*x1);
            y_distorted = y1*(1+k1*r2+k2*pow(r2,2)+k3*pow(r2,3))+p1*(r2+2*y1*y1)+2*p2*x1*y1;
            //  3 将畸变后的点由内参矩阵投影到像素平面,得到该点在输入的带有畸变图像上的位置 
            u_distorted = fx*x_distorted+cx;
            v_distorted = fy*y_distorted+cy;
            // 4 对空白图像的每个像素
#if 0
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                image_undistort.at<cv::Vec3b>(v, u)[0] = image.at<cv::Vec3b>(int(std::round(v_distorted)), (int)std::round(u_distorted))[0];
                image_undistort.at<cv::Vec3b>(v, u)[1] = image.at<cv::Vec3b>(int(std::round(v_distorted)), (int)std::round(u_distorted))[1];
                image_undistort.at<cv::Vec3b>(v, u)[2] = image.at<cv::Vec3b>(int(std::round(v_distorted)), (int)std::round(u_distorted))[2];
            } else {
                image_undistort.at<cv::Vec3b>(v, u)[0] = 0;
                image_undistort.at<cv::Vec3b>(v, u)[1] = 0;
                image_undistort.at<cv::Vec3b>(v, u)[2] = 0;
            }
#endif
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                image_undistort.at<uint8_t>(v, u) = image.at<uint8_t>(int(std::round(v_distorted)), (int)std::round(u_distorted));
            } else {
                image_undistort.at<uint8_t>(v, u)= 0;
            }
        } // for 
    }// for
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool SaveImage(const cv::Mat& img, std::string& absolutePath)
{
    std::vector<int> compression_params;
    // compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);//选择PNG格式
    compression_params.push_back(16);//选择PNG格式
    compression_params.push_back(0); // 无压缩png（从0-9.较高的值意味着更小的尺寸和更长的压缩时间而默认值是3.本人选择0表示不压缩）
    cv::imwrite(absolutePath, img, compression_params);
    return true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void YU12toRGB(std::string &yuv_file_path,cv::Mat &rgb_Img,const int w , const int h ,bool blSave)
{
	printf("YUV file w: %d, h: %d \n", w, h);
	FILE* pFileIn = fopen((yuv_file_path.data()), "rb+");
	int bufLen = w*h*3/2;
	unsigned char* pYuvBuf = new unsigned char[bufLen];
	auto size_ = fread(pYuvBuf, bufLen*sizeof(unsigned char), 1, pFileIn);
	cv::Mat yuvImg;
	yuvImg.create(h*3/2, w, CV_8UC1); 
	memcpy(yuvImg.data, pYuvBuf, bufLen*sizeof(unsigned char));
	// cv::cvtColor(yuvImg, rgbImg,  CV_YUV2BGR_I420);
    cv::cvtColor(yuvImg, rgb_Img,  101);
    // cv::namedWindow("new_img", CV_WINDOW_NORMAL); //图像自适应大小，否者会因为图像太大，看不全
    // cv::namedWindow("new_img", 0x00000000); //图像自适应大小，否者会因为图像太大，看不全
    if (0)
    {
        std::string path = "/tmp/222.jpg";
        SaveImage(rgb_Img,path);
    }
	delete[] pYuvBuf;
	fclose(pFileIn);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void poseOptimization(const std::vector<Eigen::Vector3d>& tag1_points, 
                                             const std::vector<Eigen::Vector3d>& tag2_points,
                                             const Eigen::Matrix3d &K,
                                             Eigen::Matrix3d & R1, Eigen::Vector3d & t1,
                                             Eigen::Matrix3d & R2, Eigen::Vector3d & t2 )
{
    const std::vector<Eigen::Vector3d> real_points{ Eigen::Vector3d (-0.03,0.03,0.0), Eigen::Vector3d (0.03,0.03,0.0), Eigen::Vector3d (0.03,-0.03,0.0),
                                                                                        Eigen::Vector3d (-0.03,-0.03,0.0), Eigen::Vector3d (0.0,0.0,0.0)};

    // std::cout << "[Before] t1  = \n" << t1[0] << "," <<t1[1] << "," << t1[2]<< std::endl;
    // std::cout << "[Before] t2  = \n" << t2[0] << "," <<t2[1] << "," << t2[2]<< std::endl;                                                                              

    // 旋转矩阵转成四元数
    Eigen::Quaterniond quaternion1(R1);
    Eigen::Quaterniond quaternion2(R2);

    // Build the problem.
    ceres::Problem problem;
    // ceres::LossFunction* loss_function = new ceres::CauchyLoss(1);
    ceres::LossFunction* loss_function = NULL;
    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;

    for(int i = 0 ; i < 5; i++)
    {
        CostFunctor *Cost_functor1 = new CostFunctor (tag1_points[i],tag1_points[4],tag2_points[4],real_points[i],K,1);
        problem.AddResidualBlock( new AutoDiffCostFunction<CostFunctor,7,4,3,4,3> (Cost_functor1), loss_function,
                                                         quaternion1.coeffs().data(),t1.data(),quaternion2.coeffs().data(),t2.data());
        problem.SetParameterization(quaternion1.coeffs().data(), quaternion_local_parameterization);
        problem.SetParameterization(quaternion2.coeffs().data(), quaternion_local_parameterization);
    }
    for(int i = 0 ; i < 5; i++)
    {
        CostFunctor *Cost_functor2 = new CostFunctor (tag2_points[i],tag1_points[4],tag2_points[4],real_points[i],K,2);
        problem.AddResidualBlock( new AutoDiffCostFunction<CostFunctor,7,4,3,4,3> (Cost_functor2), loss_function,
                                                         quaternion1.coeffs().data(),t1.data(),quaternion2.coeffs().data(),t2.data());
        problem.SetParameterization(quaternion1.coeffs().data(), quaternion_local_parameterization);
        problem.SetParameterization(quaternion2.coeffs().data(), quaternion_local_parameterization);
    }
    // Solve
    ceres::Solver::Options solver_options;//实例化求解器对象    
    solver_options.linear_solver_type=ceres::DENSE_QR;
    solver_options.minimizer_progress_to_stdout= false;
    //实例化求解对象
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options,&problem,&summary);

    // print result
    R1 = quaternion1.toRotationMatrix();
    R2 = quaternion2.toRotationMatrix();
    Eigen::Vector3d m = R1.col(2);
    Eigen::Vector3d n = R2.col(2);
    auto theta = std::acos((double)(m.transpose()*n) / (m.norm()*n.norm()));
    printf ("[AFTER]-----------------------------------The angle diff between id-3 and id-6 = %f \n",theta * 180/3.14159);
    // std::cout << "R1.transpose()*R1 = \n" << (R1.transpose()*R1) << std::endl;
    // std::cout << "R2.transpose()*R2= \n" << (R2.transpose()*R2)<< std::endl;
    // std::cout << "[AFTER] t1  = \n" << t1[0] << "," <<t1[1] << "," << t1[2]<< std::endl;
    // std::cout << "[AFTER] t2  = \n" << t2[0] << "," <<t2[1] << "," << t2[2]<< std::endl;
    // std::cout << summary.FullReport() << '\n';
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void singleTagPoseOptimization(const std::vector<Eigen::Vector3d>& tag1_points,  const Eigen::Matrix3d &K, Eigen::Matrix3d & R1, Eigen::Vector3d & t1)
{
    const std::vector<Eigen::Vector3d> real_points{ Eigen::Vector3d (-0.03,0.03,0.0), Eigen::Vector3d (0.03,0.03,0.0), Eigen::Vector3d (0.03,-0.03,0.0),
                                                                                        Eigen::Vector3d (-0.03,-0.03,0.0), Eigen::Vector3d (0.0,0.0,0.0)};
    // std::cout << "[Before] t1  = \n" << t1[0] << "," <<t1[1] << "," << t1[2]<< std::endl;                                                                          

    // 旋转矩阵转成四元数
    Eigen::Quaterniond quaternion1(R1);
    // Build the problem.
    ceres::Problem problem;
    // ceres::LossFunction* loss_function = new ceres::CauchyLoss(1);
    ceres::LossFunction* loss_function = NULL;
    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;

    for(int i = 0 ; i < 5; i++)
    {
        ReprojectCostFunctor *Cost_functor = new ReprojectCostFunctor (tag1_points[i],real_points[i],K);
        problem.AddResidualBlock( new AutoDiffCostFunction<ReprojectCostFunctor,2,4,3> (Cost_functor), loss_function,
                                                         quaternion1.coeffs().data(),t1.data());
        problem.SetParameterization(quaternion1.coeffs().data(), quaternion_local_parameterization);
    }

    // Solve
    ceres::Solver::Options solver_options;//实例化求解器对象    
    solver_options.linear_solver_type=ceres::DENSE_QR;
    solver_options.minimizer_progress_to_stdout= false;
    //实例化求解对象
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options,&problem,&summary);

    // print result
    R1 = quaternion1.toRotationMatrix();
    // std::cout << "R1.transpose()*R1 = \n" << (R1.transpose()*R1) << std::endl;
    // std::cout << "[AFTER] t1  = \n" << t1[0] << "," <<t1[1] << "," << t1[2]<< std::endl;
    // std::cout << summary.FullReport() << '\n';
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief 
 * 
 * @param rgbImageRaw 
 * @param frame 
 * @param distortLookupTable 
 */
void  imagePreprocess( cv::Mat &rgbImageRaw , cv::Mat & frame, std::vector<std::vector<distortion_uv_4>> &distortLookupTable)
{
        //  2 裁剪
        auto my_select = cv::Rect(300,200,1320,680);
        cv::Mat rgbImage = rgbImageRaw(my_select);
        // 3 图像降采样
        int rows = rgbImage.rows, cols = rgbImage.cols;
        cv::Mat newFrame = rgbImage;
        // cv::pyrDown(rgbImage,newFrame,cv::Size(cols/2,rows/2));
        // 4 对RGB图像去畸变
        cv::Mat frame1 = cv::Mat(newFrame.rows, newFrame.cols, CV_8UC1);   // 去畸变以后的图
        myImageDistorted(newFrame,frame1,distortLookupTable);
        //  5 直方图均衡化
        // cv::equalizeHist(frame1,frame);
        // 6   RGB转成灰度图  
        // cvtColor(frame1, frame, COLOR_BGR2GRAY);
        frame = frame1.clone();
        // frame = newFrame.clone();
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief 
 * 
 * @param frame 
 * @param corners 
 * @param corners_after_gftt 
 */
void refinementCornerWithGFTT(const cv::Mat &frame, const std::vector<cv::Point2f> &corners ,std::vector<cv::Point2f> &corners_after_gftt)
{
     //  计算corners的周围7*7的点的最小特征值
    const int halfWindowSize = 7;
    const int halfSmallWindowSize = 2;
    std::vector<cv::Mat> cornersMinValuesMatVec;
    // 计算每个角点周围11*11区域的响应（最小特征值）
    for ( auto  i = 0 ;  i < corners.size(); i ++)
    {
        auto currentSelect = cv::Rect( std::round(corners[i].x-halfWindowSize),std::round(corners[i].y-halfWindowSize),2*halfWindowSize+1,2*halfWindowSize+1);
        cv::Mat  tmpMat = frame(currentSelect);
        cv::Mat cornerMinValueMat;
        // cv::GaussianBlur(tmpMat,tmpMat,cv::Size(3,3),0);
        cv::cornerMinEigenVal(tmpMat,cornerMinValueMat,3,3,4); // sobel 算子的size 3*3
        cornersMinValuesMatVec.push_back(cornerMinValueMat);
    }
    // 遍历每个角点周围的11*11响应矩阵，            corners_after_refinementOnRawImage = corners;
    corners_after_gftt.resize(4);
    corners_after_gftt = corners;
    int count =0;
    for ( auto mat : cornersMinValuesMatVec)
    {
        auto currentSelect = cv::Rect(halfWindowSize-halfSmallWindowSize,halfWindowSize-halfSmallWindowSize,2*halfSmallWindowSize+1,2*halfSmallWindowSize+1);
        cv::Mat  tmpMat = mat(currentSelect);
        double maxValue;    // 最大值，最小值
        cv::Point  maxIdx;    // 最小值坐标，最大值坐标
        cv::minMaxLoc(tmpMat, nullptr, &maxValue, nullptr, &maxIdx);
        // std::cout << "corner_" << count << "\n"<< tmpMat << "\n"; 
        // std::cout << "最大值：" << maxValue  << ", 最大值位置：" << maxIdx << std::endl;
        // 计算
        corners_after_gftt[count].y = corners[count].y + (maxIdx.y - halfSmallWindowSize);
        corners_after_gftt[count].x = corners[count].x + (maxIdx.x - halfSmallWindowSize);
        count++;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief 
 * 
 * @param frame 
 * @param corners_ 
 * @param corners_after_subpixel_ 
 */
void refinementCornerWithSubPixel(const cv::Mat &frame, const std::vector<cv::Point2f> &corners_ ,std::vector<cv::Point2f> &corners_after_subpixel_)
{
    corners_after_subpixel_ = corners_;
    cv::cornerSubPix(frame, corners_after_subpixel_, cv::Size(3, 3), cv::Size(-1, -1), cv::TermCriteria(2+1, 1000000, 0.001));
    // cv::cornerSubPix(frame, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief 
 * 
 * @param corners_final 
 * @param newH 
 */
void computeHomographyWithCorners( const std::vector<cv::Point2f> &corners_final , Eigen::Matrix3d &newH)
{
    //  计算新的单应矩阵
    double corr_arr[4][4];
    for (int i = 0; i < 4; i++) 
    {
        corr_arr[i][0] = (i==0 || i==3) ? -1 : 1;
        corr_arr[i][1] = (i==0 || i==1) ? -1 : 1;
    }
    corr_arr[0][2] = corners_final[3].x;
    corr_arr[0][3] = corners_final[3].y;
    corr_arr[1][2] = corners_final[2].x;
    corr_arr[1][3] = corners_final[2].y;
    corr_arr[2][2] = corners_final[1].x;
    corr_arr[2][3] = corners_final[1].y;
    corr_arr[3][2] = corners_final[0].x;
    corr_arr[3][3] = corners_final[0].y;
    homography_compute3(corr_arr,newH);
    // std::cout << "New Homography Matrix: \n"<< newH << "\n";
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief 
 * 
 * @param corners 
 * @param rawImage 
 * @param frame 
 * @param new_corners 
 */
void refinementCornersOnRawImage( const std::vector<cv::Point2f> &corners, const cv::Mat &rawImage, const cv::Mat &frame ,std::vector<cv::Point2f> &new_corners)
{
    Eigen::Matrix3d cameraK;
    cameraK << fx,0,cx,0,fy,cy,0,0,1;
    cv::Mat raw_im = rawImage.clone();
    cv::Mat im = frame.clone();
    // 1 转回到原始图像上
    std::vector<cv::Point2f> corners_on_raw;
    for ( int i = 0 ; i < 5; i ++ )
    {
        auto x1 = (corners[i].x - cx)/fx;
        auto y1 = (corners[i].y - cy)/fy;
        double r2;
        //  由畸变参数计算每个点发生畸变后在归一化平面的对应坐标 (x_distorted,y_distorted)
        r2 = pow(x1,2)+pow(y1,2);
        auto x_distorted  = x1*(1+k1*r2+k2*pow(r2,2)+k3*pow(r2,3))+2*p1*x1*y1+p2*(r2+2*x1*x1);
        auto y_distorted = y1*(1+k1*r2+k2*pow(r2,2)+k3*pow(r2,3))+p1*(r2+2*y1*y1)+2*p2*x1*y1;
        //  将畸变后的点由内参矩阵投影到像素平面,得到该点在输入的带有畸变图像上的位置
        corners_on_raw.emplace_back(cv::Point2f(fx*x_distorted+cx,fy*y_distorted+cy));
    }
    // corners_on_raw = corners;
    // 2 在原始图像上计算亚像素的角点
    std::vector<cv::Point2f> corners_on_raw_after_gftt;
    refinementCornerWithGFTT(raw_im,corners_on_raw,corners_on_raw_after_gftt);
    std::vector<cv::Point2f> corners_on_raw_after_subpixel;
    refinementCornerWithSubPixel(raw_im,corners_on_raw_after_gftt,corners_on_raw_after_subpixel);
    // 3 反投回
    for (int j = 0 ; j < 5; j++)
    {
        double x_distortion = (corners_on_raw_after_subpixel[j].x - cx)/fx;
        double y_distortion = (corners_on_raw_after_subpixel[j].y - cy)/fy;
        double point[2];
        point[0] = x_distortion;
        point[1] = y_distortion;
        cv::Point2d point_distortion(x_distortion,y_distortion);
        ceres::Problem problem;
        ceres::LossFunction* loss_function = NULL;
        distortedCostFunctor *Cost_functor = new distortedCostFunctor (point_distortion,cameraK);
        problem.AddResidualBlock(new AutoDiffCostFunction<distortedCostFunctor,2,2> (Cost_functor), loss_function,point);
        ceres::Solver::Options solver_options;//实例化求解器对象    
        solver_options.linear_solver_type=ceres::DENSE_QR;
        solver_options.minimizer_progress_to_stdout= false;
        //实例化求解对象
        ceres::Solver::Summary summary;
        ceres::Solve(solver_options,&problem,&summary);
        float x_undistortion = fx*point[0]+cx;
        float y_undistortion = fy*point[1]+cy;
        new_corners.emplace_back(cv::Point2f(x_undistortion,y_undistortion));
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void writeOnImage(cv::Mat &frame , const cv::Point2i &position,std::string &text_str)
{
        cv::String text = text_str;
        int fontface = FONT_HERSHEY_SIMPLEX;
        double fontscale = 1.0;
        int baseline;
        Size textsize = getTextSize(text, fontface, fontscale, 2,&baseline);
        putText(frame, text, Point(position.x-textsize.width/2,position.y+textsize.height/2),fontface, fontscale, Scalar(0xff, 0x99, 0), 2);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    getopt_t *getopt = getopt_create();
    Eigen::Matrix3d K ;
    getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
    getopt_add_bool(getopt, 'd', "debug", 0, "Enable debugging output (slow)");
    getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
    getopt_add_string(getopt, 'f', "family", "tagCustom16h5", "Tag family to use");
    getopt_add_int(getopt, 'i', "iters", "1", "Repeat processing on input set this many times");
    getopt_add_int(getopt, 't', "threads", "2", "Use this many CPU threads");
    getopt_add_int(getopt, 'a', "hamming", "1", "Detect tags with up to this many bit errors.");
    getopt_add_double(getopt, 'x', "decimate", "4.0", "Decimate input image by this factor");
    getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input; negative sharpens");
    getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");

    // Initialize tag detector with options
    apriltag_family_t *tf = NULL;
    const char *famname = getopt_get_string(getopt, "family");

    if (!strcmp(famname, "tag36h11")) {
        tf = tag36h11_create();
    } else if (!strcmp(famname, "tag25h9")) {
        tf = tag25h9_create();
    } else if (!strcmp(famname, "tag16h5")) {
        tf = tag16h5_create();
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tf = tagCircle21h7_create();
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tf = tagCircle49h12_create();
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tf = tagStandard41h12_create();
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tf = tagStandard52h13_create();
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tf = tagCustom48h12_create();
    } else if (!strcmp(famname, "tagCustom16h5")) {
        tf = tagCustom16h5_create();
    } else {
        printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
        exit(-1);
    }

    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family_bits(td, tf, getopt_get_int(getopt, "hamming"));
    td->quad_decimate = getopt_get_double(getopt, "decimate");
    td->quad_sigma = getopt_get_double(getopt, "blur");
    td->nthreads = getopt_get_int(getopt, "threads");
    td->debug = getopt_get_bool(getopt, "debug");
    td->refine_edges = getopt_get_bool(getopt, "refine-edges");

    // 
    std::vector<std::vector<distortion_uv_4>> distortLookupTable;
    preBuildDistortedLookupTable(distortLookupTable,(1920-600),(1080-400));
    // 
    Mat frame,rgbImageRaw;
    cv::Mat backup_image; // 用于保存预处理后的图像
    const int testNumber = 20;

    for ( int imageIndex = 1 ; imageIndex < testNumber; imageIndex++)
    {
        std::cout << "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NEW IMAGE <<<<<<<<<<<<<<<<<<< diatance = "<< imageIndex*10 <<"cm \n";
        //  1 循环读取YUV将其转成GRAY
        std::string new_path = path_prefix +std::to_string(imageIndex)+".yuv";

#ifdef TIME_STATISTICS
    auto  imagePreprocess_startTime = getTime();
#endif 

        YUV4202GRAY_CV_SAVE(new_path,rgbImageRaw,1920,1080);
        
        auto my_select = cv::Rect(300,200,1320,680);
        cv::Mat rawImageFor = rgbImageRaw(my_select);

        imagePreprocess(rgbImageRaw,frame,distortLookupTable);

#ifdef TIME_STATISTICS
    auto  imagePreprocess_endTime = getTime();
    std::cout << "[Time] Step 1 :  imagePreprocess_Time =  " << imagePreprocess_endTime - imagePreprocess_startTime << "\n";
#endif 

        backup_image = frame.clone();

        // Make an image_u8_t header for the Mat data
        image_u8_t im = 
        { 
            .width = frame.cols,
            .height = frame.rows,
            .stride = frame.cols,
            .buf = frame.data
        };

#ifdef TIME_STATISTICS
    auto  apriltagDetect_startTime = getTime();
#endif

        // 检测tag，并计算H  
        zarray_t *detections = apriltag_detector_detect(td, &im);
        
#ifdef TIME_STATISTICS
    auto  apriltagDetect_endTime = getTime();
    std::cout << "[Time] Step 2:  apriltagDetect_Time =  " << apriltagDetect_endTime - apriltagDetect_startTime << "\n";
#endif

        Eigen::Vector3d rotation_z_3;
        Eigen::Vector3d rotation_z_6;
        bool id3ready = false , id6ready =false;
        Eigen::Matrix3d rotation_matrix;

        // cv::Mat init_frame;
        cv::Mat image_with_init_corners = frame.clone();
        cv::Mat image_with_gftt_corners = frame.clone();
        cv::Mat image_with_subpixel_corners = frame.clone();
        cv::Mat image_check = frame.clone();
        std::vector<Eigen::Vector3d> tag1_points; // 用于存储Tag1的图像上角点及中心点
        std::vector<Eigen::Vector3d> tag2_points; // 用于存储Tag2的图像上角点及中心点
        Eigen::Matrix3d rotationMatrixTag1; 
        Eigen::Matrix3d rotationMatrixTag2;
        Eigen::Vector3d tranVecTag1;
        Eigen::Vector3d tranVecTag2;
        
        //  遍历每个检测结果
        for (int i = 0; i < zarray_size(detections); i++) 
        {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);


            if ((det->id != rightTagId && det->id != leftTagId))
            {
                continue;
            }
            // 
            std::vector<cv::Point2f> corners;
            corners.resize(5);
            printf("\n<<<<<<<<<<<< TAG ID = %i, decision_margin = %f\n", det->id,det->decision_margin);

            auto updateCorners = [&]()
            {
                for (int i = 0 ; i < 4; i++)
                {
                    cv::Point2f tmp(det->p[i][0],det->p[i][1]);
                    corners[i] = tmp;
                }
                corners[4] = cv::Point2f(det->c[0],det->c[1]);

                if (0)
                {
                    for ( int i = 0 ; i < 4 ; i ++)
                    {
                        printf("Apriltag detect conner %i =(%f , %f) \n",i,det->p[i][0],det->p[i][1]);
                    }
                    printf("center = %f ,%f \n",det->c[0],det->c[1]);
                    printf("Homography Matrix :  \n %f,%f,%f \n %f,%f,%f\n  %f,%f,%f\n", det->H->data[0],det->H->data[1],det->H->data[2],det->H->data[3],
                                det->H->data[4],det->H->data[5],det->H->data[6],det->H->data[7],det->H->data[8]);
                }
            };
            updateCorners();

#if 0
            //  cal GFTT
            std::vector<cv::Point2f> corners_after_gftt;
            refinementCornerWithGFTT(frame,corners,corners_after_gftt);

            //  do cornerSubPix
            std::vector<cv::Point2f> corners_after_subpixel;
            refinementCornerWithSubPixel(frame,corners_after_gftt,corners_after_subpixel);
#endif

#ifdef TIME_STATISTICS
    auto  refinementCornersOnRawImage_startTime = getTime();
#endif

            std::vector<cv::Point2f> corners_after_refinementOnRawImage;
            refinementCornersOnRawImage(corners,rawImageFor,frame,corners_after_refinementOnRawImage);

#ifdef TIME_STATISTICS
    auto  refinementCornersOnRawImage_endTime = getTime();
    std::cout << "[Time] Step 3-1:  refinementCornersOnRawImage_Time =  " << refinementCornersOnRawImage_endTime - refinementCornersOnRawImage_startTime << "\n";
#endif

            //  确定最终的角点
            std::vector<cv::Point2f> corners_final;
            corners_final = corners_after_refinementOnRawImage;

            // 计算新的单应 
            Eigen::Matrix3d newH;
            computeHomographyWithCorners(corners_final,newH);
            
            // 使用新的角点更新det的 H 及 corners
            auto updateDetwithNewCorners = [&] ()
            {
                det->H->data[0] = newH(0,0);
                det->H->data[1] = newH(0,1);
                det->H->data[2] = newH(0,2);
                det->H->data[3] = newH(1,0);
                det->H->data[4] = newH(1,1);
                det->H->data[5] = newH(1,2);
                det->H->data[6] = newH(2,0);
                det->H->data[7] = newH(2,1);
                det->H->data[8] = newH(2,2);
                for (int i = 0; i < 4; i++) 
                {
                    det->p[i][0] =  corners_final[i].x;
                    det->p[i][1]  = corners_final[i].y;
                }
            };
            updateDetwithNewCorners();

            line(frame, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[1][0], det->p[1][1]),
                     Scalar(0, 0xff, 0), 2);
            line(frame, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0, 0, 0xff), 2);
            line(frame, Point(det->p[1][0], det->p[1][1]),
                     Point(det->p[2][0], det->p[2][1]),
                     Scalar(0xff, 0, 0), 2);
            line(frame, Point(det->p[2][0], det->p[2][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0xff, 0, 0), 2);

            stringstream ss;
            ss << det->id;
            String text = ss.str();
            int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
            double fontscale = 1.0;
            int baseline;
            Size textsize = getTextSize(text, fontface, fontscale, 2,&baseline);
            putText(frame, text, Point(det->c[0]-textsize.width/2,det->c[1]+textsize.height/2),fontface, fontscale, Scalar(0xff, 0x99, 0), 2);


#ifdef TIME_STATISTICS
    auto  estimate_tag_pose_startTime = getTime();
#endif
            //  3 estimate_tag_pose
            // First create an apriltag_detection_info_t struct using your known parameters.
            apriltag_detection_info_t info;
            info.det = det;
            info.tagsize = 0.06;
            info.fx =  fx;
            info.fy =  fy;
            info.cx = cx;
            info.cy = cy;
            K << info.fx,0,info.cx,0,info.fy,info.cy,0,0,1;
            // Then call estimate_tag_pose.
            apriltag_pose_t pose;
            double err = estimate_tag_pose(&info, &pose);

            // 将位姿估计的结果整理成旋转矩阵
            rotation_matrix << pose.R->data[0],pose.R->data[1],pose.R->data[2],pose.R->data[3],pose.R->data[4],pose.R->data[5],pose.R->data[6],pose.R->data[7],pose.R->data[8];

#ifdef TIME_STATISTICS
    auto  estimate_tag_pose_endTime = getTime();
    std::cout << "[Time] Step 3-2:  estimate_tag_pose_endTime =  " << estimate_tag_pose_endTime - estimate_tag_pose_startTime << "\n";
#endif


            if ( det->id == rightTagId )
            {
                rotation_z_3 << rotation_matrix(0,2) , rotation_matrix(1,2), rotation_matrix(2,2);
                tag1_points.resize(5);
                for(int index = 0 ; index < 5; index++ )
                {
                    tag1_points[index] = Eigen::Vector3d(double(corners_final[index].x) , double(corners_final[index].y) ,1.0);
                }
                rotationMatrixTag1 = rotation_matrix;
                tranVecTag1 = Eigen::Vector3d (double(pose.t->data[0]),double(pose.t->data[1]),double(pose.t->data[2]));
                id3ready = true;
            }
            if ( det->id == leftTagId )
            {
                rotation_z_6 << rotation_matrix(0,2), rotation_matrix(1,2), rotation_matrix(2,2);
                tag2_points.resize(5);
                for(int index = 0 ; index < 5; index++ )
                {
                    tag2_points[index] = Eigen::Vector3d(double(corners_final[index].x) , double(corners_final[index].y) ,1.0);
                }
                rotationMatrixTag2 = rotation_matrix;
                tranVecTag2=  Eigen::Vector3d (double(pose.t->data[0]),double(pose.t->data[1]),double(pose.t->data[2]));
                id6ready = true;
            }

            for(int corner_id = 0; corner_id < 5; corner_id++)
            {
                image_with_init_corners.at<uint8_t>(std::round(corners[corner_id].y),std::round(corners[corner_id].x)) = 255;
                cv::circle(image_with_init_corners, corners[corner_id], 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                // image_with_gftt_corners.at<uint8_t>(std::round(corners_after_gftt[corner_id].y),std::round(corners_after_gftt[corner_id].x)) = 255;
                // cv::circle(image_with_gftt_corners, corners_after_gftt[corner_id], 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                image_with_subpixel_corners.at<uint8_t>(std::round(corners_final[corner_id].y),std::round(corners_final[corner_id].x)) = 255;
                cv::circle(image_with_subpixel_corners, corners_final[corner_id], 5, cv::Scalar(0, 255, 0), 2, 8, 0);
            }
        } // FOR det
        
        auto printEstimateTagPose = [&]()
        {
            if ( id3ready && id6ready )
            {
                double tmp = rotation_z_3.transpose()*rotation_z_6;
                auto theta = std::acos(tmp/(rotation_z_6.norm()*rotation_z_3.norm()));
                printf ("[Before]>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>The angle diff between id-3 and id-6 = %f \n",theta * 180/3.14159);
            }
        };
        printEstimateTagPose();

        //   对pose进行的检验
        auto checkTagPose = [&]( bool optimized )
        {
            if ( id3ready && id6ready )
            {
                Eigen::Vector3d pointTagTest(tagCenterDistance/2,tagCenterHeight,-pointZ);
                Eigen::Vector3d pointTagTest_(-tagCenterDistance/2,tagCenterHeight,-pointZ);
                Eigen::Vector3d pointTagTest1 = (rotationMatrixTag2*pointTagTest+tranVecTag2);
                Eigen::Vector3d pointTagTest2 = (rotationMatrixTag1*pointTagTest_+tranVecTag1);
                Eigen::Vector3d point_uv_1 = K*pointTagTest1;
                Eigen::Vector3d point_uv_2 = K*pointTagTest2;
                
                if ( optimized )
                {
                    image_check.at<uint8_t>(std::round(point_uv_1[1]/point_uv_1[2]),std::round(point_uv_1[0]/point_uv_1[2])) = 255;
                    Rect rect1(int(point_uv_1[0]/point_uv_1[2]-5), int(point_uv_1[1]/point_uv_1[2]-5), 10, 10);//左上坐标（x,y）和矩形的长(x)宽(y)
                    cv::rectangle(image_check, rect1, Scalar(0,255, 0),1, LINE_8,0);
                }else{
                    image_check.at<uint8_t>(std::round(point_uv_1[1]/point_uv_1[2]),std::round(point_uv_1[0]/point_uv_1[2])) = 255;
                    cv::circle(image_check, cv::Point(point_uv_1[0]/point_uv_1[2],point_uv_1[1]/point_uv_1[2]), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                }
                
                if ( optimized )
                {
                    image_check.at<uint8_t>(std::round(point_uv_2[1]/point_uv_2[2]),std::round(point_uv_2[0]/point_uv_2[2])) = 255;
                     Rect rect(int(point_uv_2[0]/point_uv_2[2]-5), int(point_uv_2[1]/point_uv_2[2]-5), 10, 10);//左上坐标（x,y）和矩形的长(x)宽(y)
                    cv::rectangle(image_check, rect, Scalar(0, 255, 0),1, LINE_8,0);
                }else{
                    image_check.at<uint8_t>(std::round(point_uv_2[1]/point_uv_2[2]),std::round(point_uv_2[0]/point_uv_2[2])) = 255;
                    cv::circle(image_check, cv::Point(point_uv_2[0]/point_uv_2[2],point_uv_2[1]/point_uv_2[2]), 8, cv::Scalar(0, 255, 0), 2, 8, 0);
                }

                const std::vector<Eigen::Vector3d> real_points_{Eigen::Vector3d (-0.03,0.03,0.0), Eigen::Vector3d (0.03,0.03,0.0), Eigen::Vector3d (0.03,-0.03,0.0),
                                                                                                     Eigen::Vector3d (-0.03,-0.03,0.0), Eigen::Vector3d (0.0,0.0,0.0),
                                                                                                     Eigen::Vector3d (-0.02,0.02,0.0), Eigen::Vector3d (0.02,0.02,0.0), Eigen::Vector3d (0.02,-0.02,0.0),
                                                                                                     Eigen::Vector3d (-0.02,-0.02,0.0),Eigen::Vector3d (-0.01,0.01,0.0), Eigen::Vector3d (0.01,0.01,0.0), 
                                                                                                     Eigen::Vector3d (0.01,-0.01,0.0), Eigen::Vector3d (-0.01,-0.01,0.0)};
                for (auto p : real_points_)
                {
                    Eigen::Vector3d point_uv_r = K*(rotationMatrixTag1*p+tranVecTag1);
                    image_check.at<uint8_t>(std::round(point_uv_r[1]/point_uv_r[2]),std::round(point_uv_r[0]/point_uv_r[2])) = 255;
                    cv::circle(image_check, cv::Point(point_uv_r[0]/point_uv_r[2],point_uv_r[1]/point_uv_r[2]), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                    Eigen::Vector3d point_uv_r2 = K*(rotationMatrixTag2*p+tranVecTag2);
                    image_check.at<uint8_t>(std::round(point_uv_r2[1]/point_uv_r2[2]),std::round(point_uv_r2[0]/point_uv_r2[2])) = 255;
                    cv::circle(image_check, cv::Point(point_uv_r2[0]/point_uv_r2[2],point_uv_r2[1]/point_uv_r2[2]), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                }
            }
        };
        // 
        

#ifdef TIME_STATISTICS
    auto  poseOptimization_startTime = getTime();
#endif
        

        // R1 t1 R2 t2 refienment
        if ( id3ready && id6ready )
        {       
            // 单独对R1和t1做优化
            singleTagPoseOptimization(tag1_points,K,rotationMatrixTag1,tranVecTag1);
            
            // 单独对R2和t2做优化
            singleTagPoseOptimization(tag2_points,K,rotationMatrixTag2,tranVecTag2);

            // T1 T2 联合优化
            poseOptimization(tag1_points,tag2_points,K,rotationMatrixTag1,tranVecTag1,rotationMatrixTag2,tranVecTag2);
        }

#ifdef TIME_STATISTICS
    auto  poseOptimization_endTime = getTime();
    std::cout << "[Time] Step 4:  poseOptimization_Time =  " << poseOptimization_endTime - poseOptimization_startTime << "\n";
#endif


#ifdef TIME_STATISTICS
    auto  image_endTime = getTime();
    std::cout << "[Time]  ALL  Time  :  =  " << image_endTime - imagePreprocess_startTime << "\n";
#endif 

        checkTagPose(true);

        apriltag_detections_destroy(detections);
        // 
        auto saveImageResult = [&]()
        {
            std::string out_path0 = "/data/rgb/res_"+std::to_string(imageIndex)+".jpg";
            SaveImage(frame,out_path0);
            //  
            std::string out_path1 = "/data/rgb/image_with_init_corners_"+std::to_string(imageIndex)+".jpg";
            std::string image_with_init_corners_label = "AprilTag corner detection results";
            writeOnImage(image_with_init_corners,cv::Point2i(660,200),image_with_init_corners_label);
            SaveImage(image_with_init_corners,out_path1);
            //  
            std::string out_path2 = "/data/rgb/image_with_gftt_corners_"+std::to_string(imageIndex)+".jpg";
            SaveImage(image_with_gftt_corners,out_path2);
            //  保存最后优化完的角点
            std::string out_path3 = "/data/rgb/image_with_subpixel_corners_"+std::to_string(imageIndex)+".jpg";
            std::string image_with_subpixel_corners_label = "The result of corner optimization on the original image";
            writeOnImage(image_with_subpixel_corners,cv::Point2i(660,200),image_with_subpixel_corners_label);
            SaveImage(image_with_subpixel_corners,out_path3);
            // 在检测图上打印文本
            std::string image_check_label = "pose check";
            writeOnImage(image_check,cv::Point2i(660,200),image_check_label);
            // 保存去除畸变的灰度图
            // std::string out_path_backup = "/home/xinyu/workspace/360/apriltags_tas/catkin_ws/src/apriltags_tas/apriltags_tas/example_images/"+std::to_string(imageIndex)+".jpg";
            // SaveImage(backup_image,out_path_backup);

            // combine images
            std::vector<cv::Mat> imageVec;
            cv::Mat combineImage;
            imageVec.push_back(image_with_init_corners);
            // imageVec.push_back(image_with_gftt_corners);
            imageVec.push_back(image_with_subpixel_corners);
            imageVec.push_back(image_check);
            // imageVec.push_back(image_raw);
            cv::vconcat(imageVec,combineImage);
            std::string out_path4 = "/data/rgb/combineImage_"+std::to_string(imageIndex)+".jpg";
            SaveImage(combineImage,out_path4);
        };
        saveImageResult();


    } // for image

    apriltag_detector_destroy(td);

    if (!strcmp(famname, "tag36h11")) {
        tag36h11_destroy(tf);
    } else if (!strcmp(famname, "tag25h9")) {
        tag25h9_destroy(tf);
    } else if (!strcmp(famname, "tag16h5")) {
        tag16h5_destroy(tf);
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tagCircle21h7_destroy(tf);
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tagCircle49h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tagStandard41h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tagStandard52h13_destroy(tf);
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tagCustom48h12_destroy(tf);
    }
    getopt_destroy(getopt);
    return 0;
}