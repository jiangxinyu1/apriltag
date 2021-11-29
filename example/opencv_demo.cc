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
#include <ceres/ceres.h>
#include "ceres/autodiff_cost_function.h"
#include "ceres/local_parameterization.h"

extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
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
        // todo：1 计算tag1和tag2的重投影残差
        Eigen::Matrix<T,3,1> tmp1;
        if ( tagFlag_ == 1 )
        {
            tmp1= KMat_*(R1* real_point_ + t1);
        }
        else if ( tagFlag_ == 2 )
        {
            tmp1= KMat_*(R2* real_point_ + t2);
        }
        T forward_error[2];
        tmp1(0,0) = tmp1(0,0) / tmp1(2,0);
        tmp1(1,0) = tmp1(1,0) / tmp1(2,0);
        tmp1(2,0) = tmp1(2,0) / tmp1(2,0);
        forward_error[0] = tag_point_[0]  - tmp1(0,0);
        forward_error[1]= tag_point_[1] - tmp1(1,0);
        /////////////////////////////////////////////////////////////////////////////////
        // todo : 2 两个z轴的的夹角应为0
        Eigen::Matrix<double,3,1> tt (0,0,1);
        Eigen::Matrix<T,3,1> z1 = R1*tt;
        Eigen::Matrix<T,3,1> z2 = R2*tt;
        // T r2 = ((z1.transpose()*z2).norm() /(z1.norm()*z2.norm())) -1.0 ;
        T r2 = (z1.cross(z2)).norm();
        /////////////////////////////////////////////////////////////////////////////////
        // todo: 3  Camera系下，tag1 的每个点（角点与中心点）与 tag2 的中心点组成的向量与 tag2的pose垂直
        Eigen::Matrix<T,3,1> curPoint;
        Eigen::Matrix<T,3,1> targetCenterPoint;
        Eigen::Matrix<T,3,1> vec_corner_center;
        T r3;
        if ( tagFlag_ == 1 )
        {
            // 使用对应的tag系下的点，计算当前点的深度值
            Eigen::Matrix<T,3,1> tmp =  R1*real_point_+t1;
            curPoint[2] =tmp[2];
            curPoint[0] = curPoint[2]*(tag_point_[0] - KMat_(0,2))/KMat_(0,0);
            curPoint[1] = curPoint[2]*(tag_point_[1] - KMat_(1,2))/KMat_(1,1);
            curPoint = R1.transpose()*(curPoint-t1);
            // 
            targetCenterPoint[2] = t2[2];
            targetCenterPoint[0] = targetCenterPoint[2]*(tag_center_2_point_[0] - KMat_(0,2))/KMat_(0,0);
            targetCenterPoint[1] = targetCenterPoint[2]*(tag_center_2_point_[1] - KMat_(1,2))/KMat_(1,1);
            targetCenterPoint = R1.transpose()*(targetCenterPoint-t1);
            // 
            vec_corner_center = curPoint - targetCenterPoint;
            r3 = (vec_corner_center.transpose()*z2).norm();
        }
        if ( tagFlag_ == 2 )
        {
            // 使用对应的tag系下的点，计算当前点的深度值
            Eigen::Matrix<T,3,1> tmp =  R2*real_point_+t2;
            curPoint[2] =tmp[2];
            curPoint[0] = curPoint[2]*(tag_point_[0] - KMat_(0,2))/KMat_(0,0);
            curPoint[1] = curPoint[2]*(tag_point_[1] - KMat_(1,2))/KMat_(1,1);
            curPoint = R2.transpose()*(curPoint-t2);
            // 
            targetCenterPoint[2] = t1[2];
            targetCenterPoint[0] = targetCenterPoint[2]*(tag_center_1_point_[0] - KMat_(0,2))/KMat_(0,0);
            targetCenterPoint[1] = targetCenterPoint[2]*(tag_center_1_point_[1] - KMat_(1,2))/KMat_(1,1);
            targetCenterPoint = R2.transpose()*(targetCenterPoint-t2);
            vec_corner_center = curPoint - targetCenterPoint;
            r3 = (vec_corner_center.transpose()*z1).norm();
        }
        /////////////////////////////////////////////////////////////////////////////////
        // todo: 3-2  Camera系下，tag1 的每个点（角点与中心点）与 tag2 的中心点组成的向量与 tag2的pose垂直
        Eigen::Matrix<T,3,1> TagPoint_camera;
        Eigen::Matrix<T,3,1> TagCenterPoint_camera;
        Eigen::Matrix<T,3,1> vec_;
        T r4;
        if ( tagFlag_ == 1 )
        {
            TagPoint_camera =   R1*real_point_+t1;
            TagCenterPoint_camera = R2*Eigen::Vector3d(0,0,0)+t2;
            vec_ = TagPoint_camera - TagCenterPoint_camera;
            r4 = (vec_.transpose()*z2).norm();
        }
        if ( tagFlag_ == 2 )
        {
            // 使用对应的tag系下的点，计算当前点的深度值
            TagPoint_camera =   R2*real_point_+t2;
            TagCenterPoint_camera = R1*Eigen::Vector3d(0,0,0)+t1;
            vec_ = TagPoint_camera - TagCenterPoint_camera;
            r4 = (vec_.transpose()*z1).norm();
        }
        /////////////////////////////////////////////////////////////////////////////////
        T r5;
        r5 = ((t1-t2).transpose()*(t1-t2))(0,0)-(0.205*0.205);

        /////////////////////////////////////////////////////////////////////////////////
        // std::cout << "forward_error = " << forward_error[0] << "\n";
        // std::cout << "r2 = " << r2<< "\n";
        // std::cout << "r4 = " << r4 << "\n";
        // std::cout << "r5 = " << r5 << "\n";
        residual[0] = forward_error[0] ;// 重投影误差第一项
        residual[1] = forward_error[1] ; // 重投影误差第二项
        residual[2] = r2*100.0; // 给权重
        residual[3] = r4*100.0; // 给权重
        residual[4] = r5*100.0; // 给权重

        /////////////////////////////////////////////////////////////////////////////////
        return true;
    } // operator ()

private:
    const Eigen::Vector3d tag_point_;
    const Eigen::Vector3d tag_center_1_point_; // 图像系下中心点（u,v,1）
    const Eigen::Vector3d tag_center_2_point_;
    const Eigen::Vector3d real_point_;
    Eigen::Matrix3d KMat_;
    int tagFlag_;
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CostFunctor2{
public:
    // 构造函数
    CostFunctor2(const Eigen::Vector3d& tag_point, 
                            const  Eigen::Vector3d& tag_center_1_point,
                            const  Eigen::Vector3d& tag_center_2_point,
                            const Eigen::Vector3d& real_point,
                            const Eigen::Matrix3d& KMat,
                            const int tagFlag):
        tag_point_(tag_point), tag_center_1_point_(tag_center_1_point), tag_center_2_point_(tag_center_2_point),real_point_(real_point),KMat_(KMat) ,tagFlag_(tagFlag){}

    // 定义残差项计算方法
    template <typename T>
    bool operator()(const T* const RT, T* residual) const 
    {
        // TODO : 将待优化参数组织成旋转矩阵R1和R2，t1,t2
        // std::cout << "Cost [RT_] = " << RT[0] <<","<<  RT[1] <<","<<  RT[2] <<","<<  RT[3] <<","<<  RT[4] <<","<<  RT[5] <<","<<  RT[6] << "\n";
        Eigen::AngleAxis<T> rotation_vec1(RT[0],Eigen::Matrix<T,3,1>(RT[1],RT[2],RT[3]));
        Eigen::Matrix<T,3,1> t1(RT[4],RT[5],RT[6]);
        // std::cout << "Cost rotation_vec1 angle = " << endl << rotation_vec1.angle() << "\n";
        // std::cout << "Cost rotation_vec1 axis = " << endl << rotation_vec1.axis() << "\n";
        // std::cout << "Cost rotation_vec1" << endl << rotation_vec1.matrix() << endl;
        Eigen::AngleAxis<T> rotation_vec2(RT[7],Eigen::Matrix<T,3,1>(RT[8],RT[9],RT[10]));
        Eigen::Matrix<T,3,1> t2(RT[11],RT[12],RT[13]);
        Eigen::Matrix<T,3,3> R1 = rotation_vec1.toRotationMatrix();
        Eigen::Matrix<T,3,3> R2 =rotation_vec2.toRotationMatrix();

        // std::cout << " cost rotation_vec1 = " << std::endl << rotation_vec1.matrix() << "\n";
        // std::cout << " cost R1 = "<< std::endl  << R1 << "\n";
        // std::cout << " cost t1 = "<< std::endl  << t1 << "\n";
        /////////////////////////////////////////////////////////////////////////////////
        // todo：1 计算tag1和tag2的重投影残差
        Eigen::Matrix<T,3,1> tmp1;
        if ( tagFlag_ == 1 )
        {
            tmp1= KMat_*(R1* real_point_ + t1);
        }
        else if ( tagFlag_ == 2 )
        {
            tmp1= KMat_*(R2* real_point_ + t2);
        }
        T forward_error[2];
        tmp1(0,0) = tmp1(0,0) / tmp1(2,0);
        tmp1(1,0) = tmp1(1,0) / tmp1(2,0);
        tmp1(2,0) = tmp1(2,0) / tmp1(2,0);
        forward_error[0] = tag_point_[0]  - tmp1(0,0);
        forward_error[1]= tag_point_[1] - tmp1(1,0);

        // todo : 2 
        residual[0] = (forward_error[0] *forward_error[0])+(forward_error[1]*forward_error[1]);
        residual[1] = (1.0 - (RT[1]*RT[1]+RT[2]*RT[2]+RT[3]*RT[3]));
        residual[2] = (1.0 - (RT[8]*RT[8]+RT[9]*RT[9]+RT[10]*RT[10]));

        return true;
    } // operator ()

private:
    const Eigen::Vector3d tag_point_;
    const Eigen::Vector3d tag_center_1_point_;
    const Eigen::Vector3d tag_center_2_point_;
    const Eigen::Vector3d real_point_;
    Eigen::Matrix3d KMat_;
    int tagFlag_;
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool blShowImage = false;

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
void poseOptimizationAll(const std::vector<Eigen::Vector3d>& tag1_points, 
                                             const std::vector<Eigen::Vector3d>& tag2_points,
                                             const Eigen::Matrix3d &K,
                                             Eigen::Matrix3d & R1, Eigen::Vector3d & t1,
                                             Eigen::Matrix3d & R2, Eigen::Vector3d & t2 );
void poseOptimization(const std::vector<Eigen::Vector3d>& tag1_points, 
                                             const std::vector<Eigen::Vector3d>& tag2_points,
                                             const Eigen::Matrix3d &K,
                                             Eigen::Matrix3d & R1, Eigen::Vector3d & t1,
                                             Eigen::Matrix3d & R2, Eigen::Vector3d & t2 );                                             
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
    // matd_t* tmp =  matd_create_data(3, 3, (double[]) { A[8], A[17], A[26], A[35], A[44], A[53], A[62], A[71], 1 });
    H << A[8], A[17], A[26], A[35], A[44], A[53], A[62], A[71], 1;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void preBuildDistortedLookupTable(std::vector<std::vector<distortion_uv >> &lookupTable,const int width, const int height)
{
    // 畸变参数
    double k1 =-0.338011, k2 = 0.130450, p1 = 0.000287, p2 =0.000001 ,k3=  -0.024906;
    // 内参 
    double fx = 934.166126/2, fy = 935.122766/2, cx = 1360/2/2, cy =  680/2/2;
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
    // 畸变参数
    double k1 =-0.338011, k2 = 0.130450, p1 = 0.000287, p2 =0.000001 ,k3=  -0.024906;
    // 内参 
    double fx = 934.166126, fy = 935.122766, cx = 960.504061-300, cy =  562.707915-200;
    // 初始化
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
    unsigned char* data = (unsigned char*) malloc(width*height*3/2);
    fread(data,height*width*sizeof(unsigned char),1,pFileIn);
	grayImg.create(height,width, CV_8UC1);
    memcpy(grayImg.data, data, height*width*sizeof(unsigned char));
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
    // 畸变参数
    double k1 =-0.338011, k2 = 0.130450, p1 = 0.000287, p2 =0.000001 ,k3=  -0.024906;
    // 内参
    double fx = 934.166126/2, fy = 935.122766/2, cx = 1360/2/2, cy =  680/2/2;

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
    if(!img.data || absolutePath.empty())
    {
        return false;
    }
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
	fread(pYuvBuf, bufLen*sizeof(unsigned char), 1, pFileIn);
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
void poseOptimizationAll(const std::vector<Eigen::Vector3d>& tag1_points, 
                                             const std::vector<Eigen::Vector3d>& tag2_points,
                                             const Eigen::Matrix3d &K,
                                             Eigen::Matrix3d & R1, Eigen::Vector3d & t1,
                                             Eigen::Matrix3d & R2, Eigen::Vector3d & t2 )
{
    const std::vector<Eigen::Vector3d> real_points{ Eigen::Vector3d (-0.03,0.03,0.0), Eigen::Vector3d (0.03,0.03,0.0), Eigen::Vector3d (0.03,-0.03,0.0),
                                                                                        Eigen::Vector3d (-0.03,-0.03,0.0), Eigen::Vector3d (0.0,0.0,0.0)};

    // 旋转矩阵转成旋转向量
    Eigen::AngleAxisd rotation_vec1;
    rotation_vec1.fromRotationMatrix(R1);
    Eigen::AngleAxisd rotation_vec2;
    rotation_vec2.fromRotationMatrix(R2);

    double RT_[14] = {rotation_vec1.angle(),rotation_vec1.axis()[0],rotation_vec1.axis()[1],rotation_vec1.axis()[2],t1[0],t1[1],t1[2],
                                    rotation_vec2.angle(),rotation_vec2.axis()[0],rotation_vec2.axis()[1],rotation_vec2.axis()[2],t2[0],t2[1],t2[2]};


    std::cout << "[ Before RT_] = " << RT_[0] <<","<<  RT_[1] <<","<<  RT_[2] <<","<<  RT_[3] <<","<<  RT_[4] <<","<<  RT_[5] <<","<<  RT_[6] << "\n";
    // Build the problem.
    ceres::Problem problem;
    for(int i = 0 ; i < 4; i++)
    {
        CostFunctor2 *Cost_functor = new CostFunctor2 (tag1_points[i],tag1_points[4],tag2_points[4],real_points[i],K,1);
        problem.AddResidualBlock( new AutoDiffCostFunction<CostFunctor2,3,14> (Cost_functor), nullptr,RT_);
    }
    for(int i = 0 ; i < 4; i++)
    {
        CostFunctor2 *Cost_functor2 = new CostFunctor2 (tag2_points[i],tag1_points[4],tag2_points[4],real_points[i],K,2);
        problem.AddResidualBlock(new AutoDiffCostFunction<CostFunctor2,3,14> (Cost_functor2), nullptr ,RT_);
    }
    // Run the solver!
    ceres::Solver::Options solver_options;//实例化求解器对象
    solver_options.linear_solver_type=ceres::DENSE_NORMAL_CHOLESKY;
    solver_options.minimizer_progress_to_stdout= true;
    //实例化求解对象
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options,&problem,&summary);
    std::cout << "[ After RT_] = " << RT_[0] <<","<<  RT_[1] <<","<<  RT_[2] <<","<<  RT_[3] <<","<<  RT_[4] <<","<<  RT_[5] <<","<<  RT_[6] << "\n";
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
    std::cout << "[Before] t1  = \n" << t1[0] << "," <<t1[1] << "," << t1[2]<< std::endl;
    std::cout << "[Before] t2  = \n" << t2[0] << "," <<t2[1] << "," << t2[2]<< std::endl;                                                                              

    // 旋转矩阵转成四元数
    Eigen::Quaterniond quaternion1(R1);
    Eigen::Quaterniond quaternion2(R2);

    // Build the problem.
    ceres::Problem problem;
    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;

    for(int i = 0 ; i < 5; i++)
    {
        CostFunctor *Cost_functor1 = new CostFunctor (tag1_points[i],tag1_points[4],tag2_points[4],real_points[i],K,1);
        problem.AddResidualBlock( new AutoDiffCostFunction<CostFunctor,5,4,3,4,3> (Cost_functor1), nullptr,
                                                         quaternion1.coeffs().data(),t1.data(),quaternion2.coeffs().data(),t2.data());
        problem.SetParameterization(quaternion1.coeffs().data(), quaternion_local_parameterization);
        problem.SetParameterization(quaternion2.coeffs().data(), quaternion_local_parameterization);
    }
    for(int i = 0 ; i < 5; i++)
    {
        CostFunctor *Cost_functor2 = new CostFunctor (tag2_points[i],tag1_points[4],tag2_points[4],real_points[i],K,2);
        problem.AddResidualBlock( new AutoDiffCostFunction<CostFunctor,5,4,3,4,3> (Cost_functor2), nullptr,
                                                         quaternion1.coeffs().data(),t1.data(),quaternion2.coeffs().data(),t2.data());
        problem.SetParameterization(quaternion1.coeffs().data(), quaternion_local_parameterization);
        problem.SetParameterization(quaternion2.coeffs().data(), quaternion_local_parameterization);
    }
    // todo : Solve
    ceres::Solver::Options solver_options;//实例化求解器对象    
    solver_options.linear_solver_type=ceres::DENSE_NORMAL_CHOLESKY;
    solver_options.minimizer_progress_to_stdout= true;
    //实例化求解对象
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options,&problem,&summary);

    // Todo: print result
    R1 = quaternion1.toRotationMatrix();
    R2 = quaternion2.toRotationMatrix();
    Eigen::Vector3d m = R1.col(2);
    Eigen::Vector3d n = R2.col(2);
    auto theta = std::acos((double)(m.transpose()*n) / (m.norm()*n.norm()));
    printf ("[AFTER]-----------------------------------The angle diff between id-3 and id-6 = %f \n",theta * 180/3.14159);
    std::cout << "R1.transpose()*R1 = \n" << (R1.transpose()*R1) << std::endl;
    std::cout << "R2.transpose()*R2= \n" << (R2.transpose()*R2)<< std::endl;
    std::cout << "[AFTER] t1  = \n" << t1[0] << "," <<t1[1] << "," << t1[2]<< std::endl;
    std::cout << "[AFTER] t2  = \n" << t2[0] << "," <<t2[1] << "," << t2[2]<< std::endl;
    // std::cout << summary.FullReport() << '\n';
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
    std::string path = "/home/xinyu/workspace/360/OFei-RGB/yuv2rgb/pic/";

    getopt_t *getopt = getopt_create();

    Eigen::Matrix3d K ;

    getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
    getopt_add_bool(getopt, 'd', "debug", 0, "Enable debugging output (slow)");
    getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
    getopt_add_string(getopt, 'f', "family", "tag16h5", "Tag family to use");
    getopt_add_int(getopt, 'i', "iters", "1", "Repeat processing on input set this many times");
    getopt_add_int(getopt, 't', "threads", "2", "Use this many CPU threads");
    getopt_add_int(getopt, 'a', "hamming", "1", "Detect tags with up to this many bit errors.");
    getopt_add_double(getopt, 'x', "decimate", "2.0", "Decimate input image by this factor");
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

    std::vector<std::vector<distortion_uv_4>> distortLookupTable;
    preBuildDistortedLookupTable(distortLookupTable,(1920-600),(1080-400));

    Mat gray, rgbImage,rgbImageRaw;
    const int testNumber = 10;

    for ( int imageIndex = 4 ; imageIndex < testNumber; imageIndex++)
    {
        std::cout << "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NEW IMAGE <<<<<<<<<<<<<<<<<<< diatance = "<< imageIndex*10 <<"cm \n";

        //  TODO： 1 循环读取YUV将其转成GRAY
        auto t1 = getTime();
        std::string new_path = "/data/rgb/"+std::to_string(imageIndex)+".yuv";
        // YU12toRGB(new_path,rgbImageRaw,1920,1080,0);
        YUV4202GRAY_CV_SAVE(new_path,rgbImageRaw,1920,1080);
        auto t2 = getTime();
        // std::cout << "[Time] YU12toRGB = " << t2-t1 << "ms\n";
        // TODO : 2 裁剪
        auto my_select = cv::Rect(300,200,1320,680);
        rgbImage = rgbImageRaw(my_select);
        auto t21 = getTime();
        // std::cout << "[Time] cut Image = " << t21-t2 << "ms\n";
        // TODO :  3 图像降采样
        int rows = rgbImage.rows, cols = rgbImage.cols;
        cv::Mat newFrame = rgbImage;
        // cv::pyrDown(rgbImage,newFrame,cv::Size(cols/2,rows/2));
        // std::cout << newFrame.rows << "." << newFrame.cols<< "\n";
        auto t3 = getTime();
        // std::cout << "[Time] pyrDown = " << t3-t21 << "ms\n";
        // TODO : 4 对RGB图像去畸变
        cv::Mat frame1 = cv::Mat(newFrame.rows, newFrame.cols, CV_8UC1);   // 去畸变以后的图
        // myImageDistorted(newFrame,frame);
        myImageDistorted(newFrame,frame1,distortLookupTable);
        //  TODO : 5 直方图均衡化
        cv::Mat frame = frame1;
        // cv::equalizeHist(frame1,frame);
        auto t4 = getTime();
        // std::cout << "[Time] myImageDistorted = " << t4-t3 << "ms\n";
        // TODO : 6   RGB转成灰度图  
        // cvtColor(frame, gray, COLOR_BGR2GRAY);
        // gray = frame;
        auto t5 = getTime();
        // std::cout << "[Time] cvtColor = " << t5-t4 << "ms\n";

        // Make an image_u8_t header for the Mat data
        image_u8_t im = 
        { 
            .width = frame.cols,
            .height = frame.rows,
            .stride = frame.cols,
            .buf = frame.data
        };

        // TODO : 7 检测tag，并计算H  
        zarray_t *detections = apriltag_detector_detect(td, &im);
        
        auto t6 = getTime();
        // std::cout << "[Time] apriltag_detector_detect = " << t6-t5 << "ms\n";
        auto t7_ = getTime();

        Eigen::Vector3d rotation_z_3;
        Eigen::Vector3d rotation_z_6;
        bool id3ready = false , id6ready =false;
        Eigen::Matrix3d rotation_matrix;

        // cv::Mat init_frame;
        cv::Mat image_with_init_corners = frame1.clone();
        cv::Mat image_with_gftt_corners = frame1.clone();
        cv::Mat image_with_subpixel_corners = frame1.clone();
        std::vector<Eigen::Vector3d> tag1_points; // 用于存储Tag1的图像上角点及中心点
        std::vector<Eigen::Vector3d> tag2_points; // 用于存储Tag2的图像上角点及中心点
        Eigen::Matrix3d rotationMatrixTag1; 
        Eigen::Matrix3d rotationMatrixTag2;
        Eigen::Vector3d tranVecTag1;
        Eigen::Vector3d tranVecTag2;

        /*  
        *   遍历每个检测结果
        */
        for (int i = 0; i < zarray_size(detections); i++) 
        {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);

            if (det->id != 3 && det->id != 6)
            {
                continue;
            }

            std::vector<cv::Point2f> corners;
            corners.resize(4);
            ///////////////////////////////////////////////////////////////////////////
            auto printDetectResult = [&]()
            {
                printf("\n<<<<<<<<<<<< TAG ID = %i, decision_margin = %f\n", det->id,det->decision_margin);
                // print conner and center
                printf("detection result : \n");
                for (int i = 0 ; i < 4; i++)
                {
                    cv::Point2f tmp(det->p[i][0],det->p[i][1]);
                    corners[i] = tmp;
                    // printf("detect conner %i =(%f , %f) \n",i,det->p[i][0],det->p[i][1]);
                }
                // printf("center = %f ,%f \n",det->c[0],det->c[1]);
                // // print Homography Matrix
                // printf("Homography Matrix :  \n %f,%f,%f \n %f,%f,%f\n  %f,%f,%f\n", det->H->data[0],det->H->data[1],det->H->data[2],det->H->data[3],
                //             det->H->data[4],det->H->data[5],det->H->data[6],det->H->data[7],det->H->data[8]);
            };
            printDetectResult();
            ///////////////////////////////////////////////////////////////////////////


            // todo: 计算corners的周围7*7的点的最小特征值
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
            // 遍历每个角点周围的11*11响应矩阵，在5*5范围内找最大的最小特征值（像素坐标）
            int count = 0;
            std::vector<cv::Point2f> corners_after_gftt;
            corners_after_gftt.resize(4);
            corners_after_gftt = corners;
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

            // todo:  do cornerSubPix
            std::vector<cv::Point2f> corners_final;
            corners_final = corners_after_gftt;
            // cv::cornerSubPix(frame, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            cv::cornerSubPix(frame, corners_after_gftt, cv::Size(3, 3), cv::Size(-1, -1), cv::TermCriteria(2+1, 1000000, 0.001));


            ///////////////////////////////////////////////////////////////////////////
            // todo : 确定最终的角点

            for(int i = 0; i < corners_final.size(); i++)
            {
                // frame.at<uint8_t>(std::round(corners_final[i].y),std::round(corners_final[i].x)) = 255;
                // cv::circle(frame, corners_final[i], 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                printf("corners_final %i = %f ,%f\n",i,corners_final[i].x,corners_final[i].y);
            }
            ///////////////////////////////////////////////////////////////////////////
            // TODO: 计算新的单应矩阵
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

            Eigen::Matrix3d newH;
            homography_compute3(corr_arr,newH);

            std::cout << "New Homography Matrix: \n"<< newH << "\n";
            ///////////////////////////////////////////////////////////////////////////
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
                    det->p[i][0] = corners[i].x;
                    det->p[i][1]  = corners[i].y;
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
            Size textsize = getTextSize(text, fontface, fontscale, 2,
                                            &baseline);
            putText(frame, text, Point(det->c[0]-textsize.width/2,
                                       det->c[1]+textsize.height/2),
                    fontface, fontscale, Scalar(0xff, 0x99, 0), 2);

            // TODO : 3 estimate_tag_pose
            // First create an apriltag_detection_info_t struct using your known parameters.
            apriltag_detection_info_t info;
            info.det = det;
            info.tagsize = 0.06;
            info.fx = 934.166126;
            info.fy = 935.122766;
            info.cx = (960.504061-300);
            info.cy = (562.707915-200);
            K << info.fx,0,info.cx,0,info.fy,info.cy,0,0,1;
            // Then call estimate_tag_pose.
            apriltag_pose_t pose;
            double err = estimate_tag_pose(&info, &pose);

            // 将位姿估计的结果整理成旋转矩阵
            rotation_matrix << pose.R->data[0],pose.R->data[1],pose.R->data[2],pose.R->data[3],pose.R->data[4],pose.R->data[5],pose.R->data[6],pose.R->data[7],pose.R->data[8];

            if ( det->id == 3 )
            {
                rotation_z_3 << rotation_matrix(0,2) , rotation_matrix(1,2), rotation_matrix(2,2);
                tag1_points.resize(5);
                for(int index = 0 ; index < 4; index++ )
                {
                    tag1_points[index] = Eigen::Vector3d(double(corners_final[index].x) , double(corners_final[index].y) ,1.0);
                }
                tag1_points[4] = Eigen::Vector3d (double(det->c[0]),double(det->c[1]),1.0);
                rotationMatrixTag1 = rotation_matrix;
                tranVecTag1 = Eigen::Vector3d (double(pose.t->data[0]),double(pose.t->data[1]),double(pose.t->data[2]));
                id3ready = true;
            }
            if ( det->id == 6 )
            {
                rotation_z_6 << rotation_matrix(0,2), rotation_matrix(1,2), rotation_matrix(2,2);
                tag2_points.resize(5);
                for(int index = 0 ; index < 4; index++ )
                {
                    tag2_points[index] = Eigen::Vector3d(double(corners_final[index].x) , double(corners_final[index].y) ,1.0);
                }
                tag2_points[4] = Eigen::Vector3d (double(det->c[0]),double(det->c[1]),1.0);
                rotationMatrixTag2 = rotation_matrix;
                tranVecTag2=  Eigen::Vector3d (double(pose.t->data[0]),double(pose.t->data[1]),double(pose.t->data[2]));
                id6ready = true;
            }
            
            // 
            auto reprojection = [&]()
            {
                Eigen::Vector3d transform_vector;
                transform_vector << pose.t->data[0],pose.t->data[1],pose.t->data[2];
                Eigen::Vector3d tagPoint0(-0.03,0.03,0.0);
                Eigen::Vector3d tagPoint1(0.03,0.03,0.0);
                Eigen::Vector3d tagPoint2(0.03,-0.03,0.0);
                Eigen::Vector3d tagPoint3(-0.03,-0.03,0.0);
                Eigen::Vector3d tagCenter(0.0,0.0,0.0);

                Eigen::Vector3d c0 = K*(rotation_matrix* tagPoint0 + transform_vector); 
                Eigen::Vector3d c1 = K*(rotation_matrix* tagPoint1 + transform_vector); 
                Eigen::Vector3d c2 = K*(rotation_matrix* tagPoint2 + transform_vector); 
                Eigen::Vector3d c3 = K*(rotation_matrix* tagPoint3 + transform_vector); 
                Eigen::Vector3d c = K*(rotation_matrix* tagCenter + transform_vector); 

                printf("reprojection result : \n");
                printf("c0 =  %f, %f ,%f\n",c0[0]/c0[2],c0[1]/c0[2],c0[2]/c0[2]);
                printf("c1 =  %f, %f ,%f\n",c1[0]/c1[2],c1[1]/c1[2],c1[2]/c1[2]);
                printf("c2 =  %f, %f ,%f\n",c2[0]/c2[2],c2[1]/c2[2],c2[2]/c2[2]);
                printf("c3 =  %f, %f ,%f\n",c3[0]/c3[2],c3[1]/c3[2],c3[2]/c3[2]);
                printf("center =  %f, %f ,%f\n",c[0]/c[2],c[1]/c[2],c[2]/c[2]);

                // frame.at<uint8_t>(std::round(c0[1]/c0[2]),std::round(c0[0]/c0[2])) = 255;
                // frame.at<uint8_t>(std::round(c1[1]/c1[2]),std::round(c1[0]/c1[2])) = 255;
                // frame.at<uint8_t>(std::round(c2[1]/c2[2]),std::round(c2[0]/c2[2])) = 255;
                // frame.at<uint8_t>(std::round(c3[1]/c3[2]),std::round(c3[0]/c3[2])) = 255;
                // frame.at<uint8_t>(std::round(c[1]/c[2]),std::round(c[0]/c[2])) = 255;
            };
            // reprojection();

            
            for(int corner_id = 0; corner_id < corners_after_gftt.size(); corner_id++)
            {
                image_with_init_corners.at<uint8_t>(std::round(corners[corner_id].y),std::round(corners[corner_id].x)) = 255;
                cv::circle(image_with_init_corners, corners[corner_id], 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                image_with_gftt_corners.at<uint8_t>(std::round(corners_after_gftt[corner_id].y),std::round(corners_after_gftt[corner_id].x)) = 255;
                cv::circle(image_with_gftt_corners, corners_after_gftt[corner_id], 5, cv::Scalar(0, 255, 0), 2, 8, 0);
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
                printf (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>The angle diff between id-3 and id-6 = %f \n",theta * 180/3.14159);
            }
        };
        printEstimateTagPose();

        // todo：当前图像同时正确检测到两个Tag时，构建约束优化位姿 R1 t1 R2 t2
        if ( id3ready && id6ready )
        {                
            poseOptimization(tag1_points,tag2_points,K,rotationMatrixTag1,tranVecTag1,rotationMatrixTag2,tranVecTag2);
            // poseOptimizationAll(tag1_points,tag2_points,K,rotationMatrixTag1,tranVecTag1,rotationMatrixTag2,tranVecTag2);
        }
        //  优化后再次进行重投影

        
        
        

        auto t8 = getTime();
        // std::cout << "[Time] estimate_tag_pose = " << t8-t7_<< "ms\n";
        // std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>[Time] all time = " << t8-t1<< "ms\n" << std::endl;
        apriltag_detections_destroy(detections);


        std::string out_path0 = "/data/rgb/res_"+std::to_string(imageIndex)+".jpg";
        SaveImage(frame,out_path0);
        std::string out_path1 = "/data/rgb/image_with_init_corners_"+std::to_string(imageIndex)+".jpg";
        SaveImage(image_with_init_corners,out_path1);
        std::string out_path2 = "/data/rgb/image_with_gftt_corners_"+std::to_string(imageIndex)+".jpg";
        SaveImage(image_with_gftt_corners,out_path2);
        std::string out_path3 = "/data/rgb/image_with_subpixel_corners_"+std::to_string(imageIndex)+".jpg";
        SaveImage(image_with_subpixel_corners,out_path3);

        // pinjit
        std::vector<cv::Mat> imageVec;
        cv::Mat combineImage;
        imageVec.push_back(image_with_init_corners);
        imageVec.push_back(image_with_gftt_corners);
        imageVec.push_back(image_with_subpixel_corners);
        cv::vconcat(imageVec,combineImage);
        std::string out_path4 = "/data/rgb/combineImage_"+std::to_string(imageIndex)+".jpg";
        SaveImage(combineImage,out_path4);

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
