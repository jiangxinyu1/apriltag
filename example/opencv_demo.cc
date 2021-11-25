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
    CostFunctor(const Eigen::Vector3d& tag_corner, const Eigen::Vector3d& real_corner, const Eigen::Matrix3d& K):
        tag_corner_(tag_corner), real_corner_(real_corner), K_(K){}

    template <typename T>
    bool operator()(const T* const RT, T* residual) const {

        Eigen::Matrix<T,3,3> rotation_matrix;
        rotation_matrix << RT[0],RT[1],RT[2],RT[3],RT[4],RT[5],RT[6],RT[7],RT[8];
        Eigen::Matrix<T,3,1> trans_vector;
        trans_vector << RT[9],RT[10],RT[11];
        Eigen::Matrix<double,3,1> detect_corner;
        detect_corner << tag_corner_[0],tag_corner_[1],tag_corner_[2];
        Eigen::Matrix<double,3,1> real_corner;
        real_corner << real_corner_[0],real_corner_[1],real_corner_[2];
        // double asd;
        // reprojectionError<double>(rotation_matrix,trans_vector,K_,detect_corner,real_corner,asd);
        // residual[0] = asd;
        // 
#if 1        
        T forward_error[2];
        Eigen::Matrix<T,3,1> tmp= K_*(rotation_matrix* real_corner + trans_vector);
        // 重投影误差归一化
        tmp(0,0) = tmp(0,0) / tmp(2,0);
        tmp(1,0) = tmp(1,0) / tmp(2,0);
        forward_error[0] = detect_corner(0,0) - tmp(0,0);
        forward_error[1] = detect_corner(1,0) - tmp(1,0);
        //计算重投影误差的二范数作为代价函数误差
        residual[0]=forward_error[0]*forward_error[0]+forward_error[1]*forward_error[1];
 #endif       
        return true;
    }
private:
    Eigen::Vector3d tag_corner_;
    Eigen::Vector3d real_corner_;
    Eigen::Matrix3d K_;
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CostFunctor2{
public:
    // 构造函数
    CostFunctor2(const std::vector<Eigen::Vector3d>& tag1_points, 
                            const std::vector<Eigen::Vector3d>& tag2_points,
                            const std::vector<Eigen::Vector3d>& real_points,
                            const Eigen::Matrix3d& KMat):
        tag1_points_(tag1_points), tag2_points_(tag2_points), real_points_(real_points),KMat_(KMat){}

    // 定义残差项计算方法
    template <typename T>
    bool operator()(const T* const RT, T* residual) const {
        /////////////////////////////////////////////////////////////////////////////////
        // TODO : 将待优化参数组织成旋转矩阵R1和R2，t1,t2
        Eigen::Matrix<T,3,3> R1;
        Eigen::Matrix<T,3,3> R2;
        Eigen::Matrix<T,3,1> t1;
        Eigen::Matrix<T,3,1> t2;

        /////////////////////////////////////////////////////////////////////////////////
        // todo：构建重投影残差
        T tag1Error, tag2Error,reprojectionError;
        for(int index = 0 ; index < cornerSize; index++ )
        {
            // 计算tag1角点
            Eigen::Matrix<T,3,1> tmp1= KMat_*(R1* real_points_[index] + t1);
            tmp1(0,0) = tmp1(0,0) / tmp1(2,0);
            tmp1(1,0) = tmp1(1,0) / tmp1(2,0);
            T forward_error[2];
            forward_error[0] = tag1_points_[index][0] - tmp1(0,0);
            forward_error[1] = tag1_points_[index][1] - tmp1(1,0);
            // 计算tag2角点
            Eigen::Matrix<T,3,1> tmp2= KMat_*(R2* real_points_[index] + t2);
            tmp2(0,0) = tmp2(0,0) / tmp2(2,0);
            tmp2(1,0) = tmp2(1,0) / tmp2(2,0);
            T forward_error2[2];
            forward_error2[0] = tag2_points_[index][0] - tmp2(0,0);
            forward_error2[1] = tag2_points_[index][1] - tmp2(1,0);
            //计算重投影误差的二范数作为代价函数误差
            tag1Error += forward_error[0]*forward_error[0]+forward_error[1]*forward_error[1];
            tag2Error += forward_error2[0]*forward_error2[0]+forward_error2[1]*forward_error2[1];
        }
        reprojectionError = tag1Error + tag2Error;
        /////////////////////////////////////////////////////////////////////////////////
        // todo : 


        residual[0] = reprojectionError;
        return true;

    }

private:
    const std::vector<Eigen::Vector3d> tag1_points_;
    const std::vector<Eigen::Vector3d> tag2_points_;
    const std::vector<Eigen::Vector3d> real_points_;
    Eigen::Matrix3d KMat_;
    const int cornerSize = 4;
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//代价函数计算函数
template <typename T>
void reprojectionError( const Eigen::Matrix<T,3,3> &R, //旋转矩阵 R
                                         const Eigen::Matrix<T,3,1> &t,  // t
                                         const Eigen::Matrix<T,3,3> &K, // 相机内参
                                         const Eigen::Matrix<double,3,1> &detect_corner,  // 检测到的角点
                                         const Eigen::Matrix<double,3,1> &real_corner, // 真实角点
                                         double &residual )//代价函数变量
{
    T forward_error[2];
    Eigen::Matrix<T,3,1> tmp= K*(R* real_corner + t);
    // // 重投影误差归一化
    tmp(0,0) = tmp(0,0)/tmp(2,0);
    tmp(1,0) = tmp(1,0)/tmp(2,0);
    forward_error[0] = detect_corner(0,0) - tmp(0,0);
    forward_error[1] = detect_corner(1,0) - tmp(1,0);
    //计算重投影误差的二范数作为代价函数误差
    residual=forward_error[0]*forward_error[0]+forward_error[1]*forward_error[1];
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void homography_compute3(double c[4][4] , Eigen::Matrix3d &H) {
    
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

    // if (blShowImage)
    // {
    //     std::string outputPath = "";
    //     cv::namedWindow("gray_img", CV_WINDOW_NORMAL);
    //     cv::imshow("gray_img",grayImg);
    //     SaveImage(grayImg,outputPath);
    //     cv::waitKey(0); 
    //     cv::cvDestroyWindow("gray_img");
    //     cv::Mat resultImage;
    //     cv::equalizeHist(grayImg,resultImage);
    //     cv::namedWindow("resultImage", CV_WINDOW_NORMAL);
    //     cv::imshow("resultImage",resultImage);
    //     cv::waitKey(0); 
    //     cv::cvDestroyWindow("resultImage");
    // }

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
    // 画图去畸变后图像
    // cv::imshow("image undistorted", image_undistort);
    // cv::waitKey();
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
void poseOptimizationAll(const std::vector<Eigen::Vector3d>& tag1_points, const std::vector<Eigen::Vector3d>& tag2_points,
                                        const Eigen::Matrix3d &K,std::vector<double>&RT1,std::vector<double>&RT2)
{
    std::vector<Eigen::Vector3d> real_points; 
    real_points.resize(5);
    Eigen::Vector3d tagPoint0(-0.03,0.03,0.0);
    Eigen::Vector3d tagPoint1(0.03,0.03,0.0);
    Eigen::Vector3d tagPoint2(0.03,-0.03,0.0);
    Eigen::Vector3d tagPoint3(-0.03,-0.03,0.0);
    Eigen::Vector3d tagCenter(0.0,0.0,0.0);
    real_points[0]= tagPoint0;
    real_points[1]= tagPoint1;
    real_points[2]= tagPoint2;
    real_points[3]= tagPoint3;
    real_points[4]= tagCenter;

    double RT1_[12] = {RT1[0],RT1[1],RT1[2],RT1[3],RT1[4],RT1[5],RT1[6],RT1[7],RT1[8],RT1[9],RT1[10],RT1[11]};
    double RT2_[12] = {RT2[0],RT2[1],RT2[2],RT2[3],RT2[4],RT2[5],RT2[6],RT2[7],RT2[8],RT2[9],RT2[10],RT2[11]};

    // Build the problem.
    Problem problem;

    CostFunctor2 *Cost_functor= new CostFunctor2(tag1_points,tag2_points,real_points,K);
        //ceres自动求导求增量方程<代价函数类型，代价函数维度，优化变量维度>（代价函数）
    problem.AddResidualBlock(new AutoDiffCostFunction<CostFunctor2,1,12> (Cost_functor), new ceres::CauchyLoss(1),RT1_);

    // Run the solver!
    ceres::Solver::Options solver_options;//实例化求解器对象
    //线性求解器的类型，用于计算Levenberg-Marquardt算法每次迭代中线性最小二乘问题的解
    solver_options.linear_solver_type=ceres::DENSE_QR;
    //记录优化过程，输出到cout位置
    solver_options.minimizer_progress_to_stdout= true;
    //ceres求解运算
    //实例化求解对象
    ceres::Solver::Summary summary;
    //求解
    ceres::Solve(solver_options,&problem,&summary);
}

void poseOptimization(const std::vector<Eigen::Vector3d>& tag1_points, const std::vector<Eigen::Vector3d>& tag2_points,
                                        const Eigen::Matrix3d &K,std::vector<double>&RT1,std::vector<double>&RT2)
{
    std::vector<Eigen::Vector3d> real_points; 
    real_points.resize(5);
    Eigen::Vector3d tagPoint0(-0.03,0.03,0.0);
    Eigen::Vector3d tagPoint1(0.03,0.03,0.0);
    Eigen::Vector3d tagPoint2(0.03,-0.03,0.0);
    Eigen::Vector3d tagPoint3(-0.03,-0.03,0.0);
    Eigen::Vector3d tagCenter(0.0,0.0,0.0);
    real_points[0]= tagPoint0;
    real_points[1]= tagPoint1;
    real_points[2]= tagPoint2;
    real_points[3]= tagPoint3;
    real_points[4]= tagCenter;

    double RT1_[12] = {RT1[0],RT1[1],RT1[2],RT1[3],RT1[4],RT1[5],RT1[6],RT1[7],RT1[8],RT1[9],RT1[10],RT1[11]};
    double RT2_[12] = {RT2[0],RT2[1],RT2[2],RT2[3],RT2[4],RT2[5],RT2[6],RT2[7],RT2[8],RT2[9],RT2[10],RT2[11]};

    // Build the problem.
    Problem problem;

    for(int i = 0 ; i < 4; i++)
    {
        CostFunctor *RT1_cost_functor= new CostFunctor(tag1_points[i],real_points[i],K);
        //ceres自动求导求增量方程<代价函数类型，代价函数维度，优化变量维度>（代价函数）
        problem.AddResidualBlock(        
                new AutoDiffCostFunction<CostFunctor,1,12> (RT1_cost_functor), new ceres::CauchyLoss(1),RT1_);
    }
    // // add tag2
    for(int i = 0 ; i < 4; i++)
    {
        CostFunctor *RT2_cost_functor= new CostFunctor(tag2_points[i],real_points[i],K);
        //ceres自动求导求增量方程<代价函数类型，代价函数维度，优化变量维度>（代价函数）
        problem.AddResidualBlock(        
                new AutoDiffCostFunction<CostFunctor,1,12> (RT2_cost_functor), new ceres::CauchyLoss(1),RT2_);
    }
    // Run the solver!
    ceres::Solver::Options solver_options;//实例化求解器对象
    //线性求解器的类型，用于计算Levenberg-Marquardt算法每次迭代中线性最小二乘问题的解
    solver_options.linear_solver_type=ceres::DENSE_QR;
    //记录优化过程，输出到cout位置
    solver_options.minimizer_progress_to_stdout= true;
    //ceres求解运算
    //实例化求解对象
    ceres::Solver::Summary summary;
    //求解
    ceres::Solve(solver_options,&problem,&summary);
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
        std::string new_path = "/data/rgb/left-"+std::to_string(imageIndex)+".yuv";
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
        std::vector<Eigen::Vector3d> tag1_points;
        std::vector<Eigen::Vector3d> tag2_points;
        std::vector<double> RT1;
        std::vector<double> RT2;

        /*  
        *   遍历每个的检测结果
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
                printf("Homography Matrix :  \n %f,%f,%f \n %f,%f,%f\n  %f,%f,%f\n", det->H->data[0],det->H->data[1],det->H->data[2],det->H->data[3],
                            det->H->data[4],det->H->data[5],det->H->data[6],det->H->data[7],det->H->data[8]);
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
            // updateDetwithNewCorners();

#if 1
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
#endif


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
            // Then call estimate_tag_pose.
            apriltag_pose_t pose;
            double err = estimate_tag_pose(&info, &pose);

            rotation_matrix<<pose.R->data[0],pose.R->data[1],pose.R->data[2],pose.R->data[3],pose.R->data[4],pose.R->data[5],pose.R->data[6],pose.R->data[7],pose.R->data[8];
            Eigen::AngleAxisd rotation_vector;
            rotation_vector.fromRotationMatrix(rotation_matrix);
            Eigen::Vector3d eulerAngle=rotation_vector.matrix().eulerAngles(2,1,0); // ZYX 
            // print rotation matrix
            // printf("R = \n%f,%f,%f\n%f,%f,%f\n%f,%f,%f\n"
            // ,pose.R->data[0],pose.R->data[1],pose.R->data[2],pose.R->data[3],pose.R->data[4],pose.R->data[5],pose.R->data[6],pose.R->data[7],pose.R->data[8]);
            // print eulerAngle
            // printf("Angles =  %f, %f ,%f\n",eulerAngle[2]*180/3.14159,eulerAngle[1]*180/3.14159,eulerAngle[0]*180/3.14159);

            if ( det->id == 3 )
            {
                // 
                rotation_z_3 << rotation_matrix(0,2) , rotation_matrix(1,2), rotation_matrix(2,2);
                tag1_points.resize(5);
                for(int index = 0 ; index < 4; index++ )
                {
                    Eigen::Vector3d tmp;
                    tmp << double(corners_final[index].x) , double(corners_final[index].y) ,1.0;
                    tag1_points[index] = tmp;
                }
                Eigen::Vector3d tmp; 
                tmp << double(det->c[0]),double(det->c[1]),1.0;
                tag1_points[4] = tmp;
                RT1.resize(12);
                for (int index =0; index<9; index++)
                {
                    RT1[index] = double(pose.R->data[index]);
                }
                for (int index =9; index<12; index++)
                {
                    RT1[index] = double(pose.t->data[index]);
                }
                id3ready = true;
            }
            if ( det->id == 6 )
            {
                rotation_z_6 << rotation_matrix(0,2), rotation_matrix(1,2), rotation_matrix(2,2);
                tag2_points.resize(5);
                for(int index = 0 ; index < 4; index++ )
                {
                    Eigen::Vector3d tmp;
                    tmp << (double)corners_final[index].x , (double)corners_final[index].y,1.0;
                    tag2_points[index] = tmp;
                }
                Eigen::Vector3d tmp; 
                tmp << (double)det->c[0],(double)det->c[1],1.0;
                tag2_points[4] = tmp;
                RT2.resize(12);
                for (int index =0; index<9; index++)
                {
                    RT2[index] = (double)pose.R->data[index];
                }
                for (int index =9; index<12; index++)
                {
                    RT2[index] = (double)pose.t->data[index];
                }
                id6ready = true;
            }
            

            auto reprojection = [&]()
            {
                
                K << info.fx,0,info.cx,0,info.fy,info.cy,0,0,1;
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

        // todo：优化位姿
        if ( id3ready && id6ready )
        {                
            poseOptimizationAll(tag1_points,tag2_points,K,RT1,RT2);
        }

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