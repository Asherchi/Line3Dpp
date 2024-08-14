/* 
 * Line3D++ - Line-based Multi View Stereo
 * Copyright (C) 2015  Manuel Hofer

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// check libs
#include "configLIBS.h"

// EXTERNAL
#include <tclap/CmdLine.h>
#include <tclap/CmdLineInterface.h>
#include <boost/filesystem.hpp>
#include "eigen3/Eigen/Eigen"

// std
#include <iostream>
#include <fstream>

// opencv
#ifdef L3DPP_OPENCV3
#include <opencv2/highgui.hpp>
#else
#include <opencv/highgui.h>
#endif //L3DPP_OPENCV3

// lib
#include "line3D.h"

// INFO:
// This executable reads colmap results (cameras.txt, images.txt, and points3D.txt) and executes the Line3D++ algorithm.
// If distortion coefficients are stored in the cameras.txt file, you need to use the _original_ (distorted) images!
// 读取的数据只能是colmap 的txt文件 不能是bin文件

/*
    <T>是参数值的类型，"flag"是短选项（可选），"flagname"是长选项，"description"是对这个参数的描述，
    false表示这个参数不是必需的（如果是必需的，应该设置为true），T()是参数的默认值（如果参数未指定时使用），
    "required value"是当使用帮助信息时显示的额外文本，提示用户这个参数需要一个值。
*/

int main(int argc, char *argv[])
{
    TCLAP::CmdLine cmd("LINE3D++");
    // 输入图像的文件
    gTCLAP::ValueAr<std::string> inputArg("i", "input_folder", "folder containing the images", true, "", "string");
    cmd.add(inputArg);
    // 输入colmap的文件 要求是txt的格式 
    TCLAP::ValueArg<std::string> sfmArg("m", "sfm_folder", "full path to the colmap result files (cameras.txt, images.txt, and points3D.txt), if not specified --> input_folder", false, "", "string");
    cmd.add(sfmArg);
    // 输入的文件夹路径
    TCLAP::ValueArg<std::string> outputArg("o", "output_folder", "folder where result and temporary files are stored (if not specified --> sfm_folder+'/Line3D++/')", false, "", "string");
    cmd.add(outputArg);
    // 最大图像宽度 用于线段的检测 默认是-1
    TCLAP::ValueArg<int> scaleArg("w", "max_image_width", "scale image down to fixed max width for line segment detection", false, L3D_DEF_MAX_IMG_WIDTH, "int");
    cmd.add(scaleArg);
    // 相邻图像数量 
    TCLAP::ValueArg<int> neighborArg("n", "num_matching_neighbors", "number of neighbors for matching", false, L3D_DEF_MATCHING_NEIGHBORS, "int");
    cmd.add(neighborArg);
    // sigma a 的值的设定 默认 10 
    TCLAP::ValueArg<float> sigma_A_Arg("a", "sigma_a", "angle regularizer", false, L3D_DEF_SCORING_ANG_REGULARIZER, "float");
    cmd.add(sigma_A_Arg);
    // sigma p 的值的设定 2.5
    TCLAP::ValueArg<float> sigma_P_Arg("p", "sigma_p", "position regularizer (if negative: fixed sigma_p in world-coordinates)", false, L3D_DEF_SCORING_POS_REGULARIZER, "float");
    cmd.add(sigma_P_Arg);
    // 最小极线重叠区域 默认是 0.25
    TCLAP::ValueArg<float> epipolarArg("e", "min_epipolar_overlap", "minimum epipolar overlap for matching", false, L3D_DEF_EPIPOLAR_OVERLAP, "float");
    cmd.add(epipolarArg);
    // knn 匹配保留的数量 
    TCLAP::ValueArg<int> knnArg("k", "knn_matches", "number of matches to be kept (<= 0 --> use all that fulfill overlap)", false, L3D_DEF_KNN, "int");
    cmd.add(knnArg);
    // 每张图像最多分割2d线段的数量 3000
    TCLAP::ValueArg<int> segNumArg("y", "num_segments_per_image", "maximum number of 2D segments per image (longest)", false, L3D_DEF_MAX_NUM_SEGMENTS, "int");
    cmd.add(segNumArg);
    // 相机可以看到最少的3d线段的数量 3 
    TCLAP::ValueArg<int> visibilityArg("v", "visibility_t", "minimum number of cameras to see a valid 3D line", false, L3D_DEF_MIN_VISIBILITY_T, "int");
    cmd.add(visibilityArg);
    // 在聚类之前执行复制器动态扩散 false
    TCLAP::ValueArg<bool> diffusionArg("d", "diffusion", "perform Replicator Dynamics Diffusion before clustering", false, L3D_DEF_PERFORM_RDD, "bool");
    cmd.add(diffusionArg);
    // loadAndStore 加载或存储线段 推荐用于大图像 True
    TCLAP::ValueArg<bool> loadArg("l", "load_and_store_flag", "load/store segments (recommended for big images)", false, L3D_DEF_LOAD_AND_STORE_SEGMENTS, "bool");
    cmd.add(loadArg);
    // 共线的阈值 -1
    TCLAP::ValueArg<float> collinArg("r", "collinearity_t", "threshold for collinearity", false, L3D_DEF_COLLINEARITY_T, "float");
    cmd.add(collinArg);
    // 使用cuda true 
    TCLAP::ValueArg<bool> cudaArg("g", "use_cuda", "use the GPU (CUDA)", false, true, "bool");
    cmd.add(cudaArg);
    // 使用ceres求解
    TCLAP::ValueArg<bool> ceresArg("c", "use_ceres", "use CERES (for 3D line optimization)", false, L3D_DEF_USE_CERES, "bool");
    cmd.add(ceresArg);
    // 正则化深度常数  
    TCLAP::ValueArg<float> constRegDepthArg("z", "const_reg_depth", "use a constant regularization depth (only when sigma_p is metric!)", false, -1.0f, "float");
    cmd.add(constRegDepthArg);

    // read arguments
    cmd.parse(argc,argv);
    std::string inputFolder = inputArg.getValue().c_str();
    std::string sfmFolder = sfmArg.getValue().c_str();

    if(sfmFolder.length() == 0)
        sfmFolder = inputFolder;

    // check if colmap result folder
    boost::filesystem::path sfm(sfmFolder);  // 提供文件系统操作功能的库。
    if(!boost::filesystem::exists(sfm))
    {
        std::cerr << "colmap result folder " << sfm << " does not exist!" << std::endl;
        return -1;  // 这玩意儿都是有编号的 不错 
    }

    std::string outputFolder = outputArg.getValue().c_str();
    if(outputFolder.length() == 0)
        outputFolder = sfmFolder+"/Line3D++/";

    int maxWidth = scaleArg.getValue();
    unsigned int neighbors = std::max(neighborArg.getValue(),2); // 最少要大于2 
    bool diffusion = diffusionArg.getValue();
    bool loadAndStore = loadArg.getValue();
    float collinearity = collinArg.getValue();
    bool useGPU = cudaArg.getValue();
    bool useCERES = ceresArg.getValue();
    float epipolarOverlap = fmin(fabs(epipolarArg.getValue()),0.99f);
    float sigmaA = fabs(sigma_A_Arg.getValue());
    float sigmaP = sigma_P_Arg.getValue();
    int kNN = knnArg.getValue();
    unsigned int maxNumSegments = segNumArg.getValue();
    unsigned int visibility_t = visibilityArg.getValue();
    float constRegDepth = constRegDepthArg.getValue();

    // create output directory
    boost::filesystem::path dir(outputFolder);
    boost::filesystem::create_directory(dir);

    // create Line3D++ object
    L3DPP::Line3D* Line3D = new L3DPP::Line3D(outputFolder,loadAndStore,maxWidth,
                                              maxNumSegments,true,useGPU);  // 实例化一个line3d的类 

    // check if result files exist
    boost::filesystem::path sfm_cameras(sfmFolder+"/cameras.txt"); // 获取sfm的输出以及对应的txt文件 
    boost::filesystem::path sfm_images(sfmFolder+"/images.txt");
    boost::filesystem::path sfm_points3D(sfmFolder+"/points3D.txt");
    if(!boost::filesystem::exists(sfm_cameras) || !boost::filesystem::exists(sfm_images) ||
            !boost::filesystem::exists(sfm_points3D))
    {
        std::cerr << "at least one of the colmap result files does not exist in sfm folder: " << sfm << std::endl;
        return -2;
    }

    std::cout << std::endl << "reading colmap result..." << std::endl;

    // read cameras.txt
    std::ifstream cameras_file;
    cameras_file.open(sfm_cameras.c_str());
    std::string cameras_line;

    std::map<unsigned int,Eigen::Matrix3d> cams_K;  // map是有序的 而 hash存在问题 所以更多使用map map的效率是logn 而hash是 1 
    std::map<unsigned int,Eigen::Vector3d> cams_radial;
    std::map<unsigned int,Eigen::Vector2d> cams_tangential;

    while(std::getline(cameras_file,cameras_line))
    {
        // check first character for a comment (#)
        /* 
            如果两个字符串相等，compare 函数返回 0；如果调用者字符串小于参数字符串，
            返回一个小于 0 的值；如果调用者字符串大于参数字符串，返回一个大于 0 的值
        */
        if(cameras_line.substr(0,1).compare("#") != 0) 
        {   // 解析复杂的字符串，将其分解为多个部分或变量
            std::stringstream cameras_stream(cameras_line);

            unsigned int camID,width,height;
            std::string model;

            // parse essential data
            cameras_stream >> camID >> model >> width >> height;

            double fx,fy,cx,cy,k1,k2,k3,p1,p2;

            // check camera model
            if(model.compare("SIMPLE_PINHOLE") == 0)
            {
                // f,cx,cy
                cameras_stream >> fx >> cx >> cy;
                fy = fx;
                k1 = 0; k2 = 0; k3 = 0;
                p1 = 0; p2 = 0;
            }
            else if(model.compare("PINHOLE") == 0)
            {
                // fx,fy,cx,cy
                cameras_stream >> fx >> fy >> cx >> cy;
                k1 = 0; k2 = 0; k3 = 0;
                p1 = 0; p2 = 0;
            }
            else if(model.compare("SIMPLE_RADIAL") == 0)
            {
                // f,cx,cy,k
                cameras_stream >> fx >> cx >> cy >> k1;
                fy = fx;
                k2 = 0; k3 = 0;
                p1 = 0; p2 = 0;
            }
            else if(model.compare("RADIAL") == 0)
            {
                // f,cx,cy,k1,k2
                cameras_stream >> fx >> cx >> cy >> k1 >> k2;
                fy = fx;
                k3 = 0;
                p1 = 0; p2 = 0;
            }
            else if(model.compare("OPENCV") == 0)
            {
                // fx,fy,cx,cy,k1,k2,p1,p2
                cameras_stream >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2;
                k3 = 0;
            }
            else if(model.compare("FULL_OPENCV") == 0)
            {
                // fx,fy,cx,cy,k1,k2,p1,p2,k3[,k4,k5,k6]
                cameras_stream >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2 >> k3;
            }
            else
            {
                std::cerr << "camera model " << model << " unknown!" << std::endl;
                std::cerr << "please specify its parameters in the main_colmap.cpp in order to proceed..." << std::endl;
                return -3;
            }

            Eigen::Matrix3d K;  // 3x3
            K(0,0) = fx; K(0,1) = 0;  K(0,2) = cx;
            K(1,0) = 0;  K(1,1) = fy; K(1,2) = cy;
            K(2,0) = 0;  K(2,1) = 0;  K(2,2) = 1;

            cams_K[camID] = K;
            cams_radial[camID] = Eigen::Vector3d(k1,k2,k3); // 径向畸变 
            cams_tangential[camID] = Eigen::Vector2d(p1,p2); // 切向畸变
        }
    }
    cameras_file.close();

    std::cout << "found " << cams_K.size() << " cameras in [cameras.txt]" << std::endl;

    // read images.txt
    std::ifstream images_file;
    images_file.open(sfm_images.c_str());
    std::string images_line;

    std::map<unsigned int,Eigen::Matrix3d> cams_R;  // 旋转
    std::map<unsigned int,Eigen::Vector3d> cams_t;  // 平移
    std::map<unsigned int,Eigen::Vector3d> cams_C;  // 这个C是啥 坐标？ 3d marker? 
    std::map<unsigned int,unsigned int> img2cam;    // 图像id 和 cam id的对应 
    std::map<unsigned int,std::string> cams_images; 
    std::map<unsigned int,std::list<unsigned int> > cams_worldpoints;  // list 底层实现是链表 
    std::map<unsigned int,Eigen::Vector3d> wps_coords;  // idx是3d点的下标
    std::vector<unsigned int> img_seq;
    unsigned int imgID,camID;

    bool first_line = true;
    while(std::getline(images_file,images_line))
    {
        // check first character for a comment (#)
        if(images_line.substr(0,1).compare("#") != 0)
        {
            std::stringstream images_stream(images_line);
            if(first_line)
            {
                // image data
                double qw,qx,qy,qz,tx,ty,tz;
                std::string img_name;

                images_stream >> imgID >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camID >> img_name;

                // convert rotation
                if(cams_K.find(camID) != cams_K.end())  // 保证 camId 存在 
                {
                    Eigen::Matrix3d R = Line3D->rotationFromQ(qw,qx,qy,qz);  // world to camra 底下的t也是 
                    Eigen::Vector3d t(tx,ty,tz);
                    Eigen::Vector3d C = (R.transpose()) * (-1.0 * t);  // 这个C是相机的位置 

                    cams_R[imgID] = R;
                    cams_t[imgID] = t;
                    cams_C[imgID] = C;
                    cams_images[imgID] = img_name;  // 这个... 为啥不 直接 idx_images 
                    img2cam[imgID] = camID; 
                    img_seq.push_back(imgID);   // 保存所有的imgID的idx 
                }

                first_line = false;
            }
            else
            {
                if(cams_K.find(camID) != cams_K.end())
                {
                    // 2D points
                    double x,y;
                    std::string wpID;
                    std::list<unsigned int> wps;
                    bool process = true;

                    while(process)
                    {
                        wpID = "";
                        images_stream >> x >> y >> wpID;

                        if(wpID.length() > 0)
                        {
                            int wp = atoi(wpID.c_str());

                            if(wp >= 0)  // wp是-1时 是无效的点 2d点不产生3d点 
                            {
                                wps.push_back(wp);
                                wps_coords[wp] = Eigen::Vector3d(0,0,0);  // 初始化一个3d点
                            }
                        }
                        else
                        {
                            // end reached
                            process = false;
                        }
                    }

                    cams_worldpoints[imgID] = wps;
                }

                first_line = true;
            }
        }
    }
    images_file.close();

    std::cout << "found " << cams_R.size() << " images and " << wps_coords.size() << " worldpoints in [images.txt]" << std::endl;

    // read points3D.txt
    std::ifstream points3D_file;
    points3D_file.open(sfm_points3D.c_str());
    std::string points3D_line;

    while(std::getline(points3D_file,points3D_line))
    {
        // check first character for a comment (#)
        if(images_line.substr(0,1).compare("#") != 0)
        {
            std::stringstream points3D_stream(points3D_line);

            // read id and coords
            double X,Y,Z;
            unsigned int pID;

            points3D_stream >> pID >> X >> Y >> Z;

            if(wps_coords.find(pID) != wps_coords.end())
            {
                wps_coords[pID] = Eigen::Vector3d(X,Y,Z);
            }
        }
    }
    points3D_file.close();

    // load images (parallel)
#ifdef L3DPP_OPENMP
    #pragma omp parallel for
#endif //L3DPP_OPENMP
    for(int i=0; i<img_seq.size(); ++i)
    {
        // get camera params
        unsigned int imgID = img_seq[i];  // 获取imgID 
        unsigned int camID = img2cam[imgID]; // 获取对应的 imgID 对应的camID 

        // intrinsics
        Eigen::Matrix3d K = cams_K[camID];
        Eigen::Vector3d radial = cams_radial[camID];
        Eigen::Vector2d tangential = cams_tangential[camID];

        if(cams_R.find(imgID) != cams_R.end())
        {
            // extrinsics
            Eigen::Matrix3d R = cams_R[imgID];
            Eigen::Vector3d t = cams_t[imgID];
            Eigen::Vector3d C = cams_C[imgID];

            // read image 读取对应的图像数据 类型是灰度图像 
            cv::Mat image = cv::imread(inputFolder+"/"+cams_images[imgID],CV_LOAD_IMAGE_GRAYSCALE);

            // undistort image 如果畸变大于阈值 那么进行去畸变 
            cv::Mat img_undist; 
            if(fabs(radial(0)) > L3D_EPS || fabs(radial(1)) > L3D_EPS || fabs(radial(2)) > L3D_EPS ||
                    fabs(tangential(0)) > L3D_EPS || fabs(tangential(1)) > L3D_EPS)
            {
                // undistorting  图像去畸变 可以学学  直接使用opencv 去畸变的  这里面去畸变原因也很简单 因为需要检测的是直线 而不是曲线 
                Line3D->undistortImage(image,img_undist,radial,tangential,K);
            }
            else
            {
                // already undistorted
                img_undist = image;
            }

            // compute depths
            if(cams_worldpoints.find(imgID) != cams_worldpoints.end())
            {
                std::list<unsigned int> wps_list = cams_worldpoints[imgID];
                std::vector<float> depths;

                std::list<unsigned int>::iterator it = wps_list.begin();
                for(; it!=wps_list.end(); ++it)
                {
                    depths.push_back((C-wps_coords[*it]).norm());  // 这个深度算的是 相机中心到3d点的距离 
                }

                // median depth
                if(depths.size() > 0)   
                {
                    std::sort(depths.begin(),depths.end()); 
                    float med_depth = depths[depths.size()/2];  // 这个也太粗糙了吧 取中值 深度

                    // add image
                    // 这里面会执行 线段的检测 
                    Line3D->addImage(imgID,img_undist,K,R,t,med_depth,wps_list);  // 这个函数为啥有一点不合理 最后一个形参没有传入参数啊... 这样应该不可以吧
                }
            }
        }
    }

    // match images
    Line3D->matchImages(sigmaP,sigmaA,neighbors,epipolarOverlap,
                        kNN,constRegDepth);

    // compute result
    Line3D->reconstruct3Dlines(visibility_t,diffusion,collinearity,useCERES);

    // save end result
    std::vector<L3DPP::FinalLine3D> result;
    Line3D->get3Dlines(result);

    // save as STL
    Line3D->saveResultAsSTL(outputFolder);
    // save as OBJ
    Line3D->saveResultAsOBJ(outputFolder);
    // save as TXT
    Line3D->save3DLinesAsTXT(outputFolder);
    // save as BIN
    Line3D->save3DLinesAsBIN(outputFolder);

    // cleanup
    delete Line3D;
}
