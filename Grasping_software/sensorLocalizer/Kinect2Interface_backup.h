#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <Eigen/Core>

bool stop = false;

enum processor{
	CPU, OPENCL, OPENGL
};

void sigint_handler(int s)
{
	stop = true;
}

struct Kineckt2Frame{
  cv::Mat rgb_img;
  cv::Mat depth_img;
  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> point_cloud;
  std::pair<cv::Mat, cv::Mat> rgb_depth_registered;
};

struct Kinect2Params{
  libfreenect2::Freenect2Device::ColorCameraParams rgb_params;
  libfreenect2::Freenect2Device::IrCameraParams depth_params;
};

class Kinect2Interface {

public:

    Kinect2Interface(processor p, bool mirror = 1): mirror_(mirror), listener_(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth), undistorted_(512, 424, 4), registered_(512, 424, 4),big_mat_(1920, 1082, 4),qnan_(std::numeric_limits<float>::quiet_NaN()){

      signal(SIGINT,sigint_handler);

      if(freenect2_.enumerateDevices() == 0)
      {
          std::cout << "no kinect2 connected!" << std::endl;
          exit(-1);
      }

      serial_ = freenect2_.getDefaultDeviceSerialNumber();
      switch(p){
        case CPU:
          std::cout << "creating CPU processor" << std::endl;
          pipeline_ = new libfreenect2::CpuPacketPipeline();
          break;
#ifdef HAVE_OPENCL
        case OPENCL:
          std::cout << "creating OpenCL processor" << std::endl;
          pipeline_ = new libfreenect2::OpenCLPacketPipeline();
          break;
#endif
        case OPENGL:
          std::cout << "creating OpenGL processor" << std::endl;
          pipeline_ = new libfreenect2::OpenGLPacketPipeline();
          break;
        default:
          std::cout << "creating CPU processor" << std::endl;
          pipeline_ = new libfreenect2::CpuPacketPipeline();
          break;
      }

      dev_ = freenect2_.openDevice(serial_, pipeline_);
      dev_->setColorFrameListener(&listener_);
      dev_->setIrAndDepthFrameListener(&listener_);
      dev_->start();

      registration_ = new libfreenect2::Registration(dev_->getIrCameraParams(), dev_->getColorCameraParams());

      prepareMake3D(dev_->getIrCameraParams());
 	}

	libfreenect2::Freenect2Device::IrCameraParams getIrParameters(){
      libfreenect2::Freenect2Device::IrCameraParams ir = dev_->getIrCameraParams();
      return ir;
	}

	libfreenect2::Freenect2Device::ColorCameraParams getRgbParameters(){
      libfreenect2::Freenect2Device::ColorCameraParams rgb = dev_->getColorCameraParams();
      return rgb;
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr getCloud(){
      const short w = undistorted_.width;
      const short h = undistorted_.height;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>(w, h));

      return updateCloud(cloud);
	} 

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr updateCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud){

      listener_.waitForNewFrame(frames_);
      libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
      libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];

      registration_->apply(rgb, depth, &undistorted_, &registered_, true, &big_mat_);
      
      updateCloudInPlace(cloud);
      
      listener_.release(frames_);
      return cloud;
	}
	
	void updateCloudInPlace(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud){

      const std::size_t w = undistorted_.width;
      const std::size_t h = undistorted_.height;
      cv::Mat tmp_itD0(undistorted_.height, undistorted_.width, CV_8UC4, undistorted_.data);
      cv::Mat tmp_itRGB0(registered_.height, registered_.width, CV_8UC4, registered_.data);

      if (mirror_ == true){
        cv::flip(tmp_itD0, tmp_itD0, 1);
        cv::flip(tmp_itRGB0,tmp_itRGB0,1);
      }

      const float * itD0 = (float *) tmp_itD0.ptr();
      const char * itRGB0 = (char *) tmp_itRGB0.ptr();

      pcl::PointXYZRGB * itP = &cloud->points[0];
      bool is_dense = true;

      for(std::size_t y = 0; y < h; ++y){
        const unsigned int offset = y * w;
        const float * itD = itD0 + offset;
        const char * itRGB = itRGB0 + offset * 4;
        const float dy = rowmap(y);

        for(std::size_t x = 0; x < w; ++x, ++itP, ++itD, itRGB += 4 )
        {
          const float depth_value = *itD / 1000.0f;

          if(!std::isnan(depth_value) && !(std::abs(depth_value) < 0.0001)){
            const float rx = colmap(x) * depth_value;
            const float ry = dy * depth_value;
            itP->z = depth_value;
            itP->x = rx;
            itP->y = ry;

            itP->b = itRGB[0];
            itP->g = itRGB[1];
            itP->r = itRGB[2];
          } else {
            itP->z = qnan_;
            itP->x = qnan_;
            itP->y = qnan_;

            itP->b = qnan_;
            itP->g = qnan_;
            itP->r = qnan_;
            is_dense = false;
          }
        }
      }
      cloud->is_dense = is_dense;
  }

	void shutDown(){
      dev_->stop();
      dev_->close();
	}

	cv::Mat getColor(){
      listener_.waitForNewFrame(frames_);
      libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
      cv::Mat tmp(rgb->height, rgb->width, CV_8UC4, rgb->data);
      cv::Mat r;
      if (mirror_ == true) {cv::flip(tmp,r,1);}
      else {r = tmp.clone();}

      listener_.release(frames_);
      return std::move(r);
	}

	cv::Mat getDepth(){
      listener_.waitForNewFrame(frames_);
      libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
      cv::Mat tmp(depth->height, depth->width, CV_8UC4, depth->data);
      cv::Mat r;
      if (mirror_ == true) {cv::flip(tmp,r,1);}
      else {r = tmp.clone();}

      listener_.release(frames_);
      return std::move(r);
	}

	std::pair<cv::Mat, cv::Mat> getDepthRgb(){
      listener_.waitForNewFrame(frames_);
      libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
      libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
      registration_->apply(rgb, depth, &undistorted_, &registered_);
      cv::Mat tmp_depth(undistorted_.height, undistorted_.width, CV_8UC4, undistorted_.data);
      cv::Mat tmp_color(registered_.height, registered_.width, CV_8UC4, registered_.data);
      cv::Mat r = tmp_color.clone();
      cv::Mat d = tmp_depth.clone();
      if (mirror_ == true) {
        cv::flip(tmp_depth, d, 1);
        cv::flip(tmp_color, r,1);
      }
      listener_.release(frames_);
      return std::move(std::pair<cv::Mat, cv::Mat>(r,d));
	}
	
	void getDepthRgbInPlace(cv::Mat& rgb_im, cv::Mat& depth_im, std::pair<cv::Mat, cv::Mat>& registered){
      listener_.waitForNewFrame(frames_);
      libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
      libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
      rgb_im=cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data).clone();
      depth_im=cv::Mat(depth->height, depth->width, CV_8UC4, depth->data).clone();
      registration_->apply(rgb, depth, &undistorted_, &registered_);
      if (mirror_ == true) {
        cv::flip(depth_im, depth_im, 1);
        cv::flip(rgb_im, rgb_im,1);
        cv::flip(registered.second, registered.second, 1);
        cv::flip(registered.first, registered.first,1);
      }
      listener_.release(frames_);
  }

    Kineckt2Frame getKinect2Frame(){
      Kineckt2Frame result;
      result.rgb_img = getColor();
      result.depth_img = getDepth();
      result.rgb_depth_registered = getDepthRgb();
      result.point_cloud = getCloud();
      return result;
    }
    
    void getKinect2Frame(Kineckt2Frame& k2f, bool compute_pointcloud){
      if(k2f.rgb_depth_registered.first.empty())
      {
        k2f.rgb_depth_registered.second=cv::Mat(undistorted_.height, undistorted_.width, CV_8UC4, undistorted_.data);
        k2f.rgb_depth_registered.first=cv::Mat(registered_.height, registered_.width, CV_8UC4, registered_.data);
      }
      getDepthRgbInPlace(k2f.rgb_img, k2f.depth_img, k2f.rgb_depth_registered);
      if(!compute_pointcloud) return;
      if(!k2f.point_cloud)
        k2f.point_cloud=pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>(undistorted_.width, undistorted_.height));
      updateCloudInPlace(k2f.point_cloud);
    }

    Kinect2Params getKinect2InternalParams(){
      Kinect2Params result;
      result.rgb_params = getRgbParameters();
      result.depth_params = getIrParameters();
      return result;
    }

private:

	void prepareMake3D(const libfreenect2::Freenect2Device::IrCameraParams & depth_p)
	{
      const int w = 512;
      const int h = 424;
      float * pm1 = colmap.data();
      float * pm2 = rowmap.data();
      for(int i = 0; i < w; i++)
      {
        *pm1++ = (i-depth_p.cx + 0.5) / depth_p.fx;
      }
      for (int i = 0; i < h; i++)
      {
        *pm2++ = (i-depth_p.cy + 0.5) / depth_p.fy;
      }
	}

    bool mirror_;
	libfreenect2::Freenect2 freenect2_;
	libfreenect2::Freenect2Device * dev_ = 0;
	libfreenect2::PacketPipeline * pipeline_ = 0;
	libfreenect2::Registration * registration_ = 0;
	libfreenect2::SyncMultiFrameListener listener_;
	libfreenect2::FrameMap frames_;
	libfreenect2::Frame undistorted_, registered_, big_mat_;
	Eigen::Matrix<float,512,1> colmap;
	Eigen::Matrix<float,424,1> rowmap;
	std::string serial_;
	int map_[512 * 424]; // will be used in the next libfreenect2 update
	float qnan_;   
};
