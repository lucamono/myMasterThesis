#include <cstdio>
#include <string>
#include <boost/program_options.hpp>

#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "raster_object_model2D.h"
#include "chamfer_matching.h"

#include "apps_const.h"

namespace po = boost::program_options;
using namespace std;
using namespace cv;

void parseMotionCommands( int key ,cv::Mat &r_vec, cv::Mat &t_vec,
                          double r_inc = 0.01, double t_inc = 0.01 )
{
  switch( key )
  {
    case 'a':
    case 'A':
      r_vec.at<double>(1,0) += r_inc;
      break;
    case 's':
    case 'S':
      r_vec.at<double>(1,0) -= r_inc;
      break;
    case 'w':
    case 'W':
      r_vec.at<double>(0,0) += r_inc;
      break;
    case 'z':
    case 'Z':
      r_vec.at<double>(0,0) -= r_inc;
      break;
    case 'q':
    case 'Q':
      r_vec.at<double>(2,0) += r_inc;
      break;
    case 'e':
    case 'E':
      r_vec.at<double>(2,0) -= r_inc;
      break;
    case cv_ext::KEY_UP:
      t_vec.at<double>(1,0) -= t_inc;
      break;
    case cv_ext::KEY_DOWN:
      t_vec.at<double>(1,0) += t_inc;
      break;
    case cv_ext::KEY_LEFT:
      t_vec.at<double>(0,0) -= t_inc;
      break;
    case cv_ext::KEY_RIGHT:
      t_vec.at<double>(0,0) += t_inc;
      break;
    case cv_ext::KEY_PAGE_UP:
      t_vec.at<double>(2,0) -= t_inc;
      break;
    case cv_ext::KEY_PAGE_DOWN:
      t_vec.at<double>(2,0) += t_inc;
      break;
    default:
      break;
  }
}

int main(int argc, char **argv)
{
  string app_name( argv[0] ), model_filename, camera_filename, image_filename;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "model_filename,m", po::value<string > ( &model_filename )->required(),
    "STL, PLY, OBJ, ...  model file" )
  ( "camera_filename,c", po::value<string > ( &camera_filename )->required(),
    "A YAML file that stores all the camera parameters (see the PinholeCameraModel object)" )
  ( "image_filename,i", po::value<string > ( &image_filename )->required(),
    "Input image filename" )
  ( "color", "Try to load also model colors" );

  po::variables_map vm;
  bool has_color = false;
  try
  {
    po::store ( po::parse_command_line ( argc, argv, desc ), vm );

    if ( vm.count ( "help" ) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
                << endl << endl<<desc;
      return 0;
    }

    if ( vm.count ( "color" ) )
      has_color = true;

    po::notify ( vm );
  }

  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    return -1;
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    return -1;
  }

  cout << "Loading model from file : "<<model_filename<< endl;
  cout << "Loading camera parameters from file : "<<camera_filename<< endl;

  Mat r_vec = (Mat_<double>(3,1) << 0,0,0),
      t_vec = (Mat_<double>(3,1) << 0,0,1.4);


  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile(camera_filename);
  cam_model.setSizeScaleFactor(2);

  RasterObjectModel3DPtr obj_model_ptr( new RasterObjectModel3D() );
  RasterObjectModel3D &obj_model = *obj_model_ptr;

  obj_model.setCamModel( cam_model );
  obj_model.setStepMeters(0.005);
  obj_model.setRenderWindowSize(cam_model.imgSize());
  obj_model.setCentroidOrigOffset();
  obj_model.setUnitOfMeasure(RasterObjectModel::MILLIMETER);
  obj_model.setRenderFOV(M_PI/6);
  obj_model.enableVertexColors(has_color);

  if(!obj_model.setModelFile( model_filename ))
    return -1;

  // obj_model.setMinSegmentsLen(0.01);
  obj_model.computeRaster();

  has_color = obj_model.vertexColorsEnabled();

  std::vector<cv::Point3d> cap_points;
  std::vector< Eigen::Quaterniond > quat_rotations;
  cv_ext::createIcospherePolarCap( cap_points, 2, M_PI/3/*, 1.0, true */);
//   cv_ext::show3DPoints(cap_points);
  cv_ext::sampleRotationsAroundVectors(cap_points, quat_rotations, 36 );

  cout<<"Num rotations :"<<quat_rotations.size()<<endl;

  for( double depth = 1.35; depth <= 1.45; depth+=0.02 )
  {
    for( int i = 0; i < quat_rotations.size(); i++ )
    {
      obj_model.setModelView(quat_rotations[i], Eigen::Vector3d(0.0,0.0,depth));
      obj_model.storeModelView();
    }
  }

  int increment = 4;

  DistanceTransform dc;
//   dc.setDisThreshold(25);


  vector<string> img_list;

  img_list.push_back("Hdr 0_0.png");
  img_list.push_back("Hdr 0_1.png");
  img_list.push_back("Hdr 0_2.png");
  img_list.push_back("Hdr 0_3.png");
//   img_list.push_back("Hdr 1_0.png");
//   img_list.push_back("Hdr 1_1.png");
//   img_list.push_back("Hdr 1_2.png");
//   img_list.push_back("Hdr 1_3.png");


  vector<Scalar> colors;
  colors.push_back(Scalar(0,0,255));
  colors.push_back(Scalar(255,0,0));
  colors.push_back(Scalar(0,255,0));
  colors.push_back(Scalar(255,0,255));

  for( int col = 255; col >= 0; col -= 5 )
    colors.push_back(Scalar(col,col,col));

  Mat src_mask = imread ( "left_mask_0.png",cv::IMREAD_GRAYSCALE ), mask;
  cv::resize(src_mask, mask, cam_model.imgSize());
  for( auto &filename : img_list )
  {
    Mat src_img = imread ( filename,cv::IMREAD_COLOR ),img, img_gray;


    cv::resize(src_img, img, cam_model.imgSize());
    cvtColor( img, img_gray, CV_BGR2GRAY );
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);

    clahe->apply(img_gray,img_gray);


    LSDEdgeDetectorUniquePtr edge_detector_ptr( new LSDEdgeDetector() );
  //   edge_detector_ptr->setPyrNumLevels(2);
    edge_detector_ptr->setScale(2);
    edge_detector_ptr->setQuantizationThreshold(0.7);
    edge_detector_ptr->setMask(mask);
//     edge_detector_ptr->enableWhiteBackground(true);

//     CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
//   //   Best so far...
//     edge_detector_ptr->setLowThreshold(100);
//     edge_detector_ptr->setRatio(3);

    cv::Mat edge_map;
    edge_detector_ptr->setImage(img_gray);
    edge_detector_ptr->getEdgeMap( edge_map );
    cv_ext::showImage(edge_map , "edge_map", true, 10);

    dc.enableParallelism(true);
    dc.setEdgeDetector(std::move(edge_detector_ptr));
    DirectionalChamferMatching dcm( cam_model );
    dcm.setTemplateModel(obj_model_ptr);



    cv_ext::BasicTimer timer;
    ImageTensorPtr dst_map_tensor_ptr;
    dc.computeDistanceMapTensor ( img_gray, dst_map_tensor_ptr, 60, 6);
    dcm.setInput( dst_map_tensor_ptr );
    vector< TemplateMatch > matches;
    dcm.match(colors.size(), matches, (int)increment);

    cout<<"DCM elapsed time ms : "<<timer.elapsedTimeMs()<<endl;

//     for( int i_m = 0; i_m < matches.size(); i_m++ )
    for( int i_m = matches.size() - 1; i_m >= 0; i_m-- )
//     for( auto iter = matches.begin(); iter != matches.end(); iter++ )
    {
      TemplateMatch &match = matches[i_m];
      obj_model.setModelView(match.r_quat, match.t_vec);
      vector<Point2f> proj_pts;
//       cout<<match.img_offset<<endl;
      obj_model.projectRasterPoints(proj_pts);
      for( int j = 0; j < proj_pts.size(); j++ )
      {
        proj_pts[j].x += match.img_offset.x;
        proj_pts[j].y += match.img_offset.y;
      }
//       cv::Mat display = img.clone();
//       cv_ext::drawPoints( display, proj_pts, Scalar(0,0,255) );
//       cv_ext::showImage(display,"display", true, 1000);

//         cv_ext::drawPoints( img, proj_pts, colors[i_m] );
        cv_ext::drawCircles( img, proj_pts, 3, colors[i_m] );

    }
    cv_ext::showImage(img,"display", true, 10);
    cv::waitKey();
  }

//     int i_m = 0;
//     bool found = false;
//     for( auto iter = matches.begin(); iter != matches.end(); iter++, i_m++ )
//     {
//       TemplateMatch &match = *iter;
//     /*
//
//         Rect bb = gt_poses[j].bb;
//         Point2f tl(bb.tl().x,bb.tl().y);
//         Point2f tl_offset = (*iter).second.second;
//         for( int i_p = 0; i_p < raster_pts.size(); i_p++ )
//         {
//           raster_pts[i_p] -= tl;
//           raster_pts[i_p] += tl_offset;
//         }*/
//
//
//       Eigen::Matrix3d rot_mat_gt;
//
//       cv_ext::exp2RotMat(gt_poses[i].r_vec, rot_mat_gt);
//
//       Eigen::Quaterniond q_gt(rot_mat_gt);
//       q_gt.normalize();
//       Eigen::Vector4d v_diff1, v_diff2;
//       v_diff1<<q_gt.coeffs()[0] - match.r_quat.coeffs()[0],
//                q_gt.coeffs()[1] - match.r_quat.coeffs()[1],
//                q_gt.coeffs()[2] - match.r_quat.coeffs()[2],
//                q_gt.coeffs()[3] - match.r_quat.coeffs()[3];
//       v_diff2<<q_gt.coeffs()[0] + match.r_quat.coeffs()[0],
//                q_gt.coeffs()[1] + match.r_quat.coeffs()[1],
//                q_gt.coeffs()[2] + match.r_quat.coeffs()[2],
//                q_gt.coeffs()[3] + match.r_quat.coeffs()[3];
//
//
//     Point tl(width,height);
//
//     obj_model.setModelView(match.r_quat, match.t_vec);
//     vector<Point2f> proj_pts;
//     obj_model.projectRasterPoints(proj_pts);
//     for( int j = 0; j < proj_pts.size(); j++ )
//     {
//       int x = cvRound(proj_pts[j].x), y = cvRound(proj_pts[j].y);
//       if( x < tl.x ) tl.x = x;
//       if( y < tl.y ) tl.y = y;
//     }
//
//       cv::Point2d p_t_diff(cvRound(tl.x + match.img_offset.x - gt_poses[i].bb.tl().x),
//                            cvRound(tl.y + match.img_offset.y - gt_poses[i].bb.tl().y));
//       double rot_diff = min<double>(v_diff1.norm(), v_diff2.norm()), t_diff  = cv_ext::norm2D(p_t_diff);
// //       cout<<rot_diff<<endl;
//       if( rot_diff < max_rot_diff && t_diff < max_t_diff )
//       {
// //         n_true_pos++;
// //         std::cout<<"Image "<<i<<" of "<<num_test_images<<" id_match : "<<i_m<<endl;
//
//         if( i_m >= 10 )
//           dcm_fbm[10]++;
//         else
//           dcm_fbm[i_m]++;
//
// //         for( int j = 0; j < proj_pts.size(); j++ )
// //         {
// //           proj_pts[j].x += match.img_offset.x;
// //           proj_pts[j].y += match.img_offset.y;
// //         }
// //         cv::Mat display = src_img.clone();
// //         cv_ext::drawPoints( display, proj_pts, Scalar(0,255,0) );
// //         cv_ext::showImage(display,"display");
//
//         found = true;
//         break;
//
//       }
// //       else
// //       {
// //         for( int j = 0; j < proj_pts.size(); j++ )
// //         {
// //           proj_pts[j].x += match.img_offset.x;
// //           proj_pts[j].y += match.img_offset.y;
// //         }
// //         cv::Mat display = src_img.clone();
// //         cv_ext::drawPoints( display, proj_pts, Scalar(0,0,255) );
// //         cv_ext::showImage(display,"display");
// //       }
//     }
// //     if(!found)
// //       std::cout<<"Image "<<i<<" not found"<<endl;
//   }
//
//



#if 0
  vector<Point2f> raster_pts, middle_points;
  vector<Vec4f> raster_segs;
  vector<float> raster_normals_dirs;

//   for(int i = 0; i < raster_pts.size(); i++)
//   {
//     Point2f p(round(raster_pts[i].x), round(raster_pts[i].y));
//     dbg_img.at<Vec3b>(p.y, p.x) = Vec3b( 255,255,255);
//   }

  bool segment_mode = false, show_normals = false,
       show_mask = false, show_depth = false, show_render = false;
  cout<<"Move with arrow and page up/down keys, rotate with a, s, w, z, q, e keys"<<endl
           <<"Space to change mode (points/segments)"<<endl;

  int view_idx = 0;
  bool exit_now = false;
  while( !exit_now )
  {
    Mat dbg_img;;

    obj_model.setModelView(view_idx);

    if( show_mask )
    {
      dbg_img = obj_model.getMask();
//       dbg_img.setTo(Vec3b( 0xFF, 0x80,0x80 ), mask);
    }
    else if( show_depth )
    {
      dbg_img = obj_model.getModelDepthMap();
    }
    else if( show_render )
    {
      dbg_img = obj_model.getRenderedModel();
    }
    else
    {
      dbg_img = Mat(Size(cam_model.imgWidth(),cam_model.imgHeight()),
                    DataType<Vec3b>::type, CV_RGB( 0,0,0));
      if( segment_mode )
      {
        if( show_normals )
        {
          obj_model.projectRasterSegments( raster_segs, raster_normals_dirs );

          middle_points.clear();
          middle_points.reserve(raster_segs.size());
          for(int i = 0; i < raster_segs.size(); i++)
          {
            Vec4f &seg = raster_segs[i];
            middle_points.push_back(Point2f((seg[0] + seg[2])/2, (seg[1] + seg[3])/2 ));
          }
          cv_ext::drawSegments( dbg_img, raster_segs );
          cv_ext::drawNormalsDirections(dbg_img, middle_points, raster_normals_dirs );
        }
        else
        {
          obj_model.projectRasterSegments( raster_segs );
          cv_ext::drawSegments( dbg_img, raster_segs );
        }
      }
      else
      {
        if( show_normals )
        {
          obj_model.projectRasterPoints( raster_pts, raster_normals_dirs);
          cv_ext::drawNormalsDirections(dbg_img, raster_pts, raster_normals_dirs );
        }
        else
        {
          obj_model.projectRasterPoints( raster_pts );
          cv_ext::drawPoints(dbg_img, raster_pts );
        }
      }
    }
    imshow("Test model", dbg_img);
    int key = cv_ext::waitKeyboard();

    parseMotionCommands( key, r_vec, t_vec );

    switch(key)
    {
      case '1':
        segment_mode = !segment_mode;
        break;
      case '2':
        show_normals = !show_normals;
        break;
      case '3':
        show_mask = !show_mask;
        if( show_mask )
          show_depth = show_render = false;
        break;
      case '4':
        show_depth = !show_depth;
        if( show_depth )
          show_mask = show_render = false;
        break;
      case '5':
        if( has_color )
        {
          show_render = !show_render;
          if( show_render )
            show_depth = show_mask = false;
        }

      case '0':
        view_idx++;
        view_idx %= obj_model.numPrecomputedModelsViews();

        break;
      case cv_ext::KEY_ESCAPE:
        exit_now = true;
        break;
    }
  }

#endif
  return 0;
}

