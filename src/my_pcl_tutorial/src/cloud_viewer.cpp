#include <stdlib.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include "std_msgs/String.h"
#include <pcl/registration/ia_ransac.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/common/geometry.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/pfhrgb.h>

#include <pcl/features/fpfh.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <my_pcl_tutorial/overlapping.h>
#include <pcl/features/rift.h>
#include <pcl/features/intensity_gradient.h>
#include <pcl/point_types_conversion.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/keypoints/sift_keypoint.h>
#include "pcl/kdtree/kdtree_flann.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/keypoints/susan.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <fstream>

// #include <chrono>

#define PI 3.14159265
  /* Reminder: how transformation matrices work :

           |-------> This column is the translation
    | 1 0 0 x |  \
    | 0 1 0 y |   }-> The identity 3x3 matrix (no rotation) on the left
    | 0 0 1 z |  /
    | 0 0 0 1 |    -> We do not use this line (and it has to stay 0,0,0,1)

  */


using namespace pcl;
using namespace std;
using namespace io;
using namespace Eigen;


typedef Histogram<32> RIFT32;

visualization::PCLVisualizer *p;

vector<float> get_percent_differences(vector<float> tar, vector<float> source) {
  vector<float> percent_differences;
  for (size_t i = 0; i < tar.size(); i ++) {
    percent_differences.push_back(100 * fabs(tar[i] - source[i])/ fabs(source[i]));
  }

  cout << "(";
  for (size_t i = 0; i < percent_differences.size(); i ++) {
    cout << percent_differences[i] << ", "; 
  }
  cout << ")" << endl;

  return percent_differences;
}

vector<float> get_tf_parameters(Matrix4f tf) {

  Matrix3f m;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j ++) {
      m(i, j) = tf(i, j);
    }
  }

  Eigen::Vector3f ea = m.eulerAngles(0, 1, 2) * 180/PI;
  cout << "Euler angles: " << ea(0) << " " << ea(1) << " " << ea(2) << endl; 
  cout << "Translations: " << tf(0, 3) << " " << tf(1, 3) << " " << tf(2, 3) << endl;

  vector<float> params;

  params.push_back(ea(0));
  params.push_back(ea(1));
  params.push_back(ea(2));
  params.push_back(tf(0, 3));
  params.push_back(tf(1, 3));
  params.push_back(tf(2, 3));
  return params;

}

vector<float> get_tf_errors(Matrix4f tf, Matrix4f tf_truth) {
  Matrix3f tf_rot;
  Matrix3f tf_truth_rot;

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j ++) {
      tf_rot(i, j) = tf(i, j);
      tf_truth_rot(i, j) = tf_truth(i, j);
    }
  }

  Matrix3f r_error = tf_truth_rot * tf_rot.transpose();


  float arg = (r_error.trace() - 1) / 2;
  if (arg > 1.0 && arg < 1.0005) {
    arg = 1.0;
  }
  float theta = acos(arg) * 180/PI;

  float norm_square = 0;
  for (size_t i = 0; i < 3; i++) {
    norm_square += pow(tf(i, 3) - tf_truth(i, 3), 2);
  }

  vector<float> tf_errors;
  tf_errors.push_back(theta);
  tf_errors.push_back(norm_square);

  return tf_errors;

}


Matrix4f get_inverse_transformation(Matrix4f tf) {
  Matrix3f rot;

  for (size_t i = 0; i < 3; i ++) {
    for (size_t j = 0; j < 3; j ++) {
      rot(i, j) = tf(i, j);
    }
  }

  Vector3f og_t;
  og_t(0) = tf(0, 3);
  og_t(1) = tf(1, 3);
  og_t(2) = tf(2, 3);

  Matrix3f rot_t = rot.transpose();
  Vector3f t = -1 * rot_t * og_t;



  Matrix4f inv_tf = Matrix4f::Identity();
  for (size_t i = 0; i < 3; i ++) {
    for (size_t j = 0; j < 3; j ++ ) {
      inv_tf(i, j) = rot_t(i, j);
    }
  }

  inv_tf(0, 3) = t(0);
  inv_tf(1, 3) = t(1);
  inv_tf(2, 3) = t(2);

  return inv_tf;

}

Matrix4f get_random_transformation() {
  float x_angle = -1*PI + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX/ (2*PI)));
  float y_angle = -1*PI + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX/ (2*PI)));
  float z_angle = -1*PI + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX/ (2*PI)));


  float x_translation = -3 + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX/ (6)));
  float y_translation = -3 + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX/ (6)));
  float z_translation = -3 + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX/ (6)));


  Matrix3f rot;
  rot = AngleAxisf(x_angle, Vector3f::UnitX())
    * AngleAxisf(y_angle,  Vector3f::UnitY())
    * AngleAxisf(z_angle, Vector3f::UnitZ());

  Eigen::Vector3f ea = rot.eulerAngles(0, 1, 2);

  Matrix4f tf = Matrix4f::Identity();

  for (size_t i = 0; i < 3; i ++) {
    for (size_t j = 0; j < 3; j ++ ) {
      tf(i, j) = rot(i, j);
    }
  }

  tf(0, 3) = x_translation;
  tf(1, 3) = y_translation;
  tf(2, 3) = z_translation;

  return tf;
}

Matrix4f get_rotation_matrix() {
  float x_angle = 0;-PI/10;
  float y_angle = -PI/1.2;
  float z_angle = -PI*1.05;


  float x_translation = 0;
  float y_translation = 0;
  float z_translation = 0;


  Matrix3f rot;
  rot = AngleAxisf(x_angle, Vector3f::UnitX())
    * AngleAxisf(y_angle,  Vector3f::UnitY())
    * AngleAxisf(z_angle, Vector3f::UnitZ());

  Eigen::Vector3f ea = rot.eulerAngles(0, 1, 2);

  Matrix4f tf = Matrix4f::Identity();

  for (size_t i = 0; i < 3; i ++) {
    for (size_t j = 0; j < 3; j ++ ) {
      tf(i, j) = rot(i, j);
    }
  }

  tf(0, 3) = x_translation;
  tf(1, 3) = y_translation;
  tf(2, 3) = z_translation;

  return tf;
}

Matrix4f get_transformation_matrix(float x_a, float y_a, float z_a, float x_t, float y_t, float z_t) {
  float x_angle = x_a;
  float y_angle = y_a;
  float z_angle = z_a;


  float x_translation = x_t;
  float y_translation = y_t;
  float z_translation = z_t;


  Matrix3f rot;
  rot = AngleAxisf(x_angle, Vector3f::UnitX())
    * AngleAxisf(y_angle,  Vector3f::UnitY())
    * AngleAxisf(z_angle, Vector3f::UnitZ());

  Eigen::Vector3f ea = rot.eulerAngles(0, 1, 2);

  Matrix4f tf = Matrix4f::Identity();

  for (size_t i = 0; i < 3; i ++) {
    for (size_t j = 0; j < 3; j ++ ) {
      tf(i, j) = rot(i, j);
    }
  }

  tf(0, 3) = x_translation;
  tf(1, 3) = y_translation;
  tf(2, 3) = z_translation;

  return tf;
}


double
find_feature_correspondences (PointCloud<PFHRGBSignature250>::Ptr &source_descriptors,
                              PointCloud<PFHRGBSignature250>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out, std::vector<float> &correspondence_scores_out)
{

  

  // Resize the output vector
  correspondences_out.resize (source_descriptors->size ());
  correspondence_scores_out.resize (source_descriptors->size ());

  // Use a KdTree to search for the nearest matches in feature space
  search::KdTree<PFHRGBSignature250> descriptor_kdtree;
  descriptor_kdtree.setInputCloud (target_descriptors);

  // Find the index of the best match for each keypoint, and store it in "correspondences_out"
  const int k = 1;
  std::vector<int> k_indices (k);
  std::vector<float> k_squared_distances (k);

  double total_correspondence_score = 0;
  for (size_t i = 0; i < source_descriptors->size(); ++i)
  {
    descriptor_kdtree.nearestKSearch (*source_descriptors, i, k, k_indices, k_squared_distances);
    correspondences_out[i] = k_indices[0];
    correspondence_scores_out[i] = k_squared_distances[0];
    total_correspondence_score += k_squared_distances[0];
  }

  return total_correspondence_score / source_descriptors->size();

}


double
find_fpfh_feature_correspondences (PointCloud<FPFHSignature33>::Ptr &source_descriptors,
                              PointCloud<FPFHSignature33>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out, std::vector<float> &correspondence_scores_out)
{

  // Resize the output vector
  correspondences_out.resize (source_descriptors->size ());
  correspondence_scores_out.resize (source_descriptors->size ());

  // Use a KdTree to search for the nearest matches in feature space
  search::KdTree<FPFHSignature33> descriptor_kdtree;
  descriptor_kdtree.setInputCloud (target_descriptors);

  // Find the index of the best match for each keypoint, and store it in "correspondences_out"
  const int k = 1;
  std::vector<int> k_indices (k);
  std::vector<float> k_squared_distances (k);

  double total_correspondence_score = 0;
  for (size_t i = 0; i < source_descriptors->size(); ++i)
  {
    descriptor_kdtree.nearestKSearch (*source_descriptors, i, k, k_indices, k_squared_distances);
    correspondences_out[i] = k_indices[0];
    correspondence_scores_out[i] = k_squared_distances[0];
    total_correspondence_score += k_squared_distances[0];
  }

  return total_correspondence_score / source_descriptors->size();

}

void
find_rift_correspondences (PointCloud<RIFT32>::Ptr &source_descriptors,
                              PointCloud<RIFT32>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out, std::vector<float> &correspondence_scores_out)
{

  cout << "finding feature correspondences" << endl;

  // Resize the output vector
  correspondences_out.resize (source_descriptors->size ());
  correspondence_scores_out.resize (source_descriptors->size ());

  // Use a KdTree to search for the nearest matches in feature space
  KdTreeFLANN<Histogram<32> > descriptor_kdtree;
    descriptor_kdtree.setInputCloud (target_descriptors);
    
  // Find the index of the best match for each keypoint, and store it in "correspondences_out"
  const int k = 1;
  std::vector<int> k_indices (k);
  std::vector<float> k_squared_distances (k);

  for (size_t i = 0; i < source_descriptors->size(); ++i)
  {
    descriptor_kdtree.nearestKSearch (*source_descriptors, i, k, k_indices, k_squared_distances);
    correspondences_out[i] = k_indices[0];
    correspondence_scores_out[i] = k_squared_distances[0];
  }
  
  cout << "Finding correspondences" << endl;
}

void visualize_keypoints (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points,
                          const pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints)
{
  // Add the points to the vizualizer
  pcl::visualization::PCLVisualizer viz;
  viz.addPointCloud (points, "points");
  // Draw each keypoint as a sphere
  for (size_t i = 0; i < keypoints->size (); ++i)
  {
    // Get the point data
    const pcl::PointXYZRGB & p = keypoints->points[i];
    // Pick the radius of the sphere *
    // float r = 2 * p.scale;
    float r = 2 * 0.01;

    // * Note: the scale is given as the standard deviation of a Gaussian blur, so a
    //   radius of 2*p.scale is a good illustration of the extent of the keypoint
    // Generate a unique string for each sphere
    std::stringstream ss ("keypoint");
    ss << i;
    // Add a sphere at the keypoint
    viz.addSphere (p, r, 1.0, 0.0, 0.0, ss.str ());
  }
  // Give control over to the visualizer
  viz.spin ();
}


void visualize_correspondences_scores (const PointCloud<PointXYZRGB>::Ptr points1,
                                const PointCloud<PointXYZRGB>::Ptr keypoints1,
                                const PointCloud<PointXYZRGB>::Ptr points2,
                                const PointCloud<PointXYZRGB>::Ptr keypoints2,
                                const std::vector<int> &correspondences,
                                const std::vector<float> &correspondence_scores)
{

  cout << "Starting visualizer " << endl;


  // We want to visualize two clouds side-by-side, so do to this, we'll make copies of the clouds and transform them
  // by shifting one to the left and the other to the right.  Then we'll draw lines between the corresponding points
  // Create some new point clouds to hold our transformed data
  PointCloud<PointXYZRGB>::Ptr points_left (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr keypoints_left (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr points_right (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr keypoints_right (new PointCloud<PointXYZRGB>);

  // Shift the first clouds' points to the left
  //const Eigen::Vector3f translate (0.0, 0.0, 0.3);
  const Eigen::Vector3f translate (5, 0.0, 0.0);
  const Eigen::Quaternionf no_rotation (0, 0, 0, 0);
  transformPointCloud (*points1, *points_left, -translate, no_rotation);
  transformPointCloud (*keypoints1, *keypoints_left, -translate, no_rotation);
  // Shift the second clouds' points to the right
  transformPointCloud (*points2, *points_right, translate, no_rotation);
  transformPointCloud (*keypoints2, *keypoints_right, translate, no_rotation);

  cout << "About to addiv isualizer " << endl;

  // Add the clouds to the vizualizer
  visualization::PCLVisualizer viz;
  viz.addPointCloud (points_left, "points_left");
  viz.addPointCloud (points_right, "points_right");
  // Compute the median correspondence score
  std::vector<float> temp (correspondence_scores);
  std::sort (temp.begin (), temp.end ());
  float median_score = temp[temp.size ()/2];
  // Draw lines between the best corresponding points
  for (size_t i = 0; i < keypoints_left->size(); ++i)

  {
    if (correspondence_scores[i] > median_score)
    {
      continue; // Don't draw weak correspondences
    }
    // Get the pair of points
    const PointXYZRGB & p_left = keypoints_left->points[i];
    const PointXYZRGB & p_right = keypoints_right->points[correspondences[i]];
    // Generate a random (bright) color
    double r = (rand() % 100);
    double g = (rand() % 100);
    double b = (rand() % 100);
    double max_channel = std::max (r, std::max (g, b));
    r /= max_channel;
    g /= max_channel;
    b /= max_channel;
    // Generate a unique string for each line
    std::stringstream ss ("line");
    ss << i;
    // Draw the line
    viz.addLine (p_left, p_right, r, g, b, ss.str ());
  }
  // Give control over to the visualizer
  viz.spin ();
}


void visualize_correspondences (const PointCloud<PointXYZRGB>::Ptr points1,
                                const PointCloud<PointXYZRGB>::Ptr keypoints1,
                                const PointCloud<PointXYZRGB>::Ptr points2,
                                const PointCloud<PointXYZRGB>::Ptr keypoints2,
                                const CorrespondencesPtr pCorrespondences)
{

  cout << "Starting visualizer " << endl;


  // We want to visualize two clouds side-by-side, so do to this, we'll make copies of the clouds and transform them
  // by shifting one to the left and the other to the right.  Then we'll draw lines between the corresponding points
  // Create some new point clouds to hold our transformed data
  PointCloud<PointXYZRGB>::Ptr points_left (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr keypoints_left (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr points_right (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr keypoints_right (new PointCloud<PointXYZRGB>);

  // Shift the first clouds' points to the left
  //const Eigen::Vector3f translate (0.0, 0.0, 0.3);
  const Eigen::Vector3f translate (5, 0.0, 0.0);
  const Eigen::Quaternionf no_rotation (0, 0, 0, 0);
  transformPointCloud (*points1, *points_left, -translate, no_rotation);
  transformPointCloud (*keypoints1, *keypoints_left, -translate, no_rotation);
  // Shift the second clouds' points to the right
  transformPointCloud (*points2, *points_right, translate, no_rotation);
  transformPointCloud (*keypoints2, *keypoints_right, translate, no_rotation);

  cout << "About to addiv isualizer " << endl;

  // Add the clouds to the vizualizer
  visualization::PCLVisualizer viz;
  viz.addPointCloud (points_left, "points_left");
  viz.addPointCloud (points_right, "points_right");

  // Draw lines between the best corresponding points
  for (size_t i = 0; i < pCorrespondences->size(); ++i)
  // for (size_t i = 0; i < 50; ++i)

  {

    // Get the pair of points
    const PointXYZRGB & p_left = keypoints_left->points[(*pCorrespondences)[i].index_query];
    const PointXYZRGB & p_right = keypoints_right->points[(*pCorrespondences)[i].index_match];
    // Generate a random (bright) color
    double r = (rand() % 100);
    double g = (rand() % 100);
    double b = (rand() % 100);
    double max_channel = std::max (r, std::max (g, b));
    r /= max_channel;
    g /= max_channel;
    b /= max_channel;
    // Generate a unique string for each line
    std::stringstream ss ("line");
    ss << i;
    // Draw the line
    viz.addLine (p_left, p_right, r, g, b, ss.str ());
  }
  // Give control over to the visualizer
  viz.spin ();
}


void downsample(PointCloud<PointXYZRGB>::Ptr &points,
                float leaf_size,
                PointCloud<PointXYZRGB>::Ptr &downsampled_out) 
{

  VoxelGrid<PointXYZRGB> vox_grid;
  vox_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
  vox_grid.setInputCloud(points);
  vox_grid.filter(*downsampled_out);

}

void compute_surface_normals
  (PointCloud<PointXYZRGB>::Ptr &points,
  float normal_radius,
  PointCloud<Normal>::Ptr &normals_out) {

  NormalEstimation<PointXYZRGB, Normal> norm_est;

  // Use a KdTree to perform neighbourhood searches
  search::KdTree<PointXYZRGB>::Ptr tree (new search::KdTree<PointXYZRGB>);
  norm_est.setSearchMethod(tree);

  // Specify the size of the local neighbourhood to use when computing the surface normals
  norm_est.setRadiusSearch(normal_radius);

  // Set the input points
  norm_est.setInputCloud(points);

  // Estimate the surface normals and store the result in "normals_out"
  norm_est.compute(*normals_out);

}

void compute_surface_intensity_normals (PointCloud<PointXYZI>::Ptr &points, float normal_radius, PointCloud<Normal>::Ptr &normals_out) {
  NormalEstimation<PointXYZI, Normal> norm_est;
  norm_est.setInputCloud(points);
  search::KdTree<PointXYZI>::Ptr tree (new search::KdTree<PointXYZI> (false));
  norm_est.setSearchMethod(tree);
  norm_est.setRadiusSearch(normal_radius);
  norm_est.compute(*normals_out);
}


void compute_intensity_gradient(PointCloud<PointXYZI>::Ptr &cloud, PointCloud<Normal>::Ptr &normals,
  float radius, PointCloud<IntensityGradient>::Ptr &cloud_ig) {

  IntensityGradientEstimation<PointXYZI, Normal, IntensityGradient> gradient_est;
  gradient_est.setInputCloud(cloud);
  gradient_est.setInputNormals(normals);

  search::KdTree<PointXYZI>::Ptr tree (new search::KdTree<PointXYZI> (false));
  gradient_est.setSearchMethod(tree);
  gradient_est.setRadiusSearch(radius);


  // PointCloud<IntensityGradient>::Ptr new_cloud_ig (new PointCloud<IntensityGradient>);
  gradient_est.compute(*cloud_ig);

}


// http://www.pointclouds.org/assets/rss2011/05_features.pdf
// https://github.com/otherlab/pcl/blob/master/examples/keypoints/example_sift_normal_keypoint_estimation.cpp
void detect_keypoints
  (PointCloud<PointXYZRGB>::Ptr &points, 
    float min_scale, 
    int nr_octaves, 
    int nr_scales_per_octave, 
    float min_contrast,
    PointCloud<PointWithScale>::Ptr &keypoints_out) {

  SIFTKeypoint<PointXYZRGB, PointWithScale> sift_detect;

  // Use a FLANN-based KdTree to perform neighbourhood searches
  sift_detect.setSearchMethod(search::KdTree<PointXYZRGB>::Ptr (new search::KdTree<PointXYZRGB>));

  // Set the detection parameters
  sift_detect.setScales(min_scale, nr_octaves, nr_scales_per_octave);
  sift_detect.setMinimumContrast(min_contrast);

  // Set the input
  sift_detect.setInputCloud(points);

  // Detect the keypoints and store them in "keypoints_out"
  sift_detect.compute(*keypoints_out);
}

void detect_susan_keypoints(PointCloud<PointXYZRGB>::Ptr &points,
  PointCloud<PointXYZRGB>::Ptr &keypoints_out,
  PointCloud<Normal>::Ptr &normals) {

  SUSANKeypoint<PointXYZRGB, PointXYZRGB> susan3D;
  susan3D.setInputCloud(points);
  susan3D.setNormals(normals);
  susan3D.setNonMaxSupression(true); // This is the thing that gives me 0 keypoints
  susan3D.setGeometricValidation(true);
  susan3D.setRadius(0.05); // was 0.05. Needs to be made higher if the points used are downsampled
  // susan3D.setIntensityThreshold(1.0);

  susan3D.compute(*keypoints_out);
  // There is a setNormals method to set normals if precalculated normals are available

}

void compute_PFH_features_at_keypoints
  (PointCloud<PointXYZRGB>::Ptr &points,
  PointCloud<Normal>::Ptr &normals,
  PointCloud<PointXYZRGB>::Ptr &keypoints,
  float feature_radius,
  PointCloud<PFHRGBSignature250>::Ptr &descriptors_out) {



  // Create a PFHEstimation object
  PFHRGBEstimation<PointXYZRGB, Normal, PFHRGBSignature250> pfh_est;

  // Set it to use a FLANN_based KdTree to perform its neighbourhood searches
  pfh_est.setSearchMethod(search::KdTree<PointXYZRGB>::Ptr (new search::KdTree<PointXYZRGB>));

  // Specify the radius of the PFH feature
  pfh_est.setRadiusSearch(feature_radius);


  // Use all of the points for analyzing the local structure of the cloud
  pfh_est.setSearchSurface(points);
  pfh_est.setInputNormals(normals);

  // But only compute features at the keypoints
  pfh_est.setInputCloud(keypoints);

  // Compute features
  pfh_est.compute(*descriptors_out);

}

void compute_FPFH_features_at_keypoints(PointCloud<PointXYZRGB>::Ptr &points, 
  PointCloud<Normal>::Ptr &normals,
  PointCloud<PointXYZRGB>::Ptr &keypoints, 
  float feature_radius, 
  PointCloud<FPFHSignature33>::Ptr &descriptors_out) {

  FPFHEstimation<PointXYZRGB, Normal, FPFHSignature33> fpfh_est;

  search::KdTree<PointXYZRGB>::Ptr tree (new search::KdTree<PointXYZRGB>);
  fpfh_est.setSearchMethod(tree);
  fpfh_est.setRadiusSearch(feature_radius);
  fpfh_est.setInputCloud(keypoints);
  fpfh_est.setSearchSurface(points);
  fpfh_est.setInputNormals(normals);
  fpfh_est.compute(*descriptors_out);
}

void compute_rift_features_at_keypoints(PointCloud<PointXYZI>::Ptr &cloud,
                                        PointCloud<IntensityGradient>::Ptr &cloud_ig,
                                        PointCloud<Histogram<32> >::Ptr &rift_output,
                                        PointCloud<PointXYZRGB>::Ptr &keypoints,
                                        float radius,
                                        float nr_distance_bins,
                                        float nr_gradient_bins) {

  RIFTEstimation<PointXYZI, IntensityGradient, Histogram<32> > rift_est;
  search::KdTree<PointXYZI>::Ptr tree (new search::KdTree<PointXYZI> (false));
  rift_est.setSearchMethod(tree);
  rift_est.setRadiusSearch(radius);
  rift_est.setNrDistanceBins(nr_distance_bins);
  rift_est.setNrGradientBins(nr_gradient_bins);

  // Use all of the points for analyzing the local structure of the cloud
  // rift_est.setSearchSurface(cloud);
  rift_est.setInputGradient(cloud_ig);

  PointCloud<PointXYZI>::Ptr keypoints_xyzi (new PointCloud<PointXYZI>);
  // copyPointCloud(*keypoints, *keypoints_xyzi);
  PointCloudXYZRGBtoXYZI(*keypoints, *keypoints_xyzi);

  // But only compute features at the keypoints
  rift_est.setInputCloud(keypoints_xyzi);
  rift_est.setSearchSurface(cloud);
  // rift_est.setInputCloud(cloud);
  rift_est.compute(*rift_output);
}

Eigen::Matrix4f getCorrespondenceRejectionTransformation(
  CorrespondencesPtr pCorrespondences,
  PointCloud<PointXYZRGB>::Ptr &keypoints1,
  PointCloud<PointXYZRGB>::Ptr &keypoints2,
  CorrespondencesPtr &pCorrespondences2,
  float inlier_threshold,
  int max_iterations) {

  // CorrespondencesPtr pCorrespondences2 (new Correspondences);
  registration::CorrespondenceRejectorSampleConsensus<PointXYZRGB> rej;
  rej.setInlierThreshold(inlier_threshold);
  // rej.setSaveInliers(true);
  // rej.setRefineModel(true);
  rej.setMaximumIterations(max_iterations);
  rej.setInputSource(keypoints1);
  rej.setInputTarget(keypoints2);
  rej.setInputCorrespondences(pCorrespondences);
  rej.getCorrespondences(*pCorrespondences2);
  std::cout << "Size of correspondences: " << pCorrespondences2->size()  << std::endl;


  return rej.getBestTransformation();
  // std::vector<int> *inlier_indices = new std::vector<int>();
  // std::vector<int> inlier_indices;
  // rej.getInliersIndices(inlier_indices);
  // std::cout << "Size of inlier indices: " << inlier_indices.size()  << std::endl;
}

float getFitnessScore(
  PointCloud<PointXYZRGB>::Ptr &cloud1,
  PointCloud<PointXYZRGB>::Ptr &cloud2 
  ) {
  IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
  icp.setMaxCorrespondenceDistance(0.05);
  icp.setMaximumIterations(1);
  icp.setInputSource(cloud1);
  icp.setInputTarget(cloud2);

  PointCloud<PointXYZRGB>::Ptr registration_output (new PointCloud<PointXYZRGB>);
  icp.align(*registration_output);
  Eigen::Matrix4f refined_T = icp.getFinalTransformation();

  return icp.getFitnessScore();
}


void get_overlapping_clouds(string filename, PointCloud<PointXYZRGB>::Ptr &source, PointCloud<PointXYZRGB>::Ptr &target) {

  cout << "===== Getting overlapping clouds =====" << endl;
  // Make cloud to contain entire point cloud
  PointCloud<PointXYZRGB>::Ptr pre_total_cloud (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr total_cloud (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr cloud1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr cloud2 (new PointCloud<PointXYZRGB>);
  io::loadPCDFile(filename, *pre_total_cloud);

  vector<int> indices1;
  removeNaNFromPointCloud(*pre_total_cloud, *total_cloud, indices1);

  // Segment this point cloud into the two other ones
  PointXYZRGB min_pt, max_pt;
  
  getMinMax3D(*total_cloud, min_pt, max_pt);

  float x_mean = (min_pt.x + max_pt.x) / 2;
  float y_mean = (min_pt.y + max_pt.y) / 2;
  float z_mean = (min_pt.z + max_pt.z) / 2;

  float x_range = max_pt.x - min_pt.x;
  float y_range = max_pt.y - min_pt.y;
  float z_range = max_pt.z - min_pt.z;

  // The bigger these constants, the bigger the overlap in that direction
  float kx = 0.1;// + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX/ (0.5)));
  float kz = 0.1;// + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX/ (0.5)));
  float ky = 0.1;// + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX/ (0.5)));

  // cout << "Kx, Ky, Kz " << kx << " " << ky << " " << kz << endl;

  kx = 0.05;
  ky = 0.05;
  kz = 0.05;

  int overlap_counter = 0;

  for (size_t i = 0 ; i < total_cloud->points.size(); i ++) {

    if (total_cloud->points[i].y > (y_mean - ky * y_range) 
      && total_cloud->points[i].x > (x_mean - kx * x_range) 
      && total_cloud->points[i].z > (z_mean - kz * z_range) ) {
      cloud2->points.push_back(total_cloud->points[i]);
    }

    if (total_cloud->points[i].y < (y_mean + ky * y_range) 
      || total_cloud->points[i].x < (x_mean + kx * x_range)
      || total_cloud->points[i].z < (z_mean + kz * z_range)) {

      cloud1->points.push_back(total_cloud->points[i]);
    }

  }

  source = cloud2;
  target = cloud1;

  return;

}

void show_iteration(const PointCloud<PointXYZRGB>::Ptr cloud_target, const PointCloud<PointXYZRGB>::Ptr cloud_source) {
  // p->removePointCloud("source");
  // p->removePointCloud("target");
  p->addPointCloud (cloud_target, "target");
  p->addPointCloud (cloud_source, "source");

  p->spin();
}


PointCloud<PointXYZRGB>::Ptr pointcloud_registration(string pcd_path, PointCloud<PointXYZRGB>::Ptr og_pre_cloud1, PointCloud<PointXYZRGB>::Ptr og_cloud2, 
  string log_path, float feature_radius, float inlier_threshold, int ransac_iterations, int icp_iterations) {


  // Load files (first cloud)
  PointCloud<PointXYZRGB>::Ptr pre_cloud1 (new PointCloud<PointXYZRGB>);
  *pre_cloud1 = *og_pre_cloud1;
  PointCloud<PointXYZRGB>::Ptr cloud1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr downsampled1 (new PointCloud<PointXYZRGB>);

  PointCloud<PointXYZRGB>::Ptr downsampled2 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr cloud2 (new PointCloud<PointXYZRGB>);
  *cloud2 = *og_cloud2;
  PointCloud<PointXYZRGB>::Ptr tf_cloud1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr tf_downsampled1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr tf_cloud_icp1 (new PointCloud<PointXYZRGB>); 


  // Make final cloud
  PointCloud<PointXYZRGB>::Ptr cloud3 (new PointCloud<PointXYZRGB>);



  cout << "Source size: " << pre_cloud1->points.size() << ", target size: " << cloud2->points.size() << endl;


  // Transform pre_cloud1 to test
  Eigen::Matrix4f random_tf = get_random_transformation();
  Matrix4f inverse_random_tf = get_inverse_transformation(random_tf);
  transformPointCloud(*pre_cloud1, *cloud1, random_tf);

  // Initialize storage for other features
  PointCloud<Normal>::Ptr normals1(new PointCloud<Normal>);
  PointCloud<Normal>::Ptr normals2(new PointCloud<Normal>);
  PointCloud<Normal>::Ptr entire_cloud_normals1(new PointCloud<Normal>);
  PointCloud<Normal>::Ptr entire_cloud_normals2(new PointCloud<Normal>);

  PointCloud<PointXYZRGB>::Ptr keypoints1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr keypoints2 (new PointCloud<PointXYZRGB>);

  PointCloud<PointXYZRGB>::Ptr corr_keypoints1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr corr_keypoints2 (new PointCloud<PointXYZRGB>);

  PointCloud<PFHRGBSignature250>::Ptr descriptors1 (new PointCloud<PFHRGBSignature250>);
  PointCloud<PFHRGBSignature250>::Ptr descriptors2 (new PointCloud<PFHRGBSignature250>);
  PointCloud<FPFHSignature33>::Ptr fpfh_descriptors1 (new PointCloud<FPFHSignature33>);
  PointCloud<FPFHSignature33>::Ptr fpfh_descriptors2 (new PointCloud<FPFHSignature33>);

  
  // Downsample the cloud
  float voxel_grid_leaf_size = 0.01;
  voxel_grid_leaf_size = 0.05; // Use this to make bigger downsampling
  voxel_grid_leaf_size = 0.025;
  downsample(cloud1, voxel_grid_leaf_size, downsampled1);
  downsample(cloud2, voxel_grid_leaf_size, downsampled2);
  // downsampled1 = cloud1;
  // downsampled2 = cloud2;
  cout << "Cloud1 used to be " << cloud1->points.size() << " but after downsampling it's " << downsampled1->points.size() << endl;
  cout << "Cloud2 used to be " << cloud2->points.size() << " but after downsampling it's " << downsampled2->points.size() << endl;

  // Compute normals
  cout << endl << "===== Calculating normals =====" << endl;
  const float normal_radius = 0.05;
  compute_surface_normals(downsampled1, normal_radius, normals1);
  compute_surface_normals(downsampled2, normal_radius, normals2);
  // compute_surface_normals(cloud1, normal_radius, entire_cloud_normals1);
  // compute_surface_normals(cloud2, normal_radius, entire_cloud_normals2);


  // Get SUSAN keypoints
  cout << endl << "===== Calculating SUSAN keypoints =====" << endl;
  clock_t susan_start_time = clock();
  // detect_susan_keypoints(cloud1, keypoints1, entire_cloud_normals1);
  // detect_susan_keypoints(cloud2, keypoints2, entire_cloud_normals2);
  detect_susan_keypoints(downsampled1, keypoints1, normals1);
  detect_susan_keypoints(downsampled2, keypoints2, normals2);
  double susan_duration = (clock() - susan_start_time) / (double) CLOCKS_PER_SEC;
  cout << "Size of keypoints1: " << keypoints1->points.size() << endl;
  cout << "Size of keypoints2: " << keypoints2->points.size() << endl;


  // Compute PFH features
  // const float feature_radius = 0.1; // Has to be larger than normal radius estimation. 0.15 is default good
  cout << endl << "===== Finding feature descriptors =====" << endl;
  clock_t pfh_start_time = clock();
  // compute_PFH_features_at_keypoints(downsampled1, normals1, keypoints1, feature_radius, descriptors1);
  // compute_PFH_features_at_keypoints(downsampled2, normals2, keypoints2, feature_radius, descriptors2);

  compute_FPFH_features_at_keypoints(downsampled1, normals1, keypoints1, feature_radius, fpfh_descriptors1);
  compute_FPFH_features_at_keypoints(downsampled2, normals2, keypoints2, feature_radius, fpfh_descriptors2);

  double pfh_duration = (clock() - pfh_start_time) / (double) CLOCKS_PER_SEC;
  cout << "Duration: " << pfh_duration << " s" << endl;


  // Find feature correspondences
  cout << endl << "===== Finding feature correspondences =====" << endl;
  vector<int> correspondences;
  vector<float> correspondence_scores;
  // double mean_correspondence_score = find_feature_correspondences(descriptors1, descriptors2, correspondences, correspondence_scores);
  double mean_correspondence_score = find_fpfh_feature_correspondences(fpfh_descriptors1, fpfh_descriptors2, correspondences, correspondence_scores);


  // Do percentiles for now
  mean_correspondence_score = mean_correspondence_score * 1.5;
  mean_correspondence_score = numeric_limits<int>::max();

  // Only add correspondences that are less than the mean_correspondence_score
  int good_correspondence_counter = 0;
  for (size_t i = 0; i < correspondence_scores.size(); i++) {
    if (correspondence_scores[i] < mean_correspondence_score) {
      good_correspondence_counter++;
    }
  }

  cout << "Before, there were " << correspondences.size() << " correspondences" << endl;
  cout << "Now, there are " << good_correspondence_counter << " of them " << endl;

  CorrespondencesPtr pCorrespondences (new Correspondences);
  pCorrespondences->resize(good_correspondence_counter);
  corr_keypoints2 = keypoints2;

  // Replace keypoints in keypoints1 with only those that had good correspondences
  for (size_t i = 0; i < correspondence_scores.size(); i++) {
    if (correspondence_scores[i] < mean_correspondence_score) {
      corr_keypoints1->points.push_back(keypoints1->points[i]);
    }
  }

  int p_iter = 0;
  for (size_t i = 0; i < correspondences.size(); i++) {
    if (correspondence_scores[i] < mean_correspondence_score) {
      (*pCorrespondences)[p_iter].index_query = p_iter;
      (*pCorrespondences)[p_iter].index_match = correspondences[i];
      p_iter ++;
    }
  }

  Eigen::Matrix4f initial_T = Eigen::Matrix4f::Identity();

  cout << endl << "===== RANSAC =====" << endl;  CorrespondencesPtr pCorrespondences2 (new Correspondences);

  clock_t ransac_start_time = clock();
  initial_T = getCorrespondenceRejectionTransformation(pCorrespondences, corr_keypoints1, corr_keypoints2, pCorrespondences2, inlier_threshold, ransac_iterations);
  double ransac_duration = (clock() - ransac_start_time) / (double) CLOCKS_PER_SEC;
  // visualize_correspondences(cloud1, keypoints1, cloud2, keypoints2, pCorrespondences2);


  /////////////////////////////////////////////////
  // pcl::visualization::PCLVisualizer ransac_viz;
  // ransac_viz.addPointCloud(downsampled2, "target");
  // ransac_viz.addPointCloud (downsampled1, "points");

  // PointCloud<PointXYZRGB>::Ptr ransac_progress (new PointCloud<PointXYZRGB>);
  // PointCloud<PointXYZRGB>::Ptr ransac_result (new PointCloud<PointXYZRGB>);
  // *ransac_result = *downsampled1;

  // Matrix4f ransac_T = Matrix4f::Identity();

  // cout << "Trying to visualize" << endl;
  // for (int i = 0; i < 100000; i += 100) {

  //   ransac_viz.removePointCloud("points");
  //   transformPointCloud(*downsampled1, *ransac_progress, ransac_T);
  //   ransac_viz.addPointCloud (ransac_progress, "points");

  //   ransac_T = getCorrespondenceRejectionTransformation(pCorrespondences, corr_keypoints1, corr_keypoints2, inlier_threshold, i);
  //   ransac_viz.spinOnce();

  //   // Ti = icp.getFinalTransformation() * Ti;
  //   // show_iteration(cloud2, icp_progress);
  // }
  // cout << "Finished visualizing" << endl;

  //////////////////////////////////////////




  // Apply transformation to the first cloud
  transformPointCloud(*cloud1, *tf_cloud1, initial_T);
  transformPointCloud(*downsampled1, *tf_downsampled1, initial_T);

  // Visualize the keypoints
  // visualize_keypoints(cloud1, keypoints1);
  // visualize_keypoints(cloud2, keypoints2);

  // Visualize the two point clouds and their feature correspondences
  // visualize_correspondences_scores(cloud1, keypoints1, cloud2, keypoints2, correspondences, correspondence_scores);
  // visualize_correspondences(cloud1, keypoints1, cloud2, keypoints2, pCorrespondences);


  *cloud3 = *cloud1 + *cloud2;
  *cloud3 = *downsampled2 + *downsampled1;

  // Do ICP for refinement
  cout << endl << "===== ICP =====" << endl;  clock_t icp_start_time = clock();
  IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
  icp.setMaxCorrespondenceDistance(0.05);

  // icp.setInputSource(tf_cloud1);
  // icp.setInputTarget(cloud2);
  

  icp.setInputSource(tf_downsampled1);
  icp.setInputTarget(downsampled2);

  icp.setTransformationEpsilon(1e-8);
  icp.setMaximumIterations(icp_iterations);
  icp.align(*tf_cloud_icp1);
  Eigen::Matrix4f refined_T = icp.getFinalTransformation();
  double icp_duration = (clock() - icp_start_time) / (double) CLOCKS_PER_SEC;

  // Transform the downsampled instead!
  // transformPointCloud(*cloud1, *tf_cloud_icp1, refined_T * initial_T);
  // *cloud3 = *tf_cloud_icp1 + *cloud2;
  transformPointCloud(*downsampled1, *tf_cloud_icp1, refined_T * initial_T);
  *cloud3 = *tf_cloud_icp1 + *downsampled2;

  /*
  IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
  icp.setMaxCorrespondenceDistance(0.05);

  icp.setMaximumIterations(2);

  icp.setInputSource(tf_downsampled1);
  icp.setInputTarget(downsampled2);


  PointCloud<PointXYZRGB>::Ptr icp_progress (new PointCloud<PointXYZRGB>);
  *icp_progress = *tf_downsampled1;
  PointCloud<PointXYZRGB>::Ptr icp_result (new PointCloud<PointXYZRGB>);
  *icp_result = *tf_downsampled1;

  Matrix4f Ti = Matrix4f::Identity();
  */


  // pcl::visualization::PCLVisualizer icp_viz;
  // icp_viz.setSize(1500, 1000);
  // icp_viz.addPointCloud(downsampled2, "target");
  // icp_viz.addPointCloud (icp_progress, "points");

  // cout << "Trying to visualize" << endl;
  // for (int i = 0; i < 100; i ++) {
  //   icp_viz.removePointCloud("points");
  //   icp_progress = icp_result;
  //   icp_viz.addPointCloud (icp_progress, "points");

  //   icp.setInputSource(icp_progress);
  //   icp.align(*icp_result);
  //   icp_viz.spinOnce();

  //   // Ti = icp.getFinalTransformation() * Ti;
  //   // show_iteration(cloud2, icp_progress);
  // }

  // icp_viz.addPointCloud(cloud3, "two_clouds");
  // icp_viz.spin();


  cout << "Finished visualizing" << endl;


  cout << endl << "===== Logging =====" << endl;  // Open log file
  ofstream log_file;
  log_file.open(log_path.c_str(), ios::out | ios::app);
  log_file << pcd_path << "," << feature_radius << "," << inlier_threshold << ",";

  vector<float> errors;
  cout << "\n---------Parameters of inverse random tf---------" << endl;
  vector<float> desired_params = get_tf_parameters(inverse_random_tf);

  cout << "\n---------Parameters of initial T---------" << endl;
  get_tf_parameters(initial_T);
  errors = get_tf_errors(initial_T, inverse_random_tf);
  log_file << errors[0] << "," << errors[1];

  cout << "\n---------Parameters of final T---------" << endl;
  get_tf_parameters(refined_T * initial_T);
  errors = get_tf_errors(refined_T * initial_T, inverse_random_tf);
  log_file << "," << errors[0] << "," << errors[1] << "," << icp.getFitnessScore(0.01);

  log_file << "," << susan_duration << "," << pfh_duration << "," << ransac_duration << "," << icp_duration << endl;


  log_file.close();
  
  


  return cloud3;
  
  
}

void log_header(string log_path) {
  ofstream log_file;
  log_file.open(log_path.c_str(), ios::out | ios::app);
  
  log_file << "\n";


  log_file.close();
}

int main(int argc, char** argv) {


  PointCloud<PointXYZRGB>::Ptr final_cloud (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr pre_cloud1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr cloud2 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr cloud3 (new PointCloud<PointXYZRGB>);

  vector<string> test_pcd_files;
  // test_pcd_files.push_back("/home/drrobot1/rgbdslam_catkin_ws/easymerge2.pcd");
  // test_pcd_files.push_back("/home/drrobot1/rgbdslam_catkin_ws/easymerge1.pcd");
  // test_pcd_files.push_back("/home/drrobot1/rgbdslam_catkin_ws/hi1.pcd");
  // test_pcd_files.push_back("/home/drrobot1/rgbdslam_catkin_ws/hi2.pcd");
  // test_pcd_files.push_back("/home/drrobot1/rgbdslam_catkin_ws/hey2.pcd");
  // test_pcd_files.push_back("/home/drrobot1/rgbdslam_catkin_ws/hey1.pcd");
  test_pcd_files.push_back("/home/drrobot1/rgbdslam_catkin_ws/rgbdslam_og1.pcd");
  test_pcd_files.push_back("/home/drrobot1/rgbdslam_catkin_ws/rgbdslam_og2.pcd");
  test_pcd_files.push_back("/home/drrobot1/rgbdslam_catkin_ws/rgbdslam_og3.pcd");
  test_pcd_files.push_back("/home/drrobot1/rgbdslam_catkin_ws/rgbdslam_og4.pcd");


  // ------------------ LOADING TWO POINT CLOUD FILES 
  /*

  io::loadPCDFile("/home/drrobot1/rgbdslam_catkin_ws/rgbdslam_og1.pcd", *pre_cloud1);
  io::loadPCDFile("/home/drrobot1/rgbdslam_catkin_ws/rgbdslam_og4.pcd", *cloud2);

  vector<int> indices1;
  removeNaNFromPointCloud(*pre_cloud1, *pre_cloud1, indices1);
  vector<int> indices2;
  removeNaNFromPointCloud(*cloud2, *cloud2, indices2);

  // 
// PointCloud<PointXYZRGB>::Ptr pointcloud_registration(string pcd_path, PointCloud<PointXYZRGB>::Ptr og_pre_cloud1, PointCloud<PointXYZRGB>::Ptr og_cloud2, 
//   string log_path, float feature_radius, float inlier_threshold, int ransac_iterations, int icp_iterations) {

  get_overlapping_clouds("/home/drrobot1/rgbdslam_catkin_ws/rgbdslam_og3.pcd", pre_cloud1, cloud2);

  final_cloud = pointcloud_registration("not generated", pre_cloud1, cloud2, 
    "/home/drrobot1/rgbdslam_catkin_ws/src/my_pcl_tutorial/src/logs/test.txt" , 0.12, 0.16, 1000000, 0);
  */
  // ------------------ 



  //Go through inlier thresholds
  for (float i = 0.02; i < 0.12; i += 0.02) {
    for (int k = 0; k < test_pcd_files.size(); k++) {
      string test_filename = test_pcd_files[k];
      for (int j = 0; j < 3; j ++) {
        string filename = "1000ransac_10icp_0.15feature_inlierthreshold.txt";
        get_overlapping_clouds(test_filename, pre_cloud1, cloud2);
        cout << "!!! " << pre_cloud1->points.size() << " " << cloud2->points.size() << endl;
        final_cloud = pointcloud_registration(test_filename, pre_cloud1, cloud2, 
          "/home/drrobot1/rgbdslam_catkin_ws/src/my_pcl_tutorial/src/rgbdslam_logs/" + filename, 0.15, i, 1000, 10);
      }
    }
  }



  // vector<int> indices3;
  // removeNaNFromPointCloud(*cloud3, *cloud3, indices3);
  // transformPointCloud(*cloud2, *cloud2, get_transformation_matrix(0, -PI/3, 0, -5, -1, 0));
  // transformPointCloud(*cloud3, *cloud3, get_transformation_matrix(0, PI/10, 0, 5, 1, 0));

  // pcl::visualization::PCLVisualizer icp_viz;
  // icp_viz.setSize(1500, 1000);

  // icp_viz.addPointCloud(pre_cloud1, "pre_cloud1");
  // icp_viz.addPointCloud(cloud2, "cloud2");
  // icp_viz.addPointCloud(cloud3, "cloud3");
  // icp_viz.spin();


  // string filename = "test.txt";
  // final_cloud = pointcloud_registration(pre_cloud1, cloud2, 
  //   "/home/drrobot1/rgbdslam_catkin_ws/src/my_pcl_tutorial/src/logs/" + filename, 0.06, 10000, 500);

  // final_cloud = pointcloud_registration(final_cloud, cloud3, 
  // "/home/drrobot1/rgbdslam_catkin_ws/src/my_pcl_tutorial/src/logs/" + filename, 0.06, 10000, 500);

      
  // // Publish to RVIZ
  ros::init(argc, argv, "pcl_create");
  ROS_INFO("Started PCL publishing node");


  ros::NodeHandle nh;
  ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2> ("pcl_output", 1);

  // Make point cloud into ROS message
  sensor_msgs::PointCloud2 output;
  toROSMsg(*final_cloud, output);
  // toROSMsg(*cloud2, output);

  output.header.frame_id = "pcl_frame";

  // Loop and publish
  ros::Rate loop_rate(1);
  while(ros::ok()) {
    pcl_pub.publish(output);
    ros::spinOnce();
    loop_rate.sleep();
  }

  
  return 0;
  
}
