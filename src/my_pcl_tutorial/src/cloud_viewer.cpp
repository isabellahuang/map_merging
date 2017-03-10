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


typedef Histogram<32> RIFT32;

void
find_feature_correspondences (PointCloud<PFHRGBSignature250>::Ptr &source_descriptors,
                              PointCloud<PFHRGBSignature250>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out, std::vector<float> &correspondence_scores_out)
{

  cout << "finding feature correspondences" << endl;

  // Resize the output vector
  correspondences_out.resize (source_descriptors->size ());
  correspondence_scores_out.resize (source_descriptors->size ());
  cout << "Should be size " << source_descriptors->size() << endl;

  // Use a KdTree to search for the nearest matches in feature space
  search::KdTree<PFHRGBSignature250> descriptor_kdtree;
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
  PointCloud<PointXYZRGB>::Ptr &keypoints_out) {
  SUSANKeypoint<PointXYZRGB, PointXYZRGB> susan3D;
  susan3D.setInputCloud(points);
  susan3D.setNonMaxSupression(true);
  susan3D.setGeometricValidation(true);
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

  /* This is a little bit messy: since our keypoint detection returns PointWithScale points, but we want to
   * use them as an input to our PFH estimation, which expects clouds of PointXYZRGB points.  To get around this,
   * we'll use copyPointCloud to convert "keypoints" (a cloud of type PointCloud<PointWithScale>) to
   * "keypoints_xyzrgb" (a cloud of type PointCloud<PointXYZRGB>).  Note that the original cloud doesn't have any RGB
   * values, so when we copy from PointWithScale to PointXYZRGB, the new r,g,b fields will all be zero.
   */


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
  float inlier_threshold) {

  CorrespondencesPtr pCorrespondences2 (new Correspondences);
  registration::CorrespondenceRejectorSampleConsensus<PointXYZRGB> rej;
  rej.setInlierThreshold(inlier_threshold);
  rej.setSaveInliers(true);
  rej.setRefineModel(true);
  rej.setInputSource(keypoints1);
  rej.setInputTarget(keypoints2);
  rej.setInputCorrespondences(pCorrespondences);
  rej.getCorrespondences(*pCorrespondences2);

  return rej.getBestTransformation();
  // std::vector<int> *inlier_indices = new std::vector<int>();
  // std::vector<int> inlier_indices;
  // rej.getInliersIndices(inlier_indices);
  // std::cout << "Size of inlier indices: " << inlier_indices.size()  << std::endl;
  // std::cout << "Size of correspondences: " << pCorrespondences->size()  << std::endl;
}

float getFitnessScore(
  PointCloud<PointXYZRGB>::Ptr &cloud1,
  PointCloud<PointXYZRGB>::Ptr &cloud2 
  ) {
  IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
  icp.setMaxCorrespondenceDistance(0.01);
  icp.setMaximumIterations(1);
  icp.setInputSource(cloud1);
  icp.setInputTarget(cloud2);

  PointCloud<PointXYZRGB>::Ptr registration_output (new PointCloud<PointXYZRGB>);
  icp.align(*registration_output);
  Eigen::Matrix4f refined_T = icp.getFinalTransformation();

  return icp.getFitnessScore();
}





int main(int argc, char** argv) {

  // Load files (first cloud)
  PointCloud<PointXYZRGB>::Ptr pre_cloud1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr cloud1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr downsampled1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZI>::Ptr cloudIntensity1(new PointCloud<PointXYZI>);

  io::loadPCDFile("/home/drrobot1/rgbdslam_catkin_ws/hey2.pcd", *pre_cloud1);

  // Load files (second cloud)
  PointCloud<PointXYZRGB>::Ptr proto_cloud2 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr pre_cloud2 (new PointCloud<PointXYZRGB>);

  PointCloud<PointXYZRGB>::Ptr downsampled2 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr cloud2 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr tf_cloud1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr tf_keypoints1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZI>::Ptr cloudIntensity2(new PointCloud<PointXYZI>);

  io::loadPCDFile("/home/drrobot1/rgbdslam_catkin_ws/hey1.pcd", *proto_cloud2);


  // Transform pre_cloud2 to test
  Eigen::Matrix4f transform_second = Eigen::Matrix4f::Identity();
  transformPointCloud(*proto_cloud2, *pre_cloud2, transform_second);

  // Make final cloud
  PointCloud<PointXYZRGB>::Ptr cloud3 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZI>::Ptr cloudIntensity3(new PointCloud<PointXYZI>);

  // Remove NAN from PC
  vector<int> indices1;
  removeNaNFromPointCloud(*pre_cloud1, *cloud1, indices1);
  vector<int> indices2;
  removeNaNFromPointCloud(*pre_cloud2, *cloud2, indices2);

  // Initialize storage for other features
  PointCloud<Normal>::Ptr normals1(new PointCloud<Normal>);
  PointCloud<Normal>::Ptr normals2(new PointCloud<Normal>);
  PointCloud<Normal>::Ptr normals_i1(new PointCloud<Normal>);
  PointCloud<Normal>::Ptr normals_i2(new PointCloud<Normal>);

  // PointCloud<PointWithScale>::Ptr keypoints1 (new PointCloud<PointWithScale>);
  // PointCloud<PointWithScale>::Ptr keypoints2 (new PointCloud<PointWithScale>);
  // PointCloud<PointWithScale>::Ptr keypoints3 (new PointCloud<PointWithScale>);
  PointCloud<PointXYZRGB>::Ptr pre_keypoints1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr pre_keypoints2 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr keypoints3 (new PointCloud<PointXYZRGB>);

  PointCloud<PointXYZRGB>::Ptr keypoints1 (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr keypoints2 (new PointCloud<PointXYZRGB>);

  PointCloud<PFHRGBSignature250>::Ptr descriptors1 (new PointCloud<PFHRGBSignature250>);
  PointCloud<PFHRGBSignature250>::Ptr descriptors2 (new PointCloud<PFHRGBSignature250>);
  // PointCloud<FPFHSignature33>::Ptr descriptors1 (new PointCloud<FPFHSignature33>);
  // PointCloud<FPFHSignature33>::Ptr descriptors2 (new PointCloud<FPFHSignature33>);

  // Downsample the cloud
  float voxel_grid_leaf_size = 0.01;
  voxel_grid_leaf_size = 0.05; // Use this to make bigger downsampling
  downsample(cloud1, voxel_grid_leaf_size, downsampled1);
  downsample(cloud2, voxel_grid_leaf_size, downsampled2);
  // downsampled1 = cloud1;
  // downsampled2 = cloud2;


  // Compute surface intensity normals
  const float normal_radius = 0.05;
  compute_surface_normals(downsampled1, normal_radius, normals1);
  compute_surface_normals(downsampled2, normal_radius, normals2);


  // Compute keypoints
  // const float min_scale = 0.05;
  // const int nr_octaves = 2;
  // const int nr_octaves_per_scale = 3;
  // const float min_contrast = 10;
  // detect_keypoints(cloud1, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints1);
  // detect_keypoints(cloud2, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints2);
  detect_susan_keypoints(cloud1, keypoints1);
  detect_susan_keypoints(cloud2, keypoints2);

  // vector<int> indices3;
  // removeNaNFromPointCloud(*pre_keypoints1, *keypoints1, indices3);

  // vector<int> indices4;
  // removeNaNFromPointCloud(*pre_keypoints2, *keypoints2, indices4);

  // Remove use of keypoints
  // keypoints1 = downsampled1;
  // keypoints2 = downsampled2;


  // Compute PFH features
  const float feature_radius = 0.15; // Has to be larger than normal radius estimation
  compute_PFH_features_at_keypoints(downsampled1, normals1, keypoints1, feature_radius, descriptors1);
  compute_PFH_features_at_keypoints(downsampled2, normals2, keypoints2, feature_radius, descriptors2);

  cout << "Size of keypoints1: " << keypoints1->points.size() << endl;
  cout << "Size of descriptors1: " << descriptors1->points.size() << endl;
  // compute_FPFH_features_at_keypoints(downsampled1, normals1, keypoints1, feature_radius, descriptors1);
  // compute_FPFH_features_at_keypoints(downsampled2, normals2, keypoints2, feature_radius, descriptors2);

/*

  // Get intensity gradient
  float intensity_radius = 0.25;
  PointCloud<PointXYZI>::Ptr cloud_i1 (new PointCloud<PointXYZI>);
  PointCloud<PointXYZI>::Ptr cloud_i2 (new PointCloud<PointXYZI>);

  PointCloud<IntensityGradient>::Ptr cloud_ig1 (new PointCloud<IntensityGradient>);
  PointCloud<IntensityGradient>::Ptr cloud_ig2 (new PointCloud<IntensityGradient>);

  PointCloudXYZRGBtoXYZI(*cloud1, *cloud_i1);
  PointCloudXYZRGBtoXYZI(*downsampled2, *cloud_i2);

 // Compute surface normals
  const float intensity_normal_radius = 0.25;
  compute_surface_intensity_normals(cloud_i1, intensity_normal_radius, normals_i1);
  compute_surface_intensity_normals(cloud_i2, intensity_normal_radius, normals_i2);


  cout << "Computing intensity gradient" << endl;
  compute_intensity_gradient(cloud_i1, normals_i1, intensity_radius, cloud_ig1); 
  compute_intensity_gradient(cloud_i2, normals_i2, intensity_radius, cloud_ig2);

  cout << "Printing cloud intensity gradient" << endl;
  cout << cloud_ig1->points[1000] << endl;
  cout << normals1->points[1000] << endl;
  cout << "Intensity gradient estimated with size " << cloud_ig1->points.size() << endl;
  cout << "Intensity estimated with size " << cloud_i1->points.size() << endl;
  // cout << "Cloud originally has size " << downsampled1->points.size() << endl;

  // Get RIFT descriptors
  float radius_rift = 0.5;
  // float nr_distance_bins = 4.0;
  // float nr_gradient_bins = 8.0;
  float nr_distance_bins = 4;
  float nr_gradient_bins = 8;

  PointCloud<Histogram<32> >::Ptr rift_output1 (new PointCloud<Histogram<32> >);
  PointCloud<Histogram<32> >::Ptr rift_output2 (new PointCloud<Histogram<32> >);

  cout << "Computing rift features at keypoints " << endl;
  compute_rift_features_at_keypoints(cloud_i1, cloud_ig1, rift_output1, keypoints1, radius_rift, nr_distance_bins, nr_gradient_bins);
  compute_rift_features_at_keypoints(cloud_i2, cloud_ig2, rift_output2, keypoints2, radius_rift, nr_distance_bins, nr_gradient_bins);

  cout << "Printing point " << cloud_i1->points.size() <<  endl;
  cout << "Printing point " << cloud_ig1->points.size() <<  endl;
*/
  // for (size_t i = 0; i < rift_output1->points.size(); i++) {
  //   cout << "RIFT_OUTPUT1 " << rift_output1->points[i] << endl;
  // }

  /*
  // DO THE ENTIRE RIFT ESTIMATION HERE for cloud_i1
  // Estimate the surface normals
  PointCloud<Normal>::Ptr cloud_n (new PointCloud<Normal>);
  NormalEstimation<PointXYZI, Normal> norm_est;
  norm_est.setInputCloud(cloud_i1);
  search::KdTree<PointXYZI>::Ptr treept1 (new search::KdTree<PointXYZI> (false));
  norm_est.setSearchMethod(treept1);
  norm_est.setRadiusSearch(0.25);
  norm_est.compute(*cloud_n);

  cout << "Surface normals estimated with size : " << cloud_n->points.size() << endl;
  cout << cloud_n->points[10] << endl;

  // Estimate the intensity gradient
  PointCloud<IntensityGradient>::Ptr cloud_ig (new PointCloud<IntensityGradient>);
  IntensityGradientEstimation<PointXYZI, Normal, IntensityGradient> gradient_est;
  gradient_est.setInputCloud(cloud_i1);
  gradient_est.setInputNormals(cloud_n);
  search::KdTree<PointXYZI>::Ptr treept2 (new search::KdTree<PointXYZI> (false));
  gradient_est.setSearchMethod(treept2);
  gradient_est.setRadiusSearch(0.25);
  gradient_est.compute(*cloud_ig);
  cout << "Intensity gradient estimated with size " << cloud_ig->points.size() << endl;
  cout << cloud_ig->points[10] << endl;

  // Estimate the RIFT feature
  RIFTEstimation<PointXYZI, IntensityGradient, Histogram<32> > rift_est;
  search::KdTree<PointXYZI>::Ptr treept3 (new search::KdTree<PointXYZI> (false));
  rift_est.setSearchMethod(treept3);
  rift_est.setRadiusSearch(0.1);
  rift_est.setNrDistanceBins(5);
  rift_est.setNrGradientBins(5);

  // rift_est.setInputCloud(cloud_i1);

  PointCloud<PointXYZI>::Ptr keypoints_xyzi (new PointCloud<PointXYZI>);
  PointCloudXYZRGBtoXYZI(*keypoints1, *keypoints_xyzi);
  rift_est.setInputCloud(keypoints_xyzi);
  rift_est.setSearchSurface(cloud_i1);

  rift_est.setInputGradient(cloud_ig);
  PointCloud<Histogram<32> > rift_output;
  rift_est.compute(rift_output);

  cout << "Rift feature estimated with size " << rift_output.points.size() << endl;


  for (size_t i = 0; i < rift_output.points.size(); i++) {
    cout << "Rift output " << rift_output.points[i] << endl;
  }
  //////////////////////////
*/

  // cout << "~~~~~" << rift_output1->points[100]<< endl;

  // Find feature correspondences
  vector<int> correspondences;
  vector<float> correspondence_scores;
  find_feature_correspondences(descriptors1, descriptors2, correspondences, correspondence_scores);
  // find_rift_correspondences(rift_output1, rift_output2, correspondences, correspondence_scores);


  CorrespondencesPtr pCorrespondences (new Correspondences);
  pCorrespondences->resize(correspondences.size());

  for (size_t i = 0; i < correspondences.size(); i++) {
    (*pCorrespondences)[i].index_query = i;
    (*pCorrespondences)[i].index_match = correspondences[i];
  }



  // std::cout << " Max iterations : " << rej.getMaximumIterations() << std::endl;
  // std::cout << "Refine model : " << rej.getRefineModel() << std::endl;
  Eigen::Matrix4f initial_T = Eigen::Matrix4f::Identity();
  float best_threshold = 0.0;
  float best_fitness_score = 100;

  // for (float threshold = 0.03; threshold < 0.15; threshold += 0.01) {
  //   initial_T = getCorrespondenceRejectionTransformation(pCorrespondences, keypoints1, keypoints2, threshold);

  //   // Apply transformation to the first cloud
  //   transformPointCloud(*cloud1, *tf_cloud1, initial_T);
  //   float fitness_score = getFitnessScore(tf_cloud1, cloud2);
  //   cout << threshold << " fitness score: " << fitness_score << endl;

  //   if (fitness_score < best_fitness_score) {
  //     best_fitness_score = fitness_score;
  //     best_threshold = threshold;
  //   }
  // }

  cout << "Best transformation: " << endl;
  initial_T = getCorrespondenceRejectionTransformation(pCorrespondences, keypoints1, keypoints2, 0.05);
  // Apply transformation to the first cloud
  transformPointCloud(*cloud1, *tf_cloud1, initial_T);

  cout << best_threshold << " fitness score: " << getFitnessScore(tf_cloud1, cloud2) << endl;


  // std::vector<int> *inlier_indices = new std::vector<int>();
  // std::vector<int> inlier_indices;
  // rej.getInliersIndices(inlier_indices);
  // std::cout << "Size of inlier indices: " << inlier_indices.size()  << std::endl;
  // std::cout << "Size of correspondences: " << pCorrespondences->size()  << std::endl;

  // Get the distances between points in the inlier indices
  transformPointCloud(*keypoints1, *tf_keypoints1, initial_T);

  for (size_t i = 0; i < pCorrespondences->size(); i++) {
    // std::cout << (*pCorrespondences)[i].index_query << endl;
    float dx = tf_keypoints1->points[(*pCorrespondences)[i].index_query].x - keypoints2->points[(*pCorrespondences)[i].index_match].x;
    float dy = tf_keypoints1->points[(*pCorrespondences)[i].index_query].y - keypoints2->points[(*pCorrespondences)[i].index_match].y;
    float dz = tf_keypoints1->points[(*pCorrespondences)[i].index_query].z - keypoints2->points[(*pCorrespondences)[i].index_match].z;
    // std::cout << sqrt(dx*dx + dy*dy + dz*dz) << std::endl;

  }




  // Visualize the keypoints
  // visualize_keypoints(cloud1, keypoints1);
  // visualize_keypoints(cloud2, keypoints2);

  // Visualize the two point clouds and their feature correspondences
  // visualize_correspondences_scores(cloud1, keypoints1, cloud2, keypoints2, correspondences, correspondence_scores);
  // visualize_correspondences(cloud1, keypoints1, cloud2, keypoints2, pCorrespondences);


  // Initial alignment of point clouds, see http://www.pointclouds.org/assets/iros2011/registration.pdf
  // cout << "Executing initial alignment of point clouds" << endl;
  // SampleConsensusInitialAlignment<PointXYZRGB, PointXYZRGB, FPFHSignature33> sac;
  // SampleConsensusInitialAlignment<PointXYZRGB, PointXYZRGB, FPFHSignature33> sac;
  // SampleConsensusInitialAlignment<PointXYZRGB, PointXYZRGB, PFHRGBSignature250> sac;

  // cout << "Size of keypoints1: " << keypoints1->points.size() << endl;
  // cout << "Size of original cloud: " << downsampled1->points.size() << endl;
  // cout << "Size of descriptors1: " << rift_output1->points.size() << endl;
  // cout << "Size of keypoints2: " << keypoints2->points.size() << endl;
  // cout << "Size of descriptors2: " << rift_output2->points.size() << endl;


  // sac.setMinSampleDistance(0.001);
  // sac.setMaxCorrespondenceDistance(0.1);
  // sac.setNumberOfSamples(100);
  // sac.setCorrespondenceRandomness(50);
  // sac.setMaximumIterations(3000);

  // sac.setInputSource(keypoints1);

  // sac.setSourceFeatures(descriptors1);

  // sac.setInputTarget(keypoints2);

  // sac.setTargetFeatures(descriptors2); 

  // cout << "aligning" << endl;
  // sac.align(*cloud3);

  // initial_T = sac.getFinalTransformation();
  // cout << initial_T << endl;
  
  // cout << "Output size:  " << cloud3->points.size() << endl;
  // cout << "Source cloud size: " << downsampled1->points.size() << endl;
  // cout << "Target cloud size: " << downsampled2->points.size() << endl;

  
  ////////////////////////////////////////////////


  // Define a translation
  Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
  // transform_1(0,3) = 10;






  // Get fitness score
  // Registration<PointXYZRGB, PointXYZRGB, float>::getFitnessScore
  *cloud3 = *cloud2 + *tf_cloud1;



  // transformPointCloud(*keypoints1, *tf_cloud1, initial_T);

  // *cloud3 = *keypoints2 + *keypoints1;

  // Show keypoints cloud instead
  // PointCloud<PointXYZRGB>::Ptr keypoints_xyzrgb(new PointCloud<PointXYZRGB>);
  // copyPointCloud(*keypoints2, *keypoints_xyzrgb);
  // *cloud3 = *keypoints_xyzrgb;


  // //////////////////////////////////////////////////////////////////////////////////////
  // // Get rough overlapping regions before applying ICP
  // PointCloud<PointXYZRGB>::Ptr cloud1_overlapping = getCloud2(cloud2, tf_cloud1, 0.15f);
  // PointCloud<PointXYZRGB>::Ptr cloud2_overlapping = getCloud2(tf_cloud1, cloud2, 0.15f);
  // /////////////////////////////////////////////////////////////////////////////////////////
  
  // Apply ICP for refined alignment
  /*
  cout << "Starting ICP" << endl;
  
  IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
  icp.setMaxCorrespondenceDistance(1);
  // icp.setRANSACOutlierRejectionThreshold(1.5);
  // icp.setTransformationEpsilon(0.0001);
  // icp.setMaximumIterations(1);

  icp.setInputSource(tf_cloud1);
  icp.setInputTarget(cloud2);


  PointCloud<PointXYZRGB>::Ptr registration_output (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr final_cloud (new PointCloud<PointXYZRGB>);

  icp.align(*registration_output);

  Eigen::Matrix4f refined_T = icp.getFinalTransformation();
  // cout << "refined_T " << refined_T << endl;

  cout << "Fitness score: " << icp.getFitnessScore() << endl;


  transformPointCloud(*tf_cloud1, *final_cloud, refined_T);

  // Concatenate clouds
  *cloud3 = *cloud2 + *final_cloud; 
*/



  ///////////////////////////////////////////////////////////////////////////////////////////////////
  // Make node handler and publisher
  ros::init(argc, argv, "pcl_create");
  ROS_INFO("Started PCL publishing node");


  ros::NodeHandle nh;
  ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2> ("pcl_output", 1);

  // Make point cloud into ROS message
  sensor_msgs::PointCloud2 output;
  toROSMsg(*cloud3, output);
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

