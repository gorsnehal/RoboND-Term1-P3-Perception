#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_data =  ros_to_pcl(pcl_msg)

    # outlier filtering
    olfilter = pcl_data.make_statistical_outlier_filter()
    # set number of neigbouring points to be analyzed
    olfilter.set_mean_k(20)
    # set threshold scale factor
    th_scale_factor = 1.0
    # mean + th_scale_factor * sd
    olfilter.set_std_dev_mul_thresh(th_scale_factor)
    # Apply the filter
    cloud_filtered = olfilter.filter()

    # TODO: Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()
    # Leaf size for downsampling of the grid
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # TODO: PassThrough Filter to isolate table & objects
    # For z direction
    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name('z')
    axis_min = 0.6
    axis_max = 0.9
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    # For y direction
    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name('y')
    axis_min = -0.5
    axis_max = 0.5
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    # TODO: RANSAC Plane Segmentation to identify tabletop & remove it
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coeff = seg.segment()

    # TODO: Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative = False)
    cloud_objects = cloud_filtered.extract(inliers, negative = True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerences for distance threshold
    # as well as min & max cluster size (in points)
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(1500)
    #Serach kd tree for clusters
    ec.set_SearchMethod(tree)
    #Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each  segmented object in the scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])


    # Cretae new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert to ROS message
    cluster_cloud_msg = pcl_to_ros(cluster_cloud)

    # TODO: Convert PCL data to ROS messages
    table_msg = pcl_to_ros(cloud_table)
    objects_msg = pcl_to_ros(cloud_objects)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(objects_msg)
    pcl_table_pub.publish(table_msg)
    pcl_cluster_pub.publish(cluster_cloud_msg)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_object_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)

        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_object_labels.append(label)

        # Publish a label into RViz
	label_pos = list(white_cloud[pts_list[0]])
	label_pos[2] += .4
	objects_markers_pub.publish(make_label(label,label_pos,index))

        # Add the detected object to the list of detected objects.
	do = DetectedObject()
	do.label = label
	do.cloud = pcl_to_ros(pcl_cluster)
	detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_object_labels),detected_object_labels))

    # Publish the list of detected objects
    detected_object_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    # test scene number
    test_scene_num = Int32()
    test_scene_num.data = 1
    # object name
    object_name = String()
    # arm name
    arm_name = String()
    # Pick Pose
    pick_pose = Pose()
    # Place pose
    place_pose = Pose()

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')

    # TODO: Parse parameters into individual variables
    object_name_list = []
    object_group_list = []
    for i, object_param_i in enumerate(object_list_param):
        object_name_list.append(object_param_i['name'])
        object_group_list.append(object_param_i['group'])

    # Pick up from dropbox parameters
    dropbox_list_param = rospy.get_param('/dropbox')
    # TODO: Parse parameters into individual variables
    dropbox_name_list = []
    dropbox_group_list = []
    dropbox_position_list = np.zeros((2,3))
    for i, dropbox_param_i in enumerate(dropbox_list_param):
        dropbox_name_list.append(dropbox_param_i['name'])
        dropbox_group_list.append(dropbox_param_i['group'])
        dropbox_position_list[i,:] = dropbox_param_i['position']

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    centroids = []
    dict_list = []
    for i in range(len(object_name_list)):
        obj_name_i = object_name_list[i]
        obj_group_i = object_group_list[i]

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        pth_mean = []
        np_mean = np.zeros(3)
        for j, object_j in enumerate(object_list):
            
            if(obj_name_i == object_j.label):
            	cloud_arr = ros_to_pcl(object_j.cloud).to_array()
	        np_mean = np.mean(cloud_arr, axis=0)[:3]
                pth_mean.append(np.asscalar(np_mean[0]))
                pth_mean.append(np.asscalar(np_mean[1]))
                pth_mean.append(np.asscalar(np_mean[2]))

        # if found correct label then only go for further process
	if pth_mean:
            centroids.append(pth_mean)
        # no correct label found
        else:
            print('list empty')
            continue

        # TODO: Create 'place_pose' for the object

        # object name
        object_name.data = obj_name_i
        # arm name
        if obj_group_i == 'green':
            arm_name.data = 'right'
        else:
            arm_name.data = 'left'
        # Pick Pose
        pick_pose.position.x = pth_mean[0]
        pick_pose.position.y = pth_mean[1]
        pick_pose.position.z = pth_mean[2]
        # TODO: Assign the arm to be used for pick_place
        # Place pose
        for j, dropbox_group_j in enumerate(dropbox_group_list):
            if dropbox_group_j == obj_group_i:
                 place_pose.position.x = np.asscalar(dropbox_position_list[j][0])
                 place_pose.position.y = np.asscalar(dropbox_position_list[j][1])
                 place_pose.position.z = np.asscalar(dropbox_position_list[j][2])

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
#        rospy.wait_for_service('pick_place_routine')

#        try:
#            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

#            # TODO: Insert your message variables to be sent as a service request
#            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

#            print ("Response: ",resp.success)

#        except rospy.ServiceException, e:
#            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    file_name = 'output_' + str(test_scene_num.data) + '.yaml'
    send_to_yaml(file_name, dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node("pr2_object_recognition", anonymous=True)

    # TODO: Create Subscribers
    rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    objects_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_object_pub = rospy.Publisher("/detected_object", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav','rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
