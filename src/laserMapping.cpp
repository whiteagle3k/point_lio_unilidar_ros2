#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <Eigen/Core>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/vector3.hpp>
// #include <livox_ros_driver/msg/custom_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

#include "IMU_Processing.hpp"
#include "parameters.h"
#include "Estimator.h"

#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

using std::placeholders::_1;

const float MOV_THRESHOLD = 1.5f;

using PointCloudXYZI = pcl::PointCloud<pcl::PointXYZI>;
using PointType = pcl::PointXYZI;

class LaserMapping : public rclcpp::Node
{
public:
    LaserMapping()
        : Node("laser_mapping")
    {
        // Initialize parameters
        readParameters(shared_from_this());

        RCLCPP_INFO(this->get_logger(), "lidar_type: %d", lidar_type);

        path.header.frame_id = "camera_init";

        frame_num = 0;
        aver_time_consu = 0;
        aver_time_icp = 0;
        aver_time_match = 0;
        aver_time_incre = 0;
        aver_time_solve = 0;
        aver_time_propag = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG)*0.5 * PI_M / 180.0);

        memset(point_selected_surf, true, sizeof(point_selected_surf));

        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);

        if (extrinsic_est_en)
        {
            if (!use_imu_as_input)
            {
                kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
                kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
            }
            else
            {
                kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
                kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
            }
        }

        p_imu->lidar_type = p_pre->lidar_type = lidar_type;
        p_imu->imu_en = imu_en;

        kf_input.init_dyn_share_modified(get_f_input, df_dx_input, h_model_input);
        kf_output.init_dyn_share_modified_2h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);

        Eigen::Matrix<double, 24, 24> P_init = MD(24, 24)::Identity() * 0.01;
        P_init.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
        P_init.block<6, 6>(15, 15) = MD(6, 6)::Identity() * 0.001;
        P_init.block<6, 6>(6, 6) = MD(6, 6)::Identity() * 0.0001;
        kf_input.change_P(P_init);

        Eigen::Matrix<double, 30, 30> P_init_output = MD(30, 30)::Identity() * 0.01;
        P_init_output.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
        P_init_output.block<6, 6>(6, 6) = MD(6, 6)::Identity() * 0.0001;
        P_init_output.block<6, 6>(24, 24) = MD(6, 6)::Identity() * 0.001;
        kf_input.change_P(P_init);
        kf_output.change_P(P_init_output);

        Q_input = process_noise_cov_input();
        Q_output = process_noise_cov_output();

        /*** debug record ***/
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(), "w");

        string mat_out_dir = string(DEBUG_FILE_DIR("mat_out.txt"));
        fout_out.open(mat_out_dir, std::ios::out);

        string imu_pbp_dir = string(DEBUG_FILE_DIR("imu_pbp.txt"));
        fout_imu_pbp.open(imu_pbp_dir, std::ios::out);
        if (fout_out && fout_imu_pbp)
            RCLCPP_INFO(this->get_logger(), "~~~~%s file opened", ROOT_DIR);
        else
            RCLCPP_INFO(this->get_logger(), "~~~~%s doesn't exist", ROOT_DIR);

        // ROS2 subscription initialization
        if (p_pre->lidar_type == AVIA)
        {
            // For Livox LiDAR, you would need to implement custom message handling
            // sub_pcl_ = this->create_subscription<livox_ros_driver::msg::CustomMsg>(
            //     lid_topic, rclcpp::SensorDataQoS(),
            //     std::bind(&LaserMapping::livox_pcl_cbk, this, _1));
        }
        else
        {
            sub_pcl_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                lid_topic, rclcpp::SensorDataQoS(),
                std::bind(&LaserMapping::standard_pcl_cbk, this, _1));
        }

        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, rclcpp::SensorDataQoS(),
            std::bind(&LaserMapping::imu_cbk, this, _1));

        pubLaserCloudFullRes_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/cloud_registered", rclcpp::QoS(100000));

        pubLaserCloudFullRes_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/cloud_registered_body", rclcpp::QoS(100000));

        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/cloud_effected", rclcpp::QoS(100000));

        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/Laser_map", rclcpp::QoS(100000));

        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/aft_mapped_to_init", rclcpp::QoS(100000));

        pubPath_ = this->create_publisher<nav_msgs::msg::Path>(
            "/path", rclcpp::QoS(100000));

        plane_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/planner_normal", rclcpp::QoS(1000));

        // Create timer for main loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1),
            std::bind(&LaserMapping::main_loop, this));

        // Initialize signals
        signal(SIGINT, LaserMapping::SigHandle);
    }

    ~LaserMapping()
    {
        if (pcl_wait_save->size() > 0 && pcd_save_en)
        {
            string file_name = string("scans.pcd");
            string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
            RCLCPP_INFO(this->get_logger(), "Saving map to file: %s", all_points_dir.c_str());
            pcl::PCDWriter pcd_writer;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
        }
        fout_out.close();
        fout_imu_pbp.close();
        fclose(fp);
    }

private:
    // Declare all member variables and functions here

    static void SigHandle(int sig)
    {
        flg_exit = true;
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "catch sig %d", sig);
        sig_buffer.notify_all();
    }

    inline void dump_lio_state_to_log(FILE *fp)
    {
        V3D rot_ang;
        if (!use_imu_as_input)
        {
            rot_ang = SO3ToEuler(kf_output.x_.rot);
        }
        else
        {
            rot_ang = SO3ToEuler(kf_input.x_.rot);
        }

        fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
        fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));
        if (use_imu_as_input)
        {

            fprintf(fp, "%lf %lf %lf ", kf_input.x_.pos(0), kf_input.x_.pos(1), kf_input.x_.pos(2));
            fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
            fprintf(fp, "%lf %lf %lf ", kf_input.x_.vel(0), kf_input.x_.vel(1), kf_input.x_.vel(2));
            fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
            fprintf(fp, "%lf %lf %lf ", kf_input.x_.bg(0), kf_input.x_.bg(1), kf_input.x_.bg(2));
            fprintf(fp, "%lf %lf %lf ", kf_input.x_.ba(0), kf_input.x_.ba(1), kf_input.x_.ba(2));
            fprintf(fp, "%lf %lf %lf ", kf_input.x_.gravity(0), kf_input.x_.gravity(1), kf_input.x_.gravity(2));
        }
        else
        {
            fprintf(fp, "%lf %lf %lf ", kf_output.x_.pos(0), kf_output.x_.pos(1), kf_output.x_.pos(2));
            fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
            fprintf(fp, "%lf %lf %lf ", kf_output.x_.vel(0), kf_output.x_.vel(1), kf_output.x_.vel(2));
            fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
            fprintf(fp, "%lf %lf %lf ", kf_output.x_.bg(0), kf_output.x_.bg(1), kf_output.x_.bg(2));
            fprintf(fp, "%lf %lf %lf ", kf_output.x_.ba(0), kf_output.x_.ba(1), kf_output.x_.ba(2));
            fprintf(fp, "%lf %lf %lf ", kf_output.x_.gravity(0), kf_output.x_.gravity(1), kf_output.x_.gravity(2));
        }
        fprintf(fp, "\r\n");
        fflush(fp);
    }

    void pointBodyLidarToIMU(PointType const *const pi, PointType *const po)
    {

        V3D p_body_lidar(pi->x, pi->y, pi->z);
        V3D p_body_imu;
        if (extrinsic_est_en)
        {
            if (!use_imu_as_input)
            {
                p_body_imu = kf_output.x_.offset_R_L_I.normalized() * p_body_lidar + kf_output.x_.offset_T_L_I;
            }
            else
            {
                p_body_imu = kf_input.x_.offset_R_L_I.normalized() * p_body_lidar + kf_input.x_.offset_T_L_I;
            }
        }
        else
        {
            p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
        }

        po->x = p_body_imu(0);
        po->y = p_body_imu(1);
        po->z = p_body_imu(2);

        po->intensity = pi->intensity;
    }

    void points_cache_collect()
    {
        PointVector points_history;
        ikdtree.acquire_removed_points(points_history);
        points_cache_size = points_history.size();
    }

    void lasermap_fov_segment()
    {
        cub_needrm.shrink_to_fit();

        V3D pos_LiD;
        if (use_imu_as_input)
        {
            pos_LiD = kf_input.x_.pos + kf_input.x_.rot.normalized() * Lidar_T_wrt_IMU;
        }
        else
        {
            pos_LiD = kf_output.x_.pos + kf_output.x_.rot.normalized() * Lidar_T_wrt_IMU;
        }

        if (!Localmap_Initialized)
        {
            for (int i = 0; i < 3; i++)
            {
                LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
                LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
            }
            Localmap_Initialized = true;
            return;
        }

        float dist_to_map_edge[3][2];
        bool need_move = false;
        for (int i = 0; i < 3; i++)
        {
            dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
            dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
                need_move = true;
        }

        if (!need_move)
            return;
        BoxPointType New_LocalMap_Points, tmp_boxpoints;
        New_LocalMap_Points = LocalMap_Points;
        float mov_dist = std::max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
        for (int i = 0; i < 3; i++)
        {
            tmp_boxpoints = LocalMap_Points;
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
            {
                New_LocalMap_Points.vertex_max[i] -= mov_dist;
                New_LocalMap_Points.vertex_min[i] -= mov_dist;
                tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
                cub_needrm.emplace_back(tmp_boxpoints);
            }
            else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            {
                New_LocalMap_Points.vertex_max[i] += mov_dist;
                New_LocalMap_Points.vertex_min[i] += mov_dist;
                tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
                cub_needrm.emplace_back(tmp_boxpoints);
            }
        }
        LocalMap_Points = New_LocalMap_Points;

        points_cache_collect();
        if (cub_needrm.size() > 0)
            int kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    }

    void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mtx_buffer);

        scan_count++;

        double preprocess_start_time = omp_get_wtime();

        if (msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9 < last_timestamp_lidar)
        {
            RCLCPP_ERROR(this->get_logger(), "lidar loop back, clear buffer");

            return;
        }

        last_timestamp_lidar = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
        PointCloudXYZI::Ptr ptr_div(new PointCloudXYZI());

        double time_div = last_timestamp_lidar;

        p_pre->process(msg, ptr);

        if (cut_frame)
        {

            sort(ptr->points.begin(), ptr->points.end(), time_list);

            for (size_t i = 0; i < ptr->size(); i++)
            {

                ptr_div->push_back(ptr->points[i]);

                if (ptr->points[i].curvature / double(1000) + last_timestamp_lidar - time_div > cut_frame_time_interval)
                {
                    if (ptr_div->size() < 1)
                        continue;

                    PointCloudXYZI::Ptr ptr_div_i(new PointCloudXYZI());
                    *ptr_div_i = *ptr_div;

                    lidar_buffer.push_back(ptr_div_i);

                    time_buffer.push_back(time_div);
                    time_div += ptr->points[i].curvature / double(1000);
                    ptr_div->clear();
                }
            }

            if (!ptr_div->empty())
            {
                lidar_buffer.push_back(ptr_div);

                time_buffer.push_back(time_div);
            }
        }
        else if (con_frame)
        {

            if (frame_ct == 0)
            {
                time_con = last_timestamp_lidar;
            }

            if (frame_ct < con_frame_num)
            {
                for (size_t i = 0; i < ptr->size(); i++)
                {
                    ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                    ptr_con->push_back(ptr->points[i]);
                }
                frame_ct++;
            }

            else
            {
                PointCloudXYZI::Ptr ptr_con_i(new PointCloudXYZI());
                *ptr_con_i = *ptr_con;
                lidar_buffer.push_back(ptr_con_i);
                double time_con_i = time_con;
                time_buffer.push_back(time_con_i);
                ptr_con->clear();
                frame_ct = 0;
            }
        }
        else
        {
            lidar_buffer.emplace_back(ptr);
            time_buffer.emplace_back(last_timestamp_lidar);
        }
        s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
        sig_buffer.notify_all();
    }

    void imu_cbk(const sensor_msgs::msg::Imu::SharedPtr msg_in)
    {
        publish_count++;

        sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

        double imu_time = msg_in->header.stamp.sec + msg_in->header.stamp.nanosec * 1e-9;
        msg->header.stamp = rclcpp::Time(imu_time - time_lag_imu_to_lidar);

        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        std::lock_guard<std::mutex> lock(mtx_buffer);

        if (timestamp < last_timestamp_imu)
        {
            RCLCPP_ERROR(this->get_logger(), "imu loop back, clear deque");

            return;
        }

        imu_deque.emplace_back(msg);

        last_timestamp_imu = timestamp;

        sig_buffer.notify_all();
    }

    bool sync_packages(MeasureGroup &meas)
    {

        if (!imu_en)
        {
            if (!lidar_buffer.empty())
            {

                meas.lidar = lidar_buffer.front();
                meas.lidar_beg_time = time_buffer.front();
                time_buffer.pop_front();
                lidar_buffer.pop_front();

                if (meas.lidar->points.size() < 1)
                {
                    RCLCPP_INFO(this->get_logger(), "lose lidar");
                    return false;
                }

                double end_time = meas.lidar->points.back().curvature;
                for (auto pt : meas.lidar->points)
                {
                    if (pt.curvature > end_time)
                    {
                        end_time = pt.curvature;
                    }
                }
                lidar_end_time = meas.lidar_beg_time + end_time / double(1000);

                meas.lidar_last_time = lidar_end_time;

                return true;
            }
            return false;
        }

        if (lidar_buffer.empty() || imu_deque.empty())
        {
            return false;
        }

        /*** push a lidar scan ***/
        if (!lidar_pushed)
        {

            meas.lidar = lidar_buffer.front();

            if (meas.lidar->points.size() < 1)
            {
                RCLCPP_INFO(this->get_logger(), "lose lidar");
                lidar_buffer.pop_front();
                time_buffer.pop_front();
                return false;
            }

            meas.lidar_beg_time = time_buffer.front();

            double end_time = meas.lidar->points.back().curvature;
            for (auto pt : meas.lidar->points)
            {
                if (pt.curvature > end_time)
                {
                    end_time = pt.curvature;
                }
            }
            lidar_end_time = meas.lidar_beg_time + end_time / double(1000);

            meas.lidar_last_time = lidar_end_time;
            lidar_pushed = true;
        }

        if (last_timestamp_imu < lidar_end_time)
        {
            return false;
        }

        /*** push imu data, and pop from imu buffer ***/
        if (p_imu->imu_need_init_)
        {
            double imu_time = imu_deque.front()->header.stamp.sec + imu_deque.front()->header.stamp.nanosec * 1e-9;
            meas.imu.shrink_to_fit();
            while ((!imu_deque.empty()) && (imu_time < lidar_end_time))
            {
                imu_time = imu_deque.front()->header.stamp.sec + imu_deque.front()->header.stamp.nanosec * 1e-9;
                if (imu_time > lidar_end_time)
                    break;
                meas.imu.emplace_back(imu_deque.front());
                imu_last = imu_next;
                imu_last_ptr = imu_deque.front();
                imu_next = *(imu_deque.front());
                imu_deque.pop_front();
            }
        }
        else if (!init_map)
        {
            double imu_time = imu_deque.front()->header.stamp.sec + imu_deque.front()->header.stamp.nanosec * 1e-9;
            meas.imu.shrink_to_fit();
            meas.imu.emplace_back(imu_last_ptr);

            while ((!imu_deque.empty()) && (imu_time < lidar_end_time))
            {
                imu_time = imu_deque.front()->header.stamp.sec + imu_deque.front()->header.stamp.nanosec * 1e-9;
                if (imu_time > lidar_end_time)
                    break;
                meas.imu.emplace_back(imu_deque.front());
                imu_last = imu_next;
                imu_last_ptr = imu_deque.front();
                imu_next = *(imu_deque.front());
                imu_deque.pop_front();
            }
        }

        lidar_buffer.pop_front();
        time_buffer.pop_front();
        lidar_pushed = false;
        return true;
    }

    void process_increments()
    {
        PointVector PointToAdd;
        PointVector PointNoNeedDownsample;
        PointToAdd.reserve(feats_down_size);
        PointNoNeedDownsample.reserve(feats_down_size);

        for (int i = 0; i < feats_down_size; i++)
        {
            if (!Nearest_Points[i].empty())
            {
                const PointVector &points_near = Nearest_Points[i];
                bool need_add = true;
                PointType downsample_result, mid_point;
                mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
                mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
                mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
                /* If the nearest points is definitely outside the downsample box */
                if (fabs(points_near[0].x - mid_point.x) > 1.732 * filter_size_map_min || fabs(points_near[0].y - mid_point.y) > 1.732 * filter_size_map_min || fabs(points_near[0].z - mid_point.z) > 1.732 * filter_size_map_min)
                {
                    PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
                    continue;
                }
                /* Check if there is a point already in the downsample box */
                float dist = calc_dist<float>(feats_down_world->points[i], mid_point);
                for (size_t readd_i = 0; readd_i < points_near.size(); readd_i++)
                {
                    /* Those points which are outside the downsample box should not be considered. */
                    if (fabs(points_near[readd_i].x - mid_point.x) < 0.5 * filter_size_map_min && fabs(points_near[readd_i].y - mid_point.y) < 0.5 * filter_size_map_min && fabs(points_near[readd_i].z - mid_point.z) < 0.5 * filter_size_map_min)
                    {
                        need_add = false;
                        break;
                    }
                }
                if (need_add)
                    PointToAdd.emplace_back(feats_down_world->points[i]);
            }
            else
            {

                PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
            }
        }
        int add_point_size = ikdtree.Add_Points(PointToAdd, true);
        ikdtree.Add_Points(PointNoNeedDownsample, false);
    }

    void publish_init_kdtree()
    {
        int size_init_ikdtree = ikdtree.size();
        PointCloudXYZI::Ptr laserCloudInit(new PointCloudXYZI(size_init_ikdtree, 1));

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        PointVector().swap(ikdtree.PCL_Storage);
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);

        laserCloudInit->points = ikdtree.PCL_Storage;
        pcl::toROSMsg(*laserCloudInit, laserCloudmsg);

        laserCloudmsg.header.stamp = rclcpp::Time(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudMap_->publish(laserCloudmsg);
    }

    void publish_frame_world()
    {
        if (scan_pub_en)
        {
            PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
            int size = laserCloudFullRes->points.size();

            PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

            for (int i = 0; i < size; i++)
            {

                laserCloudWorld->points[i].x = feats_down_world->points[i].x;
                laserCloudWorld->points[i].y = feats_down_world->points[i].y;
                laserCloudWorld->points[i].z = feats_down_world->points[i].z;
                laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
            }
            sensor_msgs::msg::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

            laserCloudmsg.header.stamp = rclcpp::Time(lidar_end_time);
            laserCloudmsg.header.frame_id = "camera_init";
            pubLaserCloudFullRes_->publish(laserCloudmsg);
            publish_count -= PUBFRAME_PERIOD;
        }

        /**************** save map ****************/
        /* 1. make sure you have enough memories
        /* 2. noted that pcd save will influence the real-time performences **/
        if (pcd_save_en)
        {
            int size = feats_down_world->points.size();
            PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

            for (int i = 0; i < size; i++)
            {
                laserCloudWorld->points[i].x = feats_down_world->points[i].x;
                laserCloudWorld->points[i].y = feats_down_world->points[i].y;
                laserCloudWorld->points[i].z = feats_down_world->points[i].z;
                laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
            }

            *pcl_wait_save += *laserCloudWorld;

            static int scan_wait_num = 0;
            scan_wait_num++;
            if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
            {
                pcd_index++;
                string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + std::to_string(pcd_index) + string(".pcd"));
                pcl::PCDWriter pcd_writer;
                RCLCPP_INFO(this->get_logger(), "current scan saved to /PCD/%s", all_points_dir.c_str());
                pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
                pcl_wait_save->clear();
                scan_wait_num = 0;
            }
        }
    }

    void publish_frame_body()
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointBodyLidarToIMU(&feats_undistort->points[i],
                                &laserCloudIMUBody->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
        laserCloudmsg.header.stamp = rclcpp::Time(lidar_end_time);
        laserCloudmsg.header.frame_id = "body";
        pubLaserCloudFullRes_body_->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    template <typename T>
    void set_posestamp(T &out)
    {
        if (!use_imu_as_input)
        {
            out.position.x = kf_output.x_.pos(0);
            out.position.y = kf_output.x_.pos(1);
            out.position.z = kf_output.x_.pos(2);
            out.orientation.x = kf_output.x_.rot.coeffs()[0];
            out.orientation.y = kf_output.x_.rot.coeffs()[1];
            out.orientation.z = kf_output.x_.rot.coeffs()[2];
            out.orientation.w = kf_output.x_.rot.coeffs()[3];
        }
        else
        {
            out.position.x = kf_input.x_.pos(0);
            out.position.y = kf_input.x_.pos(1);
            out.position.z = kf_input.x_.pos(2);
            out.orientation.x = kf_input.x_.rot.coeffs()[0];
            out.orientation.y = kf_input.x_.rot.coeffs()[1];
            out.orientation.z = kf_input.x_.rot.coeffs()[2];
            out.orientation.w = kf_input.x_.rot.coeffs()[3];
        }
    }

    void publish_odometry()
    {
        odomAftMapped.header.frame_id = "camera_init";
        odomAftMapped.child_frame_id = "aft_mapped";
        if (publish_odometry_without_downsample)
        {
            odomAftMapped.header.stamp = rclcpp::Time(time_current);
        }
        else
        {
            odomAftMapped.header.stamp = rclcpp::Time(lidar_end_time);
        }
        set_posestamp(odomAftMapped.pose.pose);

        pubOdomAftMapped_->publish(odomAftMapped);

        static tf2_ros::TransformBroadcaster br(this);
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp = odomAftMapped.header.stamp;
        transformStamped.header.frame_id = "camera_init";
        transformStamped.child_frame_id = "aft_mapped";

        transformStamped.transform.translation.x = odomAftMapped.pose.pose.position.x;
        transformStamped.transform.translation.y = odomAftMapped.pose.pose.position.y;
        transformStamped.transform.translation.z = odomAftMapped.pose.pose.position.z;
        transformStamped.transform.rotation = odomAftMapped.pose.pose.orientation;

        br.sendTransform(transformStamped);
    }

    void publish_path()
    {
        set_posestamp(msg_body_pose.pose);

        msg_body_pose.header.stamp = rclcpp::Time(lidar_end_time);
        msg_body_pose.header.frame_id = "camera_init";
        static int jjj = 0;
        jjj++;

        {
            path.poses.emplace_back(msg_body_pose);
            pubPath_->publish(path);
        }
    }

    void main_loop()
    {
        if (flg_exit)
            rclcpp::shutdown();

        if (!sync_packages(Measures))
        {
            return;
        }

        if (flg_first_scan)
        {
            first_lidar_time = Measures.lidar_beg_time;
            flg_first_scan = false;
            RCLCPP_INFO(this->get_logger(), "first lidar time %lf", first_lidar_time);
        }

        if (flg_reset)
        {
            RCLCPP_WARN(this->get_logger(), "reset when rosbag play back");
            p_imu->Reset();
            flg_reset = false;
            return;
        }

        double t0, t1, t2, t3, t4, t5, match_start, solve_start;
        match_time = 0;
        solve_time = 0;
        propag_time = 0;
        update_time = 0;
        t0 = omp_get_wtime();

        p_imu->Process(Measures, feats_undistort);

        if (feats_undistort->empty() || feats_undistort == NULL)
        {
            return;
        }

        if (imu_en)
        {
            if (!p_imu->gravity_align_)
            {
                while (Measures.lidar_beg_time > imu_next.header.stamp.sec + imu_next.header.stamp.nanosec * 1e-9)
                {
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());
                    imu_deque.pop_front();
                }
                if (non_station_start)
                {
                    state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                    state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                    state_out.acc << VEC_FROM_ARRAY(gravity_init);
                    state_out.acc *= -1;
                }
                else
                {
                    state_in.gravity = -1 * p_imu->mean_acc * G_m_s2 / acc_norm;
                    state_out.gravity = -1 * p_imu->mean_acc * G_m_s2 / acc_norm;
                    state_out.acc = p_imu->mean_acc * G_m_s2 / acc_norm;
                }
                if (gravity_align)
                {
                    Eigen::Matrix3d rot_init;
                    p_imu->gravity_ << VEC_FROM_ARRAY(gravity);
                    p_imu->Set_init(state_in.gravity, rot_init);
                    state_in.gravity = state_out.gravity = p_imu->gravity_;
                    state_in.rot = state_out.rot = rot_init;
                    state_in.rot.normalize();
                    state_out.rot.normalize();
                    state_out.acc = -rot_init.transpose() * state_out.gravity;
                }
                kf_input.change_x(state_in);
                kf_output.change_x(state_out);
            }
        }
        else
        {
            if (!p_imu->gravity_align_)
            {
                state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                state_out.acc << VEC_FROM_ARRAY(gravity_init);
                state_out.acc *= -1;
            }
        }

        /*** Segment the map in lidar FOV ***/
        lasermap_fov_segment();

        t1 = omp_get_wtime();
        if (space_down_sample)
        {
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
        }
        else
        {
            feats_down_body = Measures.lidar;

            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
        }
        time_seq = time_compressing<int>(feats_down_body);
        feats_down_size = feats_down_body->points.size();

        /*** initialize the map kdtree ***/
        if (!init_map)
        {
            if (ikdtree.Root_Node == nullptr)
            {
                ikdtree.set_downsample_param(filter_size_map_min);
            }

            feats_down_world->resize(feats_down_size);
            for (int i = 0; i < feats_down_size; i++)
            {
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }

            for (size_t i = 0; i < feats_down_world->size(); i++)
            {
                init_feats_world->points.emplace_back(feats_down_world->points[i]);
            }

            if (init_feats_world->size() < init_map_size)
                return;

            ikdtree.Build(init_feats_world->points);
            init_map = true;

            publish_init_kdtree();
            return;
        }

        /*** ICP and Kalman filter update ***/

        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);

        Nearest_Points.resize(feats_down_size);

        t2 = omp_get_wtime();

        /*** iterated state estimation ***/

        crossmat_list.reserve(feats_down_size);
        pbody_list.reserve(feats_down_size);

        for (size_t i = 0; i < feats_down_body->size(); i++)
        {

            V3D point_this(feats_down_body->points[i].x,
                           feats_down_body->points[i].y,
                           feats_down_body->points[i].z);
            pbody_list[i] = point_this;

            if (extrinsic_est_en)
            {
                if (!use_imu_as_input)
                {

                    point_this = kf_output.x_.offset_R_L_I.normalized() * point_this + kf_output.x_.offset_T_L_I;
                }
                else
                {

                    point_this = kf_input.x_.offset_R_L_I.normalized() * point_this + kf_input.x_.offset_T_L_I;
                }
            }
            else
            {
                point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
            }

            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_this);
            crossmat_list[i] = point_crossmat;
        }

        if (!use_imu_as_input)
        {
            bool imu_upda_cov = false;
            effct_feat_num = 0;

            /**** point by point update ****/

            double pcl_beg_time = Measures.lidar_beg_time;
            idx = -1;
            for (k = 0; k < time_seq.size(); k++)
            {

                PointType &point_body = feats_down_body->points[idx + time_seq[k]];

                time_current = point_body.curvature / 1000.0 + pcl_beg_time;

                if (is_first_frame)
                {
                    if (imu_en)
                    {
                        while (time_current > imu_next.header.stamp.sec + imu_next.header.stamp.nanosec * 1e-9)
                        {
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_deque.pop_front();
                        }

                        angvel_avr << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                        acc_avr << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                    }
                    is_first_frame = false;
                    imu_upda_cov = true;
                    time_update_last = time_current;
                    time_predict_last_const = time_current;
                }

                if (imu_en)
                {
                    bool imu_comes = time_current > imu_next.header.stamp.sec + imu_next.header.stamp.nanosec * 1e-9;
                    while (imu_comes)
                    {
                        imu_upda_cov = true;
                        angvel_avr << imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                        acc_avr << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                        /*** covariance update ***/
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                        double dt = imu_last.header.stamp.sec + imu_last.header.stamp.nanosec * 1e-9 - time_predict_last_const;
                        kf_output.predict(dt, Q_output, input_in, true, false);
                        time_predict_last_const = imu_last.header.stamp.sec + imu_last.header.stamp.nanosec * 1e-9;
                        imu_comes = time_current > imu_next.header.stamp.sec + imu_next.header.stamp.nanosec * 1e-9;

                        {
                            double dt_cov = imu_last.header.stamp.sec + imu_last.header.stamp.nanosec * 1e-9 - time_update_last;

                            if (dt_cov > 0.0)
                            {
                                time_update_last = imu_last.header.stamp.sec + imu_last.header.stamp.nanosec * 1e-9;
                                double propag_imu_start = omp_get_wtime();

                                kf_output.predict(dt_cov, Q_output, input_in, false, true);

                                propag_time += omp_get_wtime() - propag_imu_start;
                                double solve_imu_start = omp_get_wtime();
                                kf_output.update_iterated_dyn_share_IMU();
                                solve_time += omp_get_wtime() - solve_imu_start;
                            }
                        }
                    }
                }

                double dt = time_current - time_predict_last_const;
                double propag_state_start = omp_get_wtime();
                if (!prop_at_freq_of_imu)
                {
                    double dt_cov = time_current - time_update_last;
                    if (dt_cov > 0.0)
                    {
                        kf_output.predict(dt_cov, Q_output, input_in, false, true);
                        time_update_last = time_current;
                    }
                }
                kf_output.predict(dt, Q_output, input_in, true, false);
                propag_time += omp_get_wtime() - propag_state_start;
                time_predict_last_const = time_current;

                double t_update_start = omp_get_wtime();

                if (feats_down_size < 1)
                {
                    RCLCPP_WARN(this->get_logger(), "No point, skip this scan!");
                    idx += time_seq[k];
                    continue;
                }
                if (!kf_output.update_iterated_dyn_share_modified())
                {
                    idx = idx + time_seq[k];
                    continue;
                }

                if (prop_at_freq_of_imu)
                {
                    double dt_cov = time_current - time_update_last;
                    if (!imu_en && (dt_cov >= imu_time_inte))
                    {
                        double propag_cov_start = omp_get_wtime();
                        kf_output.predict(dt_cov, Q_output, input_in, false, true);
                        imu_upda_cov = false;
                        time_update_last = time_current;
                        propag_time += omp_get_wtime() - propag_cov_start;
                    }
                }

                solve_start = omp_get_wtime();

                if (publish_odometry_without_downsample)
                {
                    /******* Publish odometry *******/

                    publish_odometry();
                    if (runtime_pos_log)
                    {
                        state_out = kf_output.x_;
                        euler_cur = SO3ToEuler(state_out.rot);
                        fout_out << std::setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_out.pos.transpose() << " " << state_out.vel.transpose()
                                 << " " << state_out.omg.transpose() << " " << state_out.acc.transpose() << " " << state_out.gravity.transpose() << " " << state_out.bg.transpose() << " " << state_out.ba.transpose() << " " << feats_undistort->points.size() << std::endl;
                    }
                }

                for (int j = 0; j < time_seq[k]; j++)
                {
                    PointType &point_body_j = feats_down_body->points[idx + j + 1];
                    PointType &point_world_j = feats_down_world->points[idx + j + 1];
                    pointBodyToWorld(&point_body_j, &point_world_j);
                }

                solve_time += omp_get_wtime() - solve_start;

                update_time += omp_get_wtime() - t_update_start;
                idx += time_seq[k];
            }
        }
        else
        {
            bool imu_prop_cov = false;
            effct_feat_num = 0;

            double pcl_beg_time = Measures.lidar_beg_time;
            idx = -1;
            for (k = 0; k < time_seq.size(); k++)
            {
                PointType &point_body = feats_down_body->points[idx + time_seq[k]];
                time_current = point_body.curvature / 1000.0 + pcl_beg_time;
                if (is_first_frame)
                {
                    while (time_current > imu_next.header.stamp.sec + imu_next.header.stamp.nanosec * 1e-9)
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                    }
                    imu_prop_cov = true;

                    is_first_frame = false;
                    t_last = time_current;
                    time_update_last = time_current;

                    {
                        input_in.gyro << imu_last.angular_velocity.x,
                            imu_last.angular_velocity.y,
                            imu_last.angular_velocity.z;

                        input_in.acc << imu_last.linear_acceleration.x,
                            imu_last.linear_acceleration.y,
                            imu_last.linear_acceleration.z;

                        input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                    }
                }

                while (time_current > imu_next.header.stamp.sec + imu_next.header.stamp.nanosec * 1e-9)
                {
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());
                    imu_deque.pop_front();
                    input_in.gyro << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                    input_in.acc << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;

                    input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                    double dt = imu_last.header.stamp.sec + imu_last.header.stamp.nanosec * 1e-9 - t_last;

                    double dt_cov = imu_last.header.stamp.sec + imu_last.header.stamp.nanosec * 1e-9 - time_update_last;
                    if (dt_cov > 0.0)
                    {
                        kf_input.predict(dt_cov, Q_input, input_in, false, true);
                        time_update_last = imu_last.header.stamp.sec + imu_last.header.stamp.nanosec * 1e-9;
                    }
                    kf_input.predict(dt, Q_input, input_in, true, false);
                    t_last = imu_last.header.stamp.sec + imu_last.header.stamp.nanosec * 1e-9;
                    imu_prop_cov = true;
                }

                double dt = time_current - t_last;
                t_last = time_current;
                double propag_start = omp_get_wtime();

                if (!prop_at_freq_of_imu)
                {
                    double dt_cov = time_current - time_update_last;
                    if (dt_cov > 0.0)
                    {
                        kf_input.predict(dt_cov, Q_input, input_in, false, true);
                        time_update_last = time_current;
                    }
                }
                kf_input.predict(dt, Q_input, input_in, true, false);

                propag_time += omp_get_wtime() - propag_start;

                double t_update_start = omp_get_wtime();

                if (feats_down_size < 1)
                {
                    RCLCPP_WARN(this->get_logger(), "No point, skip this scan!");

                    idx += time_seq[k];
                    continue;
                }
                if (!kf_input.update_iterated_dyn_share_modified())
                {
                    idx = idx + time_seq[k];
                    continue;
                }

                solve_start = omp_get_wtime();

                if (publish_odometry_without_downsample)
                {
                    /******* Publish odometry *******/

                    publish_odometry();
                    if (runtime_pos_log)
                    {
                        state_in = kf_input.x_;
                        euler_cur = SO3ToEuler(state_in.rot);
                        fout_out << std::setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_in.pos.transpose() << " " << state_in.vel.transpose()
                                 << " " << state_in.bg.transpose() << " " << state_in.ba.transpose() << " " << state_in.gravity.transpose() << " " << feats_undistort->points.size() << std::endl;
                    }
                }

                for (int j = 0; j < time_seq[k]; j++)
                {
                    PointType &point_body_j = feats_down_body->points[idx + j + 1];
                    PointType &point_world_j = feats_down_world->points[idx + j + 1];
                    pointBodyToWorld(&point_body_j, &point_world_j);
                }
                solve_time += omp_get_wtime() - solve_start;

                update_time += omp_get_wtime() - t_update_start;
                idx = idx + time_seq[k];
            }
        }

        /******* Publish odometry downsample *******/
        if (!publish_odometry_without_downsample)
        {
            publish_odometry();
        }

        /*** add the feature points to map kdtree ***/
        t3 = omp_get_wtime();

        if (feats_down_size > 4)
        {
            process_increments();
        }

        t5 = omp_get_wtime();

        /******* Publish points *******/

        if (path_en)
            publish_path();

        if (scan_pub_en || pcd_save_en)
            publish_frame_world();

        if (scan_pub_en && scan_body_pub_en)
            publish_frame_body();

        /*** Debug variables Logging ***/
        if (runtime_pos_log)
        {
            frame_num++;
            aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
            {
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + update_time / frame_num;
            }
            aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
            aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + solve_time / frame_num;
            aver_time_propag = aver_time_propag * (frame_num - 1) / frame_num + propag_time / frame_num;
            T1[time_log_counter] = Measures.lidar_beg_time;
            s_plot[time_log_counter] = t5 - t0;
            s_plot2[time_log_counter] = feats_undistort->points.size();
            s_plot3[time_log_counter] = aver_time_consu;
            time_log_counter++;
            RCLCPP_INFO(this->get_logger(), "[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_propag);
            if (!publish_odometry_without_downsample)
            {
                if (!use_imu_as_input)
                {
                    state_out = kf_output.x_;
                    euler_cur = SO3ToEuler(state_out.rot);
                    fout_out << std::setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_out.pos.transpose() << " " << state_out.vel.transpose()
                             << " " << state_out.omg.transpose() << " " << state_out.acc.transpose() << " " << state_out.gravity.transpose() << " " << state_out.bg.transpose() << " " << state_out.ba.transpose() << " " << feats_undistort->points.size() << std::endl;
                }
                else
                {
                    state_in = kf_input.x_;
                    euler_cur = SO3ToEuler(state_in.rot);
                    fout_out << std::setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_in.pos.transpose() << " " << state_in.vel.transpose()
                             << " " << state_in.bg.transpose() << " " << state_in.ba.transpose() << " " << state_in.gravity.transpose() << " " << feats_undistort->points.size() << std::endl;
                }
            }
            dump_lio_state_to_log(fp);
        }
    }

    // Member variables
    std::string lid_topic;
    std::string imu_topic;
    int lidar_type;

    std::mutex mtx_buffer;

    std::string root_dir = ROOT_DIR;

    int feats_down_size = 0;
    int time_log_counter = 0;
    int scan_count = 0;
    int publish_count = 0;

    int frame_ct = 0;
    double time_update_last = 0.0;
    double time_current = 0.0;
    double time_predict_last_const = 0.0;
    double t_last = 0.0;

    std::shared_ptr<ImuProcess> p_imu = std::make_shared<ImuProcess>();
    Preprocess *p_pre = new Preprocess();
    bool init_map = false;
    bool flg_first_scan = true;
    PointCloudXYZI::Ptr ptr_con = std::make_shared<PointCloudXYZI>();

    double T1[MAXN];
    double s_plot[MAXN];
    double s_plot2[MAXN];
    double s_plot3[MAXN];
    double s_plot11[MAXN];

    double match_time = 0;
    double solve_time = 0;
    double propag_time = 0;
    double update_time = 0;

    bool lidar_pushed = false;
    bool flg_reset = false;
    static bool flg_exit;

    std::vector<BoxPointType> cub_needrm;

    std::deque<PointCloudXYZI::Ptr> lidar_buffer;
    std::deque<double> time_buffer;
    std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_deque;

    PointCloudXYZI::Ptr feats_undistort = std::make_shared<PointCloudXYZI>();
    PointCloudXYZI::Ptr feats_down_body = std::make_shared<PointCloudXYZI>();
    PointCloudXYZI::Ptr feats_down_world = std::make_shared<PointCloudXYZI>();
    PointCloudXYZI::Ptr init_feats_world = std::make_shared<PointCloudXYZI>();

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;

    V3D euler_cur;

    MeasureGroup Measures;

    sensor_msgs::msg::Imu imu_last, imu_next;
    sensor_msgs::msg::Imu::SharedPtr imu_last_ptr;
    nav_msgs::msg::Path path;
    nav_msgs::msg::Odometry odomAftMapped;
    geometry_msgs::msg::PoseStamped msg_body_pose;

    double last_timestamp_lidar;
    double last_timestamp_imu;
    double first_lidar_time;
    double lidar_end_time;

    // Subscribers and Publishers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    // Other member variables
    std::ofstream fout_out, fout_imu_pbp;
    FILE *fp;

    // Variables for Kalman filters
    KalmanFilter kf_input, kf_output;
    Eigen::Matrix<double, 24, 24> Q_input;
    Eigen::Matrix<double, 30, 30> Q_output;
    State state_in, state_out;
    Input input_in;

    IKDTree ikdtree;
    std::vector<PointVector> Nearest_Points;
    std::vector<M3D> crossmat_list;
    std::vector<V3D> pbody_list;
    int effct_feat_num;
    V3D angvel_avr, acc_avr;
    std::vector<int> time_seq;
    bool point_selected_surf[MAXN];
    std::shared_ptr<PointCloudXYZI> normvec = std::make_shared<PointCloudXYZI>();

    PointCloudXYZI::Ptr pcl_wait_save = std::make_shared<PointCloudXYZI>();

    // Parameters (should be set via readParameters)
    bool cut_frame;
    bool con_frame;
    double time_con;
    int con_frame_num;
    double cut_frame_time_interval;
    double fov_deg;
    double filter_size_surf_min;
    double filter_size_map_min;
    bool extrinsic_est_en;
    bool use_imu_as_input;
    double extrinT[3];
    double extrinR[9];
    bool imu_en;
    bool publish_odometry_without_downsample;
    bool runtime_pos_log;
    bool path_en;
    bool scan_pub_en;
    bool pcd_save_en;
    int pcd_save_interval;
    int pcd_index;
    bool gravity_align;
    double gravity_init[3];
    double acc_norm;
    double G_m_s2;
    bool prop_at_freq_of_imu;
    double imu_time_inte;
    bool space_down_sample;
    int init_map_size;
    double cube_len;
    double DET_RANGE;
    double time_lag_imu_to_lidar;
    bool scan_body_pub_en;
    bool non_station_start;
};

// Initialize static member variables
bool LaserMapping::flg_exit = false;

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto laser_mapping_node = std::make_shared<LaserMapping>();

    rclcpp::spin(laser_mapping_node);

    rclcpp::shutdown();
    return 0;
}
