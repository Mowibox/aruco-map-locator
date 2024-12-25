#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>

const std::string CALIBRATION_FILE = "../cam_params.yaml";
// Frame dimensions
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

sensor_msgs::msg::Image::SharedPtr mat_to_image_msg(const cv::Mat &frame, const rclcpp::Clock::SharedPtr &clock) {
   /** 
    * Converts a openCV matrix to a ROS 2 Image msg
    *@param frame: The provided frame
    *@param clock: The clock
    */
    auto msg = std::make_shared<sensor_msgs::msg::Image>();
    msg->header.stamp = clock->now();
    msg->header.frame_id = "camera_frame";
    msg->height = frame.rows;
    msg->width = frame.cols;
    msg->encoding = "bgr8";
    msg->is_bigendian = false;
    msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(frame.step);

    msg->data.assign(frame.data, frame.data + (frame.rows * frame.step));

    return msg;
}

class ImagePublisher : public rclcpp::Node{
    public:
    ImagePublisher(cv::VideoCapture &capture, const cv::Mat & camera_matrix, const cv::Mat &dist_coeffs)
        : Node("image_publisher"), capture_(capture), camera_matrix_(camera_matrix), dist_coeffs_(dist_coeffs) {
            publisher_ = this->create_publisher<sensor_msgs::msg::Image>("camera_feed", 10);
            timer_ = this->create_wall_timer(std::chrono::duration<double>(0.1), std::bind(&ImagePublisher::publish_frame, this));
    }

    private:
    void publish_frame(){
       /** 
        * Publishes the frames to the camera_feed topic
        */
        cv::Mat frame;
        if (!capture_.read(frame)) {
            RCLCPP_WARN(this->get_logger(), "Failed to read camera feed.");
            return;
        }

        cv::Mat corrected_frame;
        cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix_, dist_coeffs_, frame.size(), 1);
        cv::undistort(frame, corrected_frame, camera_matrix_, dist_coeffs_, new_camera_matrix);

        auto msg = mat_to_image_msg(corrected_frame, this->get_clock());

        publisher_->publish(*msg);
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture &capture_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
        
};

bool load_calibration_params(const std::string &filepath, cv::Mat &camera_matrix, cv::Mat &dist_coeffs){
   /** 
    * Loads the camera calibration parameters specified in the provided yaml file
    *@param filepath: The yaml file path 
    *@param camera_matrix: The camera matrix
    *@param dist_coeffs: The distortion coefficientss
    */
    try{
        YAML::Node config = YAML::LoadFile(filepath);
        auto camera_matrix_data = config["camera_matrix"].as<std::vector<std::vector<double>>>();
        camera_matrix = cv::Mat(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                camera_matrix.at<double>(i, j) = camera_matrix_data[i][j];
            }
        }

        auto dist_coeffs_data = config["distorsion_coefficients"].as<std::vector<std::vector<double>>>();
        dist_coeffs = cv::Mat(dist_coeffs_data.size(), dist_coeffs_data[0].size(), CV_64F);
        for (size_t i = 0; i < dist_coeffs_data.size(); ++i) {
            for (size_t j = 0; j < dist_coeffs_data[i].size(); ++j) {
                dist_coeffs.at<double>(i, j) = dist_coeffs_data[i][j];
            }
        }

        return true;

    } catch (const YAML::Exception &e){
        std::cerr << "Failed to load calibraiton parameters." << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    
    cv::Mat camera_matrix, dist_coeffs;
    if (!load_calibration_params(CALIBRATION_FILE, camera_matrix, dist_coeffs)){
        return EXIT_FAILURE;
    }

    cv::VideoCapture capture(0);
    if (!capture.isOpened()){
        std::cerr << "Failed to open camera." << std::endl;
        return EXIT_FAILURE;
    }

    capture.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

    auto node = std::make_shared<ImagePublisher>(capture, camera_matrix, dist_coeffs);
    rclcpp::spin(node);

    rclcpp::shutdown();
    return EXIT_SUCCESS;
}