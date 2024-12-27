#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>

const std::string CALIBRATION_FILE = "../cam_params.yaml";

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

class ImageSubscriber : public rclcpp::Node {
    public:
    ImageSubscriber() : Node("image_subscriber") {
        this->declare_parameter<std::string>("topic", "camera_feed");
        topic = this->get_parameter("topic").as_string();

        subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic, 10, std::bind(&ImageSubscriber::image_callback, this, std::placeholders::_1));
        
        if (!load_calibration_params(CALIBRATION_FILE, camera_matrix_, dist_coeffs_)){
        return;
        }

        RCLCPP_INFO(this->get_logger(), "Seeing the topic '%s'", topic.c_str()); 
    }
    private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg){
       /** 
        * Displays the images of the listened topic
        *@param msg: The ROS2 Image msg
        */
        try{
            cv::Mat frame(msg->height, msg->width, CV_8UC3, const_cast<unsigned char*>(msg->data.data()), msg->step);

            cv::Mat corrected_frame;
            cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix_, dist_coeffs_, frame.size(), 1);
            cv::undistort(frame, corrected_frame, camera_matrix_, dist_coeffs_, new_camera_matrix);

            cv::imshow(topic, corrected_frame);
            cv::waitKey(1);
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing image: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    std::string topic;

};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    cv::Mat camera_matrix, dist_coeffs;
    auto node = std::make_shared<ImageSubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}