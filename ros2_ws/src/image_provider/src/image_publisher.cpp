#include <mutex>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>

// Frame dimensions
const int FRAME_WIDTH = 320;
const int FRAME_HEIGHT = 240;

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
    ImagePublisher(cv::VideoCapture &capture)
        : Node("image_publisher"), capture_(capture), is_running_(true){
            publisher_ = this->create_publisher<sensor_msgs::msg::Image>("camera_feed", 20);
            timer_ = this->create_wall_timer(std::chrono::duration<double>(0.2), std::bind(&ImagePublisher::publish_frame, this));
            thread_ = std::thread(&ImagePublisher::capture_frames, this);
    }

    ~ImagePublisher() {
        is_running_ = false;
        thread_.join(); 
    }

    private:
    void capture_frames() {
       /** 
        * Captures the frames with threading
        */
        cv::Mat frame;
        while (is_running_){
            if (!capture_.read(frame)) {
                RCLCPP_WARN(this->get_logger(), "Failed to read camera feed.");
                continue;
            } 
        
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_ = frame.clone();
        }
    }

    void publish_frame(){
       /** 
        * Publishes the frames to the 'camera_feed' topic
        */
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (!latest_frame_.empty()) {
            auto msg = mat_to_image_msg(latest_frame_, this->get_clock());
            publisher_->publish(*msg);   
        }

    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture &capture_;
    std::thread thread_;
    std::mutex frame_mutex_;
    cv::Mat latest_frame_;
    bool is_running_;
};


int main(int argc, char **argv){
    rclcpp::init(argc, argv);

    cv::VideoCapture capture(0);
    if (!capture.isOpened()){
        std::cerr << "Failed to open camera." << std::endl;
        return 0;
    }

    capture.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

    auto node = std::make_shared<ImagePublisher>(capture);
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}