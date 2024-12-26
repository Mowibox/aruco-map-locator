#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>

class ImageSubscriber : public rclcpp::Node {
    public:
    ImageSubscriber() : Node("image_subscriber") {
        this->declare_parameter<std::string>("topic", "camera_feed");
        std::string topic = this->get_parameter("topic").as_string();

        subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic, 10, std::bind(&ImageSubscriber::image_callback, this, std::placeholders::_1));

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
            cv::imshow("Video", frame);
            cv::waitKey(1);
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing image: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;

};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImageSubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}