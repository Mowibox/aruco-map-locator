using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosPose2D = RosMessageTypes.Pose2d.Pose2DMsg;

public class BlueRobot : MonoBehaviour
{
    public int robotID = 1;
    public GameObject robotCube;

    void Start(){
        ROSConnection.GetOrCreateInstance().Subscribe<RosPose2D>("robot_pose", PoseCallback);
    }

    void PoseCallback(RosPose2D msg)
    {
        if (msg.marker_id == robotID){
            float posX = msg.x;
            float posY = msg.y;
            float theta = msg.theta;

            robotCube.transform.position = new Vector3(-posX, 1.5f, -posY);
            robotCube.transform.rotation = Quaternion.Euler(0, theta*Mathf.Rad2Deg, 0);
        }
    }

    void OnApplicationQuit(){
        ROSConnection.GetOrCreateInstance().Unsubscribe("robot_pose");
    }
}
