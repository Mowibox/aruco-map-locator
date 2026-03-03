#nullable enable
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosPose2D = RosMessageTypes.Pose2d.Pose2DMsg;

public class RobotController : MonoBehaviour
{
    private const float ROBOT_HEIGHT_Z = 1.5f;

    [Header("Robot Settings")]
    [SerializeField] private int robotID = 1;
    [SerializeField] private GameObject? robotCube;

    [Header("Real world map dimensions (meters)")]
    [SerializeField] private float worldWidth  = 0.30f;
    [SerializeField] private float worldHeight = 0.20f;

    [Header("Unity map object dimensions (Unity units)")]
    [SerializeField] private float unityMapWidth  = 30.0f;
    [SerializeField] private float unityMapHeight = 20.0f;

    private float scaleX;
    private float scaleY;

    void Start()
    {
        scaleX = unityMapWidth  / worldWidth;
        scaleY = unityMapHeight / worldHeight;
        ROSConnection.GetOrCreateInstance().Subscribe<RosPose2D>("robot_pose", PoseCallback);
    }

    void PoseCallback(RosPose2D msg)
    {
        if (msg.marker_id != robotID || robotCube == null) return;

        robotCube.transform.SetPositionAndRotation(
            new Vector3(-msg.x * scaleX, ROBOT_HEIGHT_Z, -msg.y * scaleY),
            Quaternion.Euler(0, msg.theta * Mathf.Rad2Deg, 0)
        );
    }

    void OnApplicationQuit()
    {
        ROSConnection.GetOrCreateInstance().Unsubscribe("robot_pose");
    }
}