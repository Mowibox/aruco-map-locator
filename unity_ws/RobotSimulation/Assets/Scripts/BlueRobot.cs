using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosPose2D = RosMessageTypes.Pose2d.Pose2DMsg;

public class BlueRobot : MonoBehaviour
{
    // ID du robot à suivre (exemple : robot avec ID 1)
    public int robotID = 6;
    public GameObject robotCube;

    void Start()
    {
        // S'abonner à la pose du robot
        ROSConnection.GetOrCreateInstance().Subscribe<RosPose2D>("robot_pose", PoseCallback);
    }

    void PoseCallback(RosPose2D msg)
    {
        // Vérifier que l'ID du robot correspond à celui que l'on suit
        if (msg.marker_id == robotID)
        {
            // Extraire les données de la pose
            float posX = msg.x;
            float posY = msg.y;
            float theta = msg.theta;

            // Déplacer le cube
            robotCube.transform.position = new Vector3(-posX, 1.5f, -posY);
            robotCube.transform.rotation = Quaternion.Euler(0, theta*Mathf.Rad2Deg, 0);
        }
    }

    void OnApplicationQuit()
    {
        // Se désabonner proprement lors de la fermeture de l'application
        ROSConnection.GetOrCreateInstance().Unsubscribe("robot_pose");
    }
}
