import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import tf_transformations
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
import random
import math
import numpy as np

class NoDePosicao(Node):
    def __init__(self):
        super().__init__('posicao')

        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.subscription_odom = self.create_subscription(
            Odometry, '/odom', self.listener_callback_odom, qos_profile)

        self.subscription_laser = self.create_subscription(
            LaserScan, '/scan', self.listener_callback_laser, qos_profile)

        # Publishers
        self.publisher_laser = self.create_publisher(LaserScan, '/laser_data', qos_profile)
        self.publisher_posicao = self.create_publisher(Pose2D, '/posicao', qos_profile)

        self.timer = self.create_timer(0.1, self.update)  # Atualização periódica

        # Variáveis de estado
        self.raio = 0.033
        self.distancia_rodas = 0.178
        self.pose = [0.0, 0.0, 0.0]  # x, y, theta

        # Ruídos
        # Ruídos (valores reduzidos para suavizar o comportamento)
        self.sigma_x = 0.001  # era 0.005
        self.sigma_y = 0.001  # era 0.005
        self.sigma_z = 0.001  # não usado diretamente, mas reduzido também
        self.sigma_th = math.radians(0.005)  # era 0.01 rad
        self.sigma_v = 0.0005  # era 0.001
        self.sigma_w = math.radians(0.005)  # era 0.01 rad

        # Ruído de medição (simulação de GPS)
        self.sigma_z_x = 0.005  # era 0.01
        self.sigma_z_y = 0.005  # era 0.01

        self.v = 0.5
        self.raio = 2
        self.w = self.v / self.raio
        self.dt = 0.1

    def listener_callback_odom(self, msg):
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        _, _, self.yaw = tf_transformations.euler_from_quaternion([x, y, z, w])

        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y

    def listener_callback_laser(self, msg):
        pass  # Você pode preencher isso se quiser processar os dados do LIDAR

    def update(self):

        ksi_x = self.sigma_x * random.gauss(0, 1)
        ksi_y = self.sigma_y * random.gauss(0, 1)
        ksi_th = self.sigma_th * random.gauss(0, 1)
        ksi_v = self.sigma_v * random.gauss(0, 1)
        ksi_w = self.sigma_w * random.gauss(0, 1)
        # Ruído de movimento

        Pv = self.pose.copy()
        Pv[0] = (Pv[0] + ksi_x) + (self.v + ksi_v) * math.cos(Pv[2] + ksi_th) * self.dt
        Pv[1] = (Pv[1] + ksi_y) + (self.v + ksi_v) * math.sin(Pv[2] + ksi_th) * self.dt
        Pv[2] = (Pv[2] + ksi_th) + (self.w + ksi_w) * self.dt
        self.pose = Pv

        # Medida (simulação de GPS)
        C = np.array([[1, 0, 0], [0, 1, 0]])
        R = np.array([[self.sigma_z_x**2, 0], [0, self.sigma_z_y**2]])
        ruido = np.dot(np.sqrt(R), np.random.randn(2, 1)).flatten()
        y = np.dot(C, Pv[:3]) + ruido

        # Estimativa com EKF
        Pe = self.pose.copy()
        Pe[0] += self.v * math.cos(Pe[2]) * self.dt
        Pe[1] += self.v * math.sin(Pe[2]) * self.dt
        Pe[2] += self.w * self.dt

        Q = np.array([[self.sigma_x**2, 0, 0],
                      [0, self.sigma_y**2, 0],
                      [0, 0, self.sigma_th**2]])

        M = np.array([[self.sigma_v**2, 0],
                      [0, self.sigma_w**2]])

        F = np.array([[1, 0, -self.v * math.sin(Pe[2]) * self.dt],
                      [0, 1, self.v * math.cos(Pe[2]) * self.dt],
                      [0, 0, 1]])

        G = np.array([[math.cos(Pe[2]) * self.dt, 0],
                      [math.sin(Pe[2]) * self.dt, 0],
                      [0, self.dt]])

        H = C
        z = np.dot(H, Pe)

        P = Q
        K = np.dot(P, np.dot(H.T, np.linalg.pinv(np.dot(H, np.dot(P, H.T)) + R)))
        Pe = Pe + np.dot(K, (y - z))
        P = np.dot((np.eye(Q.shape[0]) - np.dot(K, H)), P)

        self.pose = Pe
        self.publicar_posicao()

    def publicar_posicao(self):
        msg = Pose2D()
        msg.x = self.pose[0]
        msg.y = self.pose[1]
        msg.theta = self.pose[2]
        self.publisher_posicao.publish(msg)
        self.get_logger().info(f'Publicando pose -> x: {msg.x:.2f}, y: {msg.y:.2f}, theta: {math.degrees(msg.theta):.2f}°')

    def __del__(self):
        self.get_logger().info('Finalizando o nó! Tchau, tchau...')


def main(args=None):
    rclpy.init(args=args)
    node = NoDePosicao()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
