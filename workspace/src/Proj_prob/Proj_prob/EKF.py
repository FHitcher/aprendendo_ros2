import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import tf_transformations
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Header, Float64
import random
import math
import numpy as np


class NoDePosicao(Node):
    def __init__(self):
        super().__init__('posicao')

        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.subscription_joint = self.create_subscription(
            JointState, '/joint_states', self.listener_callback_joint, qos_profile)

        self.subscription_laser = self.create_subscription(
            LaserScan, '/scan', self.listener_callback_laser, qos_profile)

        # Publishers
        self.publisher_joint = self.create_publisher(JointState, '/joint_state', qos_profile)
        self.publisher_laser = self.create_publisher(LaserScan, '/laser_data', qos_profile)
        self.publisher_posicao = self.create_publisher(Float64, '/posicao', qos_profile)
        self.timer = self.create_timer(1, self.update)  # Atualização a cada 1 segundos

        # Variáveis de estado
        self.jointL = 0.0
        self.jointR = 0.0

        self.raio = 0.033
        self.distancia_rodas = 0.178
        self.pose = [0.0, 0.0, 0.0]  # x, y, theta
        self.medidas = [0.0, 0.0]  # esq, dir
        self.ultimas_medidas = [0.0, 0.0]  # esq, dir
        self.distancias = [0.0, 0.0]


        #Ruidos aleatorios que corrompem a posição verdadeira do robo real
        
        self.sigma_x = 0.05
        self.sigma_y = 0.05
        self.sigma_z = 0.05
        self.sigma_th = math.radians(0.1)
        self.sigma_v = 0.01
        self.sigma_w = math.radians(0.1)


        self.ksi_x = self.sigma_x * random.normalvariate()
        self.ksi_y = self.sigma_y * random.normalvariate()
        self.ksi_z = self.sigma_z * random.normalvariate()
        self.ksi_th = self.sigma_th * random.normalvariate()
        self.ksi_v = self.sigma_v * random.normalvariate()
        self.ksi_w = self.sigma_w * random.normalvariate()


        self.sigma_z_x = 0.1
        self.sigma_z_y = 0.1
        
        # Mapa
        self.estado_inicial = 0 #pq diabos isso estava em 4 n sei mas agr ta tudo funcionando
        self.pose[0] = self.estado_inicial
        self.sigma_odometria = 0.2
        self.sigma_lidar = 0.175
        self.sigma_movimento = 0.002
        self.porta = 0

    def listener_callback_joint(self, msg):
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        _,_,self.yaw = tf_transformations.euler_from_quaternion([x,y,z,w])

    def listener_callback_pose(self, msg):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y

    def update(self):
        ##Definindo a posição real do robo
        Pv = self.pose.copy()
        
        # Atualização de posição com ruído
        Pv[0] = (Pv[0] + self.ksi_x) + (self.v + self.ksi_v) * math.cos(Pv[2] + self.ksi_th) * self.dt
        Pv[1] = (Pv[1] + self.ksi_y) + (self.v + self.ksi_v) * math.sin(Pv[2] + self.ksi_th) * self.dt
        Pv[2] = (Pv[2] + self.ksi_th) + (self.w + self.ksi_w) * self.dt
        
        self.pose = Pv
        
        # Sensor GPS
        C = np.array([[1, 0, 0], [0, 1, 0]])
        R = np.array([[self.sigma_z_x**2, 0], [0, self.sigma_z_y**2]])
        ruido = np.dot(np.sqrt(R), np.random.randn(2, 1)).flatten()
        y = np.dot(C, Pv[:3]) + ruido

        Pv[0] = (Pv[0] + self.ksi_x) + (self.v + self.ksi_v) * math.cos(Pv[2] + self.ksi_th) * self.dt
        Pv[1] = (Pv[1] + self.ksi_y) + (self.v + self.ksi_v) * math.sin(Pv[2] + self.ksi_th) * self.dt
        
        #estiamndo a posição do robo



        # Estimativa de posição e matriz de covariância
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

        # Ganho de Kalman
        P = Q
        K = np.dot(P, np.dot(H.T, np.linalg.pinv(np.dot(H, np.dot(P, H.T)) + R)))
        Pe = Pe + np.dot(K, (y - z))
        P = np.dot((np.eye(Q.shape[0]) - np.dot(K, H)), P)
        

        self.publicar_posicao()

    def publicar_posicao(self):
        msg = Float64()
        msg.data = self.pose[0]
        self.publisher_posicao.publish(msg)
        self.get_logger().info(f'Publicando posição: {msg.data}')

    def __del__(self):
        self.get_logger().info('Finalizando o nó! Tchau, tchau...')

# Main function
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