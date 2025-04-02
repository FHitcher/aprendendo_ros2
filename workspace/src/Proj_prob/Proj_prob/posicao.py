import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Header, Float64
import random
import math

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
        self.timer = self.create_timer(1, self.update)  # Atualização a cada 0.1 segundos

        # Variáveis de estado
        self.jointL = 0.0
        self.jointR = 0.0

        self.raio = 0.033
        self.distancia_rodas = 0.178
        self.pose = [0.0, 0.0, 0.0]  # x, y, theta
        self.medidas = [0.0, 0.0]  # esq, dir
        self.ultimas_medidas = [0.0, 0.0]  # esq, dir
        self.distancias = [0.0, 0.0]

        # Mapa
        self.estado_inicial = 0 #pq diabos isso estava em 4 n sei mas agr ta tudo funcionando
        self.mapa = [1, 3, 5]  # posição central das “portas”
        self.pose[0] = self.estado_inicial
        self.sigma_odometria = 0.2
        self.sigma_lidar = 0.175
        self.sigma_movimento = 0.002
        self.porta = 0

    def listener_callback_joint(self, msg):
        self.jointL = msg.position[0]
        self.jointR = msg.position[1]

    def listener_callback_laser(self, msg):
        self.laser72 = msg.ranges[90]

    def update(self):

        if self.ultimas_medidas[0] is None or self.ultimas_medidas[1] is None:
            self.ultimas_medidas[0] = self.jointL
            self.ultimas_medidas[1] = self.jointR
            return
        
        self.medidas[0] = self.jointL
        self.medidas[1] = self.jointR
        
        diff_left = self.medidas[0] - self.ultimas_medidas[0]
        self.distancias[0] = diff_left * self.raio + random.gauss(0, 0.002)
        self.ultimas_medidas[0] = self.medidas[0]
        
        diff_right = self.medidas[1] - self.ultimas_medidas[1]
        self.distancias[1] = diff_right * self.raio + random.gauss(0, 0.002)
        self.ultimas_medidas[1] = self.medidas[1]
        
        # Cálculo da distância linear e angular percorrida
        deltaS = (self.distancias[0] + self.distancias[1]) / 2.0
        deltaTheta = (self.distancias[1] - self.distancias[0]) / self.distancia_rodas
        
        self.pose[2] = (self.pose[2] + deltaTheta) % (2 * math.pi)
        self.pose[0] += deltaS * math.cos(self.pose[2])
        self.pose[1] += deltaS * math.sin(self.pose[2])

        if self.laser72 == float('inf'):
            media_nova = (self.mapa[self.porta] * self.sigma_movimento + self.pose[0] * self.sigma_lidar) / (self.sigma_movimento + self.sigma_lidar)
            sigma_novo = 1 / (1/self.sigma_movimento + 1/self.sigma_lidar)
            self.pose[0] = media_nova
            self.sigma_movimento = sigma_novo
            
            self.porta = (self.porta + 1) % len(self.mapa)
        
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