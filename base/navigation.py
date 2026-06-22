import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage
import math
import time

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        # --- PUBLICADOR DE VELOCIDAD ---
        self.vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # --- SUSCRIPTOR DE POSICIÓN GLOBAL (TF) ---
        # Escuchamos todas las transformaciones publicadas (las de tu Capa 1)
        self.tf_sub = self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)

        # --- ESTADO DEL ROBOT ---
        self.x = 1.0
        self.y = 0.5
        self.yaw = 0.0  # Ángulo de orientación (Yaw) en radianes

        # --- LISTA DE WAYPOINTS ---
        # Puntos objetivo (x, y) que el robot debe alcanzar
        self.waypoints = [
            (6.0, 1.0),
            (6.0, 3.5),
            (6.0, 0.5)
        ]
        self.current_target_idx = 0

        # --- PARÁMETROS DE CONTROL ---
        self.distancia_tolerancia = 0.2  # A qué distancia (metros) consideramos que ha "llegado" al punto
        self.max_v = 0.6  # Velocidad lineal máxima (m/s)
        self.max_w = 1.0  # Velocidad angular máxima (rad/s)
        
        # Ganancias Proporcionales (P de PID)
        self.k_rho = 0.8  # Qué tan rápido acelera en función de la distancia
        self.k_alpha = 1.5 # Qué tan rápido gira en función del error angular

        # Bucle de control a 20 Hz
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Nodo Navegador iniciado. Esperando posición inicial...")

    def tf_callback(self, msg):
        """Extrae la posición y rotación (x, y, yaw) del robot desde la transformación 'world' a 'base_link'"""
        for transform in msg.transforms:
            if transform.header.frame_id == 'world' and transform.child_frame_id == 'base_link':
                
                # 1. Posición (x, y)
                self.x = transform.transform.translation.x
                self.y = transform.transform.translation.y
                
                # 2. Orientación (Yaw) desde el Cuaternión
                q = transform.transform.rotation
                # Fórmula para extraer el Yaw de un cuaternión [x, y, z, w]
                siny_cosp = 2 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
                self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def normalizar_angulo(self, angulo):
        """Mantiene el error angular entre -PI y +PI para girar por el camino más corto"""
        while angulo > math.pi:
            angulo -= 2.0 * math.pi
        while angulo < -math.pi:
            angulo += 2.0 * math.pi
        return angulo

    def control_loop(self):
        # Si ya no hay más waypoints, frenar y terminar
        if self.current_target_idx >= len(self.waypoints):
            msg_stop = Twist()
            self.vel_pub.publish(msg_stop)
            self.get_logger().info("¡Ruta completada! Robot detenido.")
            return

        # 1. Obtener el waypoint actual
        target_x, target_y = self.waypoints[self.current_target_idx]

        # 2. Calcular Error de Distancia (rho)
        dx = target_x - self.x
        dy = target_y - self.y
        distancia = math.hypot(dx, dy)

        # 3. Comprobar si hemos llegado
        if distancia < self.distancia_tolerancia:
            self.get_logger().info(f"Waypoint {self.current_target_idx} alcanzado: ({target_x}, {target_y})")
            self.current_target_idx += 1
            return  # Esperamos al siguiente ciclo para apuntar al nuevo target

        # 4. Calcular Error Angular (alpha)
        # Ángulo deseado para mirar directamente al waypoint
        angulo_deseado = math.atan2(dy, dx)
        # Error entre hacia donde miro y hacia donde debo mirar
        error_angular = self.normalizar_angulo(angulo_deseado - self.yaw)

        # 5. CONTROLADOR
        cmd_v = 0.0
        cmd_w = 0.0

        # Estrategia: Si el error angular es muy grande (está mirando para otro lado),
        # primero gira en el sitio, luego avanza.
        umbral_giro = math.pi / 4.0  # 45 grados

        if abs(error_angular) > umbral_giro:
            # Rotación pura (Pivot)
            cmd_v = 0.0
            cmd_w = self.k_alpha * error_angular
        else:
            # Avance y giro simultáneo
            cmd_v = self.k_rho * distancia
            cmd_w = self.k_alpha * error_angular

        # 6. SATURACIÓN (Límites máximos)
        # Evitamos mandar comandos salvajes a la Capa 2
        cmd_v = max(min(cmd_v, self.max_v), -self.max_v)
        cmd_w = max(min(cmd_w, self.max_w), -self.max_w)

        # Opcional: Impedir marchar hacia atrás mientras navega a puntos
        if cmd_v < 0: 
            cmd_v = 0.0

        # 7. PUBLICAR COMANDOS
        msg_vel = Twist()
        msg_vel.linear.x = cmd_v
        msg_vel.angular.z = cmd_w
        self.vel_pub.publish(msg_vel)

def main(args=None):
    rclpy.init(args=args)
    navigator = WaypointNavigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
