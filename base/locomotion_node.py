import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist  # <-- NUEVO: Para recibir comandos de velocidad
import math
import time

class LocomotionController(Node):
    def __init__(self):
        super().__init__('locomotion_controller')
        
        self.cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.state_sub = self.create_subscription(JointState, 'joint_states', self.state_callback, 10)
        
        # --- NUEVO SUSCRIPTOR DE VELOCIDAD ---
        self.vel_sub = self.create_subscription(Twist, 'cmd_vel', self.vel_callback, 10)

        self.current_pos = {}
        self.last_pos = {}
        self.last_time = time.time()
        self.start_time = time.time()

        self.kp = 80.0 
        self.kd = 3.0   

        self.estado_actual = 'ESTATICO'
        self.tiempo_espera = 2.0 

        # Parámetros de postura
        # MODIFICACION DE PARAMETROS 
        self.ajuste_altura =  0.0 
        self.inclinacion_pitch = -0.15 
        self.desplazamiento_x = 0.15 

        # --- VARIABLES DE VELOCIDAD DINÁMICA ---
        self.cmd_vx = 0.0  # Velocidad lineal (Adelante/Atrás)
        self.cmd_wz = 0.0  # Velocidad angular (Giro Izquierda/Derecha)

        self.timer = self.create_timer(0.01, self.control_loop)
        self.get_logger().info("Nodo de Locomoción listo. ¡Esperando comandos en /cmd_vel!")

    def state_callback(self, msg):
        for nombre, pos in zip(msg.name, msg.position):
            self.current_pos[nombre] = pos

    def vel_callback(self, msg):
        """Recibe los comandos de velocidad y los guarda"""
        self.cmd_vx = msg.linear.x
        self.cmd_wz = msg.angular.z

    def calcular_trayectoria_pata(self, phi, base_thigh, base_calf, amplitud_zancada):
        phi = phi % (2 * math.pi)
        
        # El negativo convierte el comando positivo en un avance hacia adelante
        amplitud_thigh = -amplitud_zancada 
        q_thigh = base_thigh + amplitud_thigh * math.cos(phi)
        
        # Solo levantamos la rodilla de forma exagerada si el robot se está moviendo.
        # Si la velocidad es casi 0, da pequeños pasitos en el sitio.
        amplitud_lift = 0.55 if abs(amplitud_zancada) > 0.05 else 0.15
        
        if math.sin(phi) < 0:
            q_calf = base_calf + amplitud_lift * math.sin(phi)
        else:
            q_calf = base_calf
            
        return q_thigh, q_calf

    def control_loop(self):
        if not self.current_pos:
            return

        if not hasattr(self, 'conectado'):
            self.get_logger().info("¡De pie! Activa un comando en /cmd_vel para empezar a caminar...")
            self.conectado = True

        tiempo_actual = time.time()
        t = tiempo_actual - self.start_time
        dt = tiempo_actual - self.last_time
        
        if dt < 0.001: 
            return 

        if self.estado_actual == 'ESTATICO' and t >= self.tiempo_espera:
            self.estado_actual = 'MARCHA'

        thigh_nominal = 0.8
        calf_nominal = -1.6
        base_thigh_general = thigh_nominal - self.ajuste_altura
        base_calf = calf_nominal + (self.ajuste_altura * 1.5) 
        base_thigh_front = base_thigh_general + self.inclinacion_pitch + self.desplazamiento_x
        base_thigh_rear = base_thigh_general - self.inclinacion_pitch + self.desplazamiento_x

        target_positions = {}

        if self.estado_actual == 'ESTATICO':
            target_positions = {
                'FL_hip': 0.0, 'FR_hip': 0.0, 'RL_hip': 0.0, 'RR_hip': 0.0,
                'FL_thigh': base_thigh_front, 'FR_thigh': base_thigh_front, 
                'RL_thigh': base_thigh_rear,  'RR_thigh': base_thigh_rear,
                'FL_calf': base_calf,   'FR_calf': base_calf,   
                'RL_calf': base_calf,   'RR_calf': base_calf,
            }
        
        elif self.estado_actual == 'MARCHA':
            frecuencia = 2.0  
            omega = 2.0 * math.pi * frecuencia
            t_marcha = t - self.tiempo_espera
            phi1 = omega * t_marcha
            phi2 = phi1 + math.pi

            target_positions['FL_hip'] = 0.0
            target_positions['FR_hip'] = 0.0
            target_positions['RL_hip'] = 0.0
            target_positions['RR_hip'] = 0.0

            # --- CÁLCULO DE DIFERENCIAL DE VELOCIDAD ---
            factor_lineal = 0.4  # Cuánto se traduce 1.0 m/s a radianes de apertura
            factor_angular = 0.25 # Cuánto se traduce 1.0 rad/s a diferencia entre patas
            
            zancada_base = self.cmd_vx * factor_lineal
            diferencial_giro = self.cmd_wz * factor_angular
            
            # Al girar a la izquierda (+wz), la pata derecha da zancadas más largas
            zancada_izq = zancada_base - diferencial_giro
            zancada_der = zancada_base + diferencial_giro
            
            # Limitamos la zancada máxima para evitar que las piernas colisionen
            zancada_izq = max(min(zancada_izq, 0.6), -0.6)
            zancada_der = max(min(zancada_der, 0.6), -0.6)

            # APLICAMOS A CADA PATA (¡Ojo con los lados!)
            # Izquierda (FL, RL) usa zancada_izq
            target_positions['FL_thigh'], target_positions['FL_calf'] = self.calcular_trayectoria_pata(phi1, base_thigh_front, base_calf, zancada_izq)
            target_positions['RL_thigh'], target_positions['RL_calf'] = self.calcular_trayectoria_pata(phi2, base_thigh_rear, base_calf, zancada_izq)
            
            # Derecha (FR, RR) usa zancada_der
            target_positions['FR_thigh'], target_positions['FR_calf'] = self.calcular_trayectoria_pata(phi2, base_thigh_front, base_calf, zancada_der)
            target_positions['RR_thigh'], target_positions['RR_calf'] = self.calcular_trayectoria_pata(phi1, base_thigh_rear, base_calf, zancada_der)

        # Controlador PD
        msg_cmds = JointState()
        msg_cmds.header.stamp = self.get_clock().now().to_msg()
        
        for motor_name, q_deseado in target_positions.items():
            joint_name = f"{motor_name}_joint"
            if joint_name not in self.current_pos:
                continue

            q_actual = self.current_pos[joint_name]
            q_anterior = self.last_pos.get(joint_name, q_actual)
            velocidad_actual = (q_actual - q_anterior) / dt
            error_pos = q_deseado - q_actual
            torque = (self.kp * error_pos) - (self.kd * velocidad_actual)
            
            msg_cmds.name.append(motor_name)
            msg_cmds.position.append(torque)
            self.last_pos[joint_name] = q_actual

        self.last_time = tiempo_actual
        self.cmd_pub.publish(msg_cmds)

def main(args=None):
    rclpy.init(args=args)
    locomotion_node = LocomotionController()
    rclpy.spin(locomotion_node)
    locomotion_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
