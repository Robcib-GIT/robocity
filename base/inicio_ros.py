import time
import threading
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped

class MujocoRosBridge(Node):
    def __init__(self, modelo, datos):
        super().__init__('mujoco_ros_bridge')
        self.modelo = modelo
        self.datos = datos

        # --- PUBLICADORES ---
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, 'robot_pose', 10)

        # --- SUSCRIPTORES ---
        # Escucha comandos en este tópico. (Usamos JointState por simplicidad para pasar nombres y posiciones/torques)
        self.command_sub = self.create_subscription(JointState, 'joint_commands', self.command_callback, 10)

        # --- MAPEO DE ARTICULACIONES (Para LEER qpos) ---
        self.joint_names = []
        self.qpos_indices = []
        for i in range(self.modelo.njnt):
            if self.modelo.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                name = mujoco.mj_id2name(self.modelo, mujoco.mjtObj.mjOBJ_JOINT, i)
                if name:
                    self.joint_names.append(name)
                    self.qpos_indices.append(self.modelo.jnt_qposadr[i])

        # --- MAPEO DE ACTUADORES (Para ESCRIBIR ctrl) ---
        # Mapeamos el nombre del actuador a su índice en el array datos.ctrl
        self.actuator_indices = {}
        for i in range(self.modelo.nu):
            name = mujoco.mj_id2name(self.modelo, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.actuator_indices[name] = i
            else:
                self.get_logger().warn(f"Actuador {i} no tiene nombre en el XML. Será ignorado en comandos.")

        # Buffer interno para los comandos a enviar a MuJoCo
        self.comandos_actuales = [0.0] * self.modelo.nu

        self.get_logger().info("Nodo MujocoRosBridge inicializado. Listo para recibir comandos.")

    def command_callback(self, msg):
        """
        Callback que se ejecuta cuando ROS 2 recibe un mensaje en /joint_commands.
        Actualiza el buffer interno de comandos. No interactúa directamente con MuJoCo aquí
        para evitar problemas de hilos (thread-safety).
        """
        # Se asume que msg.name coincide con el nombre de los actuadores (motors/positioners) en el XML
        # y que msg.position contiene el valor a aplicar (posición o torque, según el XML).
        for nombre, comando in zip(msg.name, msg.position):
            if nombre in self.actuator_indices:
                idx = self.actuator_indices[nombre]
                self.comandos_actuales[idx] = comando

    def apply_commands_to_mujoco(self):
        """Aplica los valores del buffer interno a los actuadores físicos de MuJoCo."""
        for i in range(self.modelo.nu):
            self.datos.ctrl[i] = self.comandos_actuales[i]

    def publish_state(self):
        """Lee el estado de MuJoCo y lo publica en ROS 2."""
        tiempo_actual = self.get_clock().now().to_msg()

        # 1. Publicar JointStates
        msg_joints = JointState()
        msg_joints.header.stamp = tiempo_actual
        msg_joints.name = self.joint_names
        msg_joints.position = [float(self.datos.qpos[idx]) for idx in self.qpos_indices]
        self.joint_pub.publish(msg_joints)

        # 2. Publicar TF del free joint (si el robot tiene uno y no está atornillado al suelo)
        if self.modelo.nq >= 7 and self.modelo.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            t = TransformStamped()
            t.header.stamp = tiempo_actual
            t.header.frame_id = 'world'
            t.child_frame_id = 'base_link'

            # Traslación (qpos[0, 1, 2])
            t.transform.translation.x = float(self.datos.qpos[0])
            t.transform.translation.y = float(self.datos.qpos[1])
            t.transform.translation.z = float(self.datos.qpos[2])
            
            # Rotación: MuJoCo [w, x, y, z] -> ROS 2 [x, y, z, w]
            t.transform.rotation.x = float(self.datos.qpos[4])
            t.transform.rotation.y = float(self.datos.qpos[5])
            t.transform.rotation.z = float(self.datos.qpos[6])
            t.transform.rotation.w = float(self.datos.qpos[3])

            self.tf_broadcaster.sendTransform(t)
            
            # --- 3. NUEVO: Publicar PoseStamped en el tópico /robot_pose ---
            msg_pose = PoseStamped()
            msg_pose.header.stamp = tiempo_actual
            msg_pose.header.frame_id = 'world' # La pose está referenciada al mundo

            # Copiamos los valores de traslación
            msg_pose.pose.position.x = t.transform.translation.x
            msg_pose.pose.position.y = t.transform.translation.y
            msg_pose.pose.position.z = t.transform.translation.z

            # Copiamos los valores de rotación (ya convertidos a x,y,z,w)
            msg_pose.pose.orientation.x = t.transform.rotation.x
            msg_pose.pose.orientation.y = t.transform.rotation.y
            msg_pose.pose.orientation.z = t.transform.rotation.z
            msg_pose.pose.orientation.w = t.transform.rotation.w

            # Publicamos el mensaje
            self.pose_pub.publish(msg_pose)


def main(args=None):
    # --- 1. CONFIGURACIÓN DE MUJOCO ---
    RUTA_XML = "/home/leo/robocity_cuadruped_robot/mujoco_menagerie/unitree_go2/scene.xml"
    #RUTA_XML = "/home/leo/robocity_cuadruped_robot/mujoco_menagerie/unitree_go2/scene_terreno_irregular.xml"
    #RUTA_XML = "/home/leo/cuadrupedo_sim/mujoco_menagerie/unitree_go2/scene_terreno_irregular.xml"
    print(f"Cargando el entorno físico desde: {RUTA_XML}")

    modelo = mujoco.MjModel.from_xml_path(RUTA_XML)
    datos = mujoco.MjData(modelo)

    mujoco.mj_resetData(modelo, datos)
    
    # --- NUEVO: MOVER EL ROBOT A LA CASILLA DE SALIDA (0.5, 0.5) ---
    # PARAMETROS A MODIFICAR
    datos.qpos[0] = 1.0  # Posición X inicial
    datos.qpos[1] = 0.5  # Posición Y inicial
    
    mujoco.mj_forward(modelo, datos)

    # --- 2. INICIALIZAR ROS 2 ---
    rclpy.init(args=args)
    ros_bridge = MujocoRosBridge(modelo, datos)

    # --- 3. MULTI-THREADING PARA ROS 2 ---
    # Creamos un hilo separado para que ROS 2 procese sus callbacks (como command_callback)
    # sin bloquear ni ralentizar la simulación física. daemon=True asegura que el hilo muera al cerrar el script.
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_bridge,), daemon=True)
    ros_thread.start()

    print("¡Entorno cargado y simulación en marcha!")

    # --- 4. BUCLE PRINCIPAL (Física y Visualización) ---
    with mujoco.viewer.launch_passive(modelo, datos) as viewer:
        while viewer.is_running() and rclpy.ok():
            # 1. Aplicar los comandos recibidos desde ROS 2
            ros_bridge.apply_commands_to_mujoco()

            # 2. Avanzar un paso de física
            mujoco.mj_step(modelo, datos)
            
            # 3. Publicar el nuevo estado a ROS 2
            ros_bridge.publish_state()

            # 4. Actualizar el visualizador
            viewer.sync()
            
            # Mantener el ritmo de la simulación en tiempo real (aprox)
            time.sleep(modelo.opt.timestep)

    # Limpieza al salir
    ros_bridge.destroy_node()
    rclpy.shutdown()
    ros_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()

