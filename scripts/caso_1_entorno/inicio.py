import time
import mujoco
import mujoco.viewer

# --- 1. RUTA DEL ROBOT
RUTA_XML = "/home/leo/robocity_cuadruped_robot/mujoco_menagerie/unitree_go2/scene.xml"
#RUTA_XML = "/home/leo/cuadrupedo_rl/mujoco_menagerie/unitree_g1/scene.xml"

print(f"Cargando el entorno físico desde: {RUTA_XML}")

# --- 2. CARGAR EL MODELO Y LOS DATOS DE FORMA DINÁMICA ---
# MuJoCo lee el XML y dimensiona automáticamente todas las matrices (qpos, qvel, ctrl)
modelo = mujoco.MjModel.from_xml_path(RUTA_XML)
datos = mujoco.MjData(modelo)

# Reiniciamos la física a su estado base (el que venga por defecto en el XML)
mujoco.mj_resetData(modelo, datos)
mujoco.mj_forward(modelo, datos)

# --- 3. INFORMACIÓN DEL ROBOT ---
print("\n--- INFO DEL MODELO ---")
print(f"Número de articulaciones (grados de libertad): {modelo.nq}")
print(f"Número de actuadores (motores a controlar): {modelo.nu}")
print("-----------------------\n")

# --- 4. VISUALIZADOR ---
print("¡Entorno cargado! La gravedad está haciendo su trabajo.")
print("Cierra la ventana gráfica para terminar el script.")

with mujoco.viewer.launch_passive(modelo, datos) as viewer:
    while viewer.is_running():
        # Avanzamos la física un instante sin aplicarle ninguna fuerza a los motores
        mujoco.mj_step(modelo, datos)
        
        # Sincronizamos la pantalla
        viewer.sync()
        
        # Pausa para que la simulación corra a tiempo real
        time.sleep(modelo.opt.timestep)
