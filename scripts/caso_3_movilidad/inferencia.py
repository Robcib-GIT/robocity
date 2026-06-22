import os
import sys
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

tipo_terreno = 1 # 0 plano, 1 para curvo, 2 para picos,

# --- 1. CONFIGURACIÓN Y RUTAS ---
RUTA_XML = "/home/leo/robocity_cuadruped_robot/mujoco_menagerie/unitree_go2/scene.xml"
#RUTA_XML = "/home/leo/robocity_cuadruped_robot/mujoco_menagerie/unitree_go2/scene_RL_irregular.xml"
#RUTA_XML = "/home/leo/robocity_cuadruped_robot/mujoco_menagerie/anybotics_anymal_c/scene.xml"
NOMBRE_MODELO = "/home/leo/robocity_cuadruped_robot/scripts/caso_3_movilidad/pesos/modelo_go2_definitivo"


# Verificamos que el cerebro existe antes de arrancar
if not os.path.exists(f"{NOMBRE_MODELO}.zip"):
    print(f"No se encontró el cerebro en {NOMBRE_MODELO}.zip")
    print("Asegúrate de haber terminado el entrenamiento y de que la ruta es correcta.")
    sys.exit(1)

# --- 2. EL ENTORNO EXACTO (Clon del entrenamiento) ---
class Go2Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.modelo = mujoco.MjModel.from_xml_path(RUTA_XML)

        # --- COMENTAR/DESCOMENTAR PARA UTILIZAR TERRENO IRREGULAR ---

        if tipo_terreno == 2:
            # Rellenamos los datos del terreno con valores aleatorios (-0.03m a +0.03m)
            # Esto crea irregularidades de hasta 3 cm. Puedes aumentar este valor si quieres más dificultad.
            ruido_terreno = np.random.uniform(-0.02, 0.02, self.modelo.nhfielddata)
            self.modelo.hfield_data[:] = ruido_terreno
        # ----------------------------------------------------------
        # --- NUEVO: Generación de terreno sinusoidal / ondulado ---
        if tipo_terreno == 1:
            
            # Obtenemos la resolución de la cuadrícula (asumiendo que es cuadrada)
            lado = int(np.sqrt(self.modelo.nhfielddata))
            
            # Generamos las coordenadas espaciales X e Y
            x = np.linspace(0, 15, lado)
            y = np.linspace(0, 15, lado)
            X, Y = np.meshgrid(x, y)
            
            # Combinamos varias ondas matemáticas para crear un terreno desestructurado
            Z = 0.04 * np.sin(2.0 * X) + 0.03 * np.cos(1.5 * Y) + 0.02 * np.sin(4.0 * X + 3.0 * Y)
            
            # Los parámetros del XML (size="... 0.1 0.1") dictan que el terreno real va de -0.1m a +0.1m.
            # MuJoCo espera que los datos del hfield estén estrictamente entre 0 y 1.
            Z_min = -0.1
            Z_max = 0.1
            
            # Mapeamos nuestras alturas (Z) al rango [0, 1]
            hfield_normalizado = (Z - Z_min) / (Z_max - Z_min)
            
            # Aseguramos por seguridad que ningún valor se salga de [0, 1]
            hfield_normalizado = np.clip(hfield_normalizado, 0.0, 1.0)
            
            # Sobrescribimos la memoria del hfield en MuJoCo
            self.modelo.hfield_data[:] = hfield_normalizado.flatten()
        # ----------------------------------------------------------

        self.datos = mujoco.MjData(self.modelo)

        self.postura_nominal = np.array([
            0.11,   0.7,  -1.3,   # FR
            -0.11,  0.7,  -1.3,   # FL
            0.11,   0.74, -1.34,  # RR
            -0.1,   0.74, -1.34   # RL
        ])

        self.limite_inf = np.array([
            -1.0472, -1.5708, -2.7227,  
            -1.0472, -1.5708, -2.7227,  
            -1.0472, -0.5236, -2.7227,  
            -1.0472, -0.5236, -2.7227   
        ])
        
        self.limite_sup = np.array([
            1.0472,  3.4907, -0.83776,  
            1.0472,  3.4907, -0.83776,  
            1.0472,  4.5379, -0.83776,  
            1.0472,  4.5379, -0.83776   
        ])

        self.Kp = 40.0  
        self.Kd = 3.0   
        self.decimation = 10 

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        obs_dim = 1 + 3 + 12 + 18 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    # inicio del robot en altura
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.modelo, self.datos)
        self.datos.qpos[2] = 0.6 # PARAMETRO DE MODIFICACION DE ALTURA DE INICIO 
        self.datos.qpos[7:19] = self.postura_nominal
        return self._get_obs(), {}
    

    def _get_obs(self):
        altura = self.datos.qpos[2:3]
        w, x, y, z = self.datos.qpos[3:7]
        gravedad_proyectada = np.array([
            2 * (x*z - w*y),       
            2 * (y*z + w*x),       
            1 - 2 * (x**2 + y**2)  
        ])
        angulos_motores = self.datos.qpos[7:19]
        velocidades = self.datos.qvel
        return np.concatenate([altura, gravedad_proyectada, angulos_motores, velocidades]).astype(np.float32)

    def step(self, action):
        angulos_deseados = self.postura_nominal + (action * 0.8)
        angulos_deseados = np.clip(angulos_deseados, self.limite_inf, self.limite_sup)
        
        for _ in range(self.decimation):
            angulos_actuales = self.datos.qpos[7:19]
            velocidades_actuales = self.datos.qvel[6:18]
            
            error_posicion = angulos_deseados - angulos_actuales
            torques = (self.Kp * error_posicion) - (self.Kd * velocidades_actuales)

            self.datos.ctrl[:12] = torques
            mujoco.mj_step(self.modelo, self.datos)
        
        obs = self._get_obs()
        
        # Calculamos la recompensa aunque no aprenda de ella (útil para logs si quieres)
        v_x = self.datos.qvel[0]
        inclinacion_pitch = obs[1] 
        inclinacion_roll = obs[2]  
        
        recompensa_avance = np.clip(v_x, 0.0, 0.5) * 20.0
        castigo_inclinacion = 5.0 * (np.square(inclinacion_pitch) + np.square(inclinacion_roll))
        
        if v_x > 0.15:
            bono_supervivencia = 0.75  
        else:
            bono_supervivencia = -3.0 
            
        velocidades_motores = self.datos.qvel[6:18]
        castigo_vibro = 0.01 * np.sum(np.square(velocidades_motores))
        costo_energia = 0.001 * np.sum(np.square(action))
        
        reward = recompensa_avance + bono_supervivencia - costo_energia - castigo_inclinacion - castigo_vibro
        
        altura_torso = self.datos.qpos[2]
        terminated = bool(altura_torso < 0.20)
            
        return obs, float(reward), terminated, False, {}

env = Go2Env()

# --- 3. CARGAR EL CEREBRO ---
print(f" Cargando el cerebro entrenado desde: {NOMBRE_MODELO}.zip...")
modelo_ppo = PPO.load(NOMBRE_MODELO, env=env)

# --- 4. SHOWTIME (Inferencia Visual) ---
obs, info = env.reset()

print("\n" + "="*50)
print("¡INFERENCIA PURA EN MARCHA!")
print("El robot ejecuta su política determinista sin explorar.")
print("Cierra la ventana de MuJoCo para salir.")
print("="*50 + "\n")

with mujoco.viewer.launch_passive(env.modelo, env.datos) as viewer:
    while viewer.is_running():
        # deterministic=True obliga a la IA a usar siempre la mejor acción que aprendió
        action, _states = modelo_ppo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        viewer.sync()
        time.sleep(env.modelo.opt.timestep * env.decimation)
        
        if terminated or truncated:
            obs, info = env.reset()
