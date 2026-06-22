import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

# --- 1. CONFIGURACIÓN Y RUTAS ---
RUTA_XML = "/home/leo/cuadrupedo_rl/mujoco_menagerie/unitree_go2/scene.xml"
NOMBRE_MODELO = "/home/leo/cuadrupedo_rl/scripts/caso_2_fallos/vibracion/vibracion"
DIRECTORIO_LOGS = "/home/leo/cuadrupedo_rl/scripts/caso_2_fallos/vibracion/logs"

os.makedirs(DIRECTORIO_LOGS, exist_ok=True)

# Obligamos a que la IA empiece de cero con estas reglas físicas estrictas
FORZAR_REENTRENAMIENTO = False 

# --- 2. EL ENTORNO DEFINITIVO (Puro + Propiocepción + Topes Mecánicos) ---
class Go2Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.modelo = mujoco.MjModel.from_xml_path(RUTA_XML)
        self.datos = mujoco.MjData(self.modelo)
        
        # Postura base para que no nazca espachurrado
        """self.postura_nominal = np.array([
            0.1,  0.8, -1.5, # Pata Delantera Derecha (FR)
            -0.1, 0.8, -1.5, # Pata Delantera Izquierda (FL)
            0.1,  1.0, -1.5, # Pata Trasera Derecha (RR)
            -0.1, 1.0, -1.5  # Pata Trasera Izquierda (RL)
        ])"""

        self.postura_nominal = np.array([
            0.11,  0.7, -1.3, # Pata Delantera Derecha (FR)
            -0.11, 0.7, -1.3, # Pata Delantera Izquierda (FL)
            0.11,  0.74, -1.34, # Pata Trasera Derecha (RR)
            -0.1, 0.74, -1.34  # Pata Trasera Izquierda (RL)
        ])

        # Topes mecánicos extraídos del XML oficial (Sim2Real)
        # Orden: [Abduction, Hip, Knee] para cada pata (FR, FL, RR, RL)
        self.limite_inf = np.array([
            -1.0472, -1.5708, -2.7227,  # FR
            -1.0472, -1.5708, -2.7227,  # FL
            -1.0472, -0.5236, -2.7227,  # RR
            -1.0472, -0.5236, -2.7227   # RL
        ])
        
        self.limite_sup = np.array([
            1.0472,  3.4907, -0.83776,  # FR
            1.0472,  3.4907, -0.83776,  # FL
            1.0472,  4.5379, -0.83776,  # RR
            1.0472,  4.5379, -0.83776   # RL
        ])

        # Ganancias del Controlador PD
        self.Kp = 40.0  # q1 60  q2 45  q3 55
        self.Kd = 3.0   # q1 4  q2 3.5  q3 3

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        # Ojos de la IA: 1 (Altura Z) + 3 (Gravedad) + 12 (Ángulos) + 18 (Velocidades) = 34
        obs_dim = 1 + 3 + 12 + 18 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.modelo, self.datos)
        self.datos.qpos[7:19] = self.postura_nominal
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Altura Z (ignoramos X e Y global)
        altura = self.datos.qpos[2:3]
        
        # 2. Vector de Gravedad Proyectada (El "Oído Interno")
        w, x, y, z = self.datos.qpos[3:7]
        gravedad_proyectada = np.array([
            2 * (x*z - w*y),       # Pitch (Inclinación Adelante/Atrás)
            2 * (y*z + w*x),       # Roll  (Inclinación Lados)
            1 - 2 * (x**2 + y**2)  # Yaw/Down (Gravedad vertical)
        ])
        
        # 3. Ángulos de las 12 patas
        angulos_motores = self.datos.qpos[7:19]
        
        # 4. Velocidades (las 6 del torso + las 12 de las patas = 18)
        velocidades = self.datos.qvel
        
        obs = np.concatenate([
            altura, 
            gravedad_proyectada, 
            angulos_motores, 
            velocidades
        ]).astype(np.float32)
        
        return obs

    def step(self, action):
        angulos_actuales = self.datos.qpos[7:19]
        velocidades_actuales = self.datos.qvel[6:18]
        
        # Le damos un buen rango de libertad (0.8 rad) porque ahora tenemos topes seguros
        angulos_deseados = self.postura_nominal + (action * 0.8)
        
        # SIM2REAL SAFETY: Recortamos los ángulos para que no rompan el robot
        angulos_deseados = np.clip(angulos_deseados, self.limite_inf, self.limite_sup)
        
        error_posicion = angulos_deseados - angulos_actuales
        torques = (self.Kp * error_posicion) - (self.Kd * velocidades_actuales)

        self.datos.ctrl[:12] = torques
        mujoco.mj_step(self.modelo, self.datos)
        
        obs = self._get_obs()
        
        # --- REWARD SHAPING ESTRICTO ---
        v_x = self.datos.qvel[0]
        inclinacion_pitch = obs[1] 
        inclinacion_roll = obs[2]  
        
        # 1. Premio por avanzar
        recompensa_avance = max(0, v_x) * 10.0 
        
        # 2. Castigo severo por inclinarse (Te obliga a mantener la espalda recta)
        castigo_inclinacion = 5.0 * (np.square(inclinacion_pitch) + np.square(inclinacion_roll))
        
        # 3. Impuesto a la pereza (Desesperación para obligar al movimiento)
        if v_x > 0.15:
            bono_supervivencia = 0.75  # Cobras si te mueves
        else:
            bono_supervivencia = -3.0 # Te desangras a puntos si te quedas quieto
            
        # 4. Uso de energía (Casi ignorado para fomentar la exploración explosiva)
        costo_energia = 0.001 * np.sum(np.square(action))
        
        reward = recompensa_avance + bono_supervivencia - costo_energia - castigo_inclinacion
        
        # Condición de muerte (Si el torso baja a 0.20m)
        altura_torso = self.datos.qpos[2]
        terminated = bool(altura_torso < 0.20)
        
        if terminated:
            reward -= 10.0
            
        return obs, float(reward), terminated, False, {}

env = Go2Env()

# --- 3. ENTRENAMIENTO MASIVO ---
if os.path.exists(f"{NOMBRE_MODELO}.zip") and not FORZAR_REENTRENAMIENTO:
    print("\n¡Cerebro encontrado! Cargando el modelo entrenado...\n")
    modelo_ppo = PPO.load(NOMBRE_MODELO, env=env)
else:
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO DEFINITIVO (AB INITIO)")
    print("Abre TensorBoard en otra terminal: tensorboard --logdir=" + DIRECTORIO_LOGS)
    print("="*60 + "\n")
    
    modelo_ppo = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        n_steps=4096,      
        batch_size=128, 
        learning_rate=3e-4,
        ent_coef=0.01,     # Entropía para que explore y salga de los óptimos locales
        tensorboard_log=DIRECTORIO_LOGS
    )
    
    # ¡1 millón de pasos para empezar! (Siéntete libre de subirlo a 3 millones si tienes tiempo)
    modelo_ppo.learn(total_timesteps=2000000, tb_log_name="PPO_Go2_Definitivo")
    modelo_ppo.save(NOMBRE_MODELO)
    print("\n¡Entrenamiento masivo completado y guardado!\n")

# --- 4. INFERENCIA VISUAL ---
obs, info = env.reset()

print("="*60)
print("LANZANDO VISUALIZADOR MUJOCO")
print("Cierra la ventana gráfica para detener el script.")
print("="*60)

with mujoco.viewer.launch_passive(env.modelo, env.datos) as viewer:
    while viewer.is_running():
        # Explotación total: determinista=True para que use lo que mejor sabe
        action, _states = modelo_ppo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        viewer.sync()
        time.sleep(env.modelo.opt.timestep)
        
        if terminated or truncated:
            obs, info = env.reset()

