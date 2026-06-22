import os
import sys
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

# --- 1. CONFIGURACIÓN Y RUTAS (CASO FALLO 1: CAE ADELANTE) ---
RUTA_XML = "/home/leo/robocity_cuadruped_robot/mujoco_menagerie/unitree_go2/scene.xml"

# Rutas para este caso de fallo específico
BASE_DIR = "/home/leo/robocity_cuadruped_robot/scripts/caso_2_fallos/cae_adelante"
NOMBRE_MODELO = f"{BASE_DIR}/pesos/modelo_cae_adelante"
DIRECTORIO_LOGS = f"{BASE_DIR}/logs/"

os.makedirs(os.path.dirname(NOMBRE_MODELO), exist_ok=True)
os.makedirs(DIRECTORIO_LOGS, exist_ok=True)

FORZAR_REENTRENAMIENTO = False 

# --- 2. EL ENTORNO (El prototipo "Ciego" y sin topes) ---
class Go2Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.modelo = mujoco.MjModel.from_xml_path(RUTA_XML)
        self.datos = mujoco.MjData(self.modelo)
        
        self.postura_nominal = np.array([
            0.1,  0.8, -1.5, # FR
            -0.1, 0.8, -1.5, # FL
            0.1,  1.0, -1.5, # RR
            -0.1, 1.0, -1.5  # RL
        ])

        self.Kp = 50.0  
        self.Kd = 1.0   

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        # EL FALLO ESTÁ AQUÍ: Solo considerábamos nq (19) + nv (18) = 37 observaciones crudas.
        # No hay vector de gravedad, no sabe que se está inclinando.
        obs_dim = self.modelo.nq + self.modelo.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.modelo, self.datos)
        self.datos.qpos[7:19] = self.postura_nominal
        return self._get_obs(), {}

    def _get_obs(self):
        # Observación ingenua: le damos todo en crudo
        return np.concatenate([self.datos.qpos, self.datos.qvel]).astype(np.float32)

    def step(self, action):
        angulos_actuales = self.datos.qpos[7:19]
        velocidades_actuales = self.datos.qvel[6:18]
        
        # No hay np.clip (topes mecánicos), ni decimation (salto de frames)
        angulos_deseados = self.postura_nominal + (action * 0.5)
        
        error_posicion = angulos_deseados - angulos_actuales
        torques = (self.Kp * error_posicion) - (self.Kd * velocidades_actuales)

        self.datos.ctrl[:12] = torques
        mujoco.mj_step(self.modelo, self.datos)
        
        obs = self._get_obs()
        
        # --- REWARD SHAPING INGENUO ---
        v_x = self.datos.qvel[0] # Velocidad Adelante/Atrás
        v_y = self.datos.qvel[1] # Velocidad Lateral 
        v_z = self.datos.qvel[2] # Velocidad Vertical 
        
        # Premia ir hacia adelante infinitamente (provoca el piscinazo)
        recompensa_avance = max(0, v_x) * 10.0 
        
        # Bono por sobrevivir (fomenta quedarse quieto)
        bono_supervivencia = 1.0
        
        castigo_lateral = 3.0 * abs(v_y)
        castigo_vertical = 1.0 * abs(v_z)
        costo_energia = 0.05 * np.sum(np.square(action))
        
        # Como no hay "castigo_inclinacion", la IA no ve problema en tirarse de boca
        reward = recompensa_avance + bono_supervivencia - castigo_lateral - castigo_vertical - costo_energia
        
        altura_torso = self.datos.qpos[2]
        terminated = bool(altura_torso < 0.22)
        
        if terminated:
            reward -= 10.0
            
        return obs, float(reward), terminated, False, {}

env = Go2Env()

# --- 3. ENTRENAMIENTO / INFERENCIA ---
ruta_zip = f"{NOMBRE_MODELO}.zip"

if os.path.exists(ruta_zip) and not FORZAR_REENTRENAMIENTO:
    print("\n" + "="*50)
    print(f"✅ Cargando el modelo 'Cae Adelante' desde: {ruta_zip}")
    print("="*50 + "\n")
    modelo_ppo = PPO.load(NOMBRE_MODELO, env=env)
    
else:
    print("\n" + "="*50)
    print("INICIANDO ENTRENAMIENTO: CASO FALLO 'CAE ADELANTE'")
    print("Vamos a ver cómo la IA aprende a tirarse de boca a propósito.")
    print("="*50 + "\n")
    
    modelo_ppo = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        n_steps=4096,      
        batch_size=128, 
        learning_rate=3e-4,
        tensorboard_log=DIRECTORIO_LOGS
    )
    
    # Con 500k o 1M de pasos es suficiente para que descubra la trampa
    modelo_ppo.learn(total_timesteps=400000, tb_log_name="PPO_Fallo_CaeAdelante")
    modelo_ppo.save(NOMBRE_MODELO)
    print("\n¡Entrenamiento completado y guardado!\n")

# --- 4. SHOWTIME (Visualización del fallo) ---
obs, info = env.reset()

print("="*50)
print("MOSTRANDO EL ÓPTIMO LOCAL (Piscinazo)")
print("Cierra la ventana gráfica para detener.")
print("="*50)

with mujoco.viewer.launch_passive(env.modelo, env.datos) as viewer:
    while viewer.is_running():
        action, _states = modelo_ppo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        viewer.sync()
        time.sleep(env.modelo.opt.timestep)
        
        if terminated or truncated:
            # Añadimos un pequeño print para que veas en terminal cómo se reinicia constantemente
            print("💀 ¡El robot se ha caído! Reiniciando episodio...")
            obs, info = env.reset()
