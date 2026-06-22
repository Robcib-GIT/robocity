import os
import sys
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

# --- 1. CONFIGURACIÓN Y RUTAS ---
RUTA_XML = "/home/leo/cuadrupedo_rl/mujoco_menagerie/unitree_go2/scene.xml"

# Rutas de tu modelo robusto (CPG + RL)
NOMBRE_MODELO = "/home/leo/cuadrupedo_rl/scripts/caso_4_robustez/pesos/modelo_go2_robusto"
DIRECTORIO_LOGS = "/home/leo/cuadrupedo_rl/scripts/caso_4_robustez/logs/"

# ¿Cuántos pasos MÁS quieres entrenarlo ahora mismo?
PASOS_EXTRA = 200000 

if not os.path.exists(f"{NOMBRE_MODELO}.zip"):
    print(f"❌ ERROR: No encuentro el modelo base en {NOMBRE_MODELO}.zip")
    sys.exit(1)

# --- 2. EL ENTORNO (Debe ser idéntico al que usaste para crear el modelo original) ---
class Go2Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.modelo = mujoco.MjModel.from_xml_path(RUTA_XML)
        self.datos = mujoco.MjData(self.modelo)
        
        self.postura_nominal = np.array([
            0.1,  0.8, -1.5,
            -0.1, 0.8, -1.5,
            0.1,  1.0, -1.5,
            -0.1, 1.0, -1.5
        ])

        self.Kp = 50.0  
        self.Kd = 1.0   
        self.tiempo_simulacion = 0.0 

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        obs_dim = self.modelo.nq + self.modelo.nv + 1 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.modelo, self.datos)
        self.datos.qpos[7:19] = self.postura_nominal
        self.tiempo_simulacion = 0.0 
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.concatenate([self.datos.qpos, self.datos.qvel, [self.tiempo_simulacion]])
        return obs.astype(np.float32)

    def step(self, action):
        self.tiempo_simulacion += self.modelo.opt.timestep
        
        frecuencia = 5.0 
        fase_cadera_1 = np.sin(2 * np.pi * frecuencia * self.tiempo_simulacion)
        fase_cadera_2 = np.sin(2 * np.pi * frecuencia * self.tiempo_simulacion + np.pi)
        fase_rodilla_1 = np.cos(2 * np.pi * frecuencia * self.tiempo_simulacion)
        fase_rodilla_2 = np.cos(2 * np.pi * frecuencia * self.tiempo_simulacion + np.pi)
        
        oscilacion = np.zeros(12)
        amp_cadera = 0.4
        amp_rodilla = 0.4 
        
        oscilacion[1] = fase_cadera_1 * amp_cadera    
        oscilacion[10] = fase_cadera_1 * amp_cadera   
        oscilacion[2] = fase_rodilla_1 * amp_rodilla  
        oscilacion[11] = fase_rodilla_1 * amp_rodilla 
        
        oscilacion[4] = fase_cadera_2 * amp_cadera    
        oscilacion[7] = fase_cadera_2 * amp_cadera    
        oscilacion[5] = fase_rodilla_2 * amp_rodilla  
        oscilacion[8] = fase_rodilla_2 * amp_rodilla  

        angulos_actuales = self.datos.qpos[7:19]
        velocidades_actuales = self.datos.qvel[6:18]
        
        angulos_deseados = self.postura_nominal + oscilacion + (action * 0.3)
        
        error_posicion = angulos_deseados - angulos_actuales
        torques = (self.Kp * error_posicion) - (self.Kd * velocidades_actuales)

        self.datos.ctrl[:12] = torques
        mujoco.mj_step(self.modelo, self.datos)
        
        obs = self._get_obs()
        
        v_x = self.datos.qvel[0]
        
        if v_x > 0:
            recompensa_avance = v_x * 10.0
            bono_supervivencia = 1.0
        else:
            recompensa_avance = v_x * 10.0 
            bono_supervivencia = 0.0 
            
        reward = recompensa_avance + bono_supervivencia
        
        altura_torso = self.datos.qpos[2]
        terminated = bool(altura_torso < 0.22)
        
        if terminated:
            reward -= 10.0
            
        return obs, float(reward), terminated, False, {}

env = Go2Env()

# --- 3. CARGAR Y CONTINUAR ENTRENAMIENTO ---
print("\n" + "="*60)
print(f"📖 CARGANDO CEREBRO EXISTENTE: {NOMBRE_MODELO}.zip")
print(f"📈 Conectando con los logs en: {DIRECTORIO_LOGS}")
print("="*60 + "\n")

# Al cargar, le volvemos a pasar la ruta del tensorboard para que reconecte con el historial
modelo_ppo = PPO.load(NOMBRE_MODELO, env=env, tensorboard_log=DIRECTORIO_LOGS)

print(f"Inyectando {PASOS_EXTRA} pasos adicionales de experiencia...")

# EL SECRETO: reset_num_timesteps=False
# Esto le dice a TensorBoard: "No empieces de 0, engánchate al final de la gráfica anterior"
modelo_ppo.learn(
    total_timesteps=PASOS_EXTRA, 
    tb_log_name="PPO_Go2_CPG_Robusto", # Mantén el mismo nombre base si quieres que agrupe bien
    reset_num_timesteps=False
)

# Sobrescribimos el modelo original con la versión mejorada
modelo_ppo.save(NOMBRE_MODELO)
print("\n✅ ¡Entrenamiento extra completado y archivo actualizado!\n")

# --- 4. INFERENCIA VISUAL (Para ver los resultados frescos) ---
obs, info = env.reset()

print("="*60)
print("🎥 LANZANDO VISUALIZADOR MUJOCO (MODELO MEJORADO)")
print("="*60)

with mujoco.viewer.launch_passive(env.modelo, env.datos) as viewer:
    while viewer.is_running():
        action, _states = modelo_ppo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        viewer.sync()
        time.sleep(env.modelo.opt.timestep)
        
        if terminated or truncated:
            obs, info = env.reset()
