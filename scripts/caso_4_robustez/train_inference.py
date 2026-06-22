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
RUTA_XML = "/home/leo/robocity_cuadruped_robot/mujoco_menagerie/unitree_go2/scene.xml"

# Rutas exactas solicitadas
NOMBRE_MODELO = "/home/leo/robocity_cuadruped_robot/scripts/caso_4_robustez/pesos/modelo_go2_robusto"
DIRECTORIO_LOGS = "/home/leo/robocity_cuadruped_robot/scripts/caso_4_robustez/logs/"

# Creamos las carpetas automáticamente si no existen para evitar errores
os.makedirs(os.path.dirname(NOMBRE_MODELO), exist_ok=True)
os.makedirs(DIRECTORIO_LOGS, exist_ok=True)

# Cambia a True si quieres pisar el modelo antiguo y entrenar de cero
FORZAR_REENTRENAMIENTO = False 

# --- 2. EL ENTORNO CPG + RL (El Pedaleo con Residuos) ---
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
        
        # El reloj interno es VITAL aquí para que la IA sepa en qué fase del paso está
        self.tiempo_simulacion = 0.0 

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        # Observaciones: qpos + qvel + 1 variable de tiempo
        obs_dim = self.modelo.nq + self.modelo.nv + 1 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.modelo, self.datos)
        self.datos.qpos[7:19] = self.postura_nominal
        self.tiempo_simulacion = 0.0 # Reiniciamos el reloj al caerse
        return self._get_obs(), {}

    def _get_obs(self):
        # Le pasamos a la IA todo su estado físico más el tiempo actual
        obs = np.concatenate([self.datos.qpos, self.datos.qvel, [self.tiempo_simulacion]])
        return obs.astype(np.float32)

    def step(self, action):
        self.tiempo_simulacion += self.modelo.opt.timestep
        
        # --- EL MOTOR CPG (Generador de Patrones Centrales) ---
        frecuencia = 5.0 
        
        # Seno (Adelante / Atrás para las caderas)
        fase_cadera_1 = np.sin(2 * np.pi * frecuencia * self.tiempo_simulacion)
        fase_cadera_2 = np.sin(2 * np.pi * frecuencia * self.tiempo_simulacion + np.pi)
        
        # Coseno (Arriba / Abajo para las rodillas)
        fase_rodilla_1 = np.cos(2 * np.pi * frecuencia * self.tiempo_simulacion)
        fase_rodilla_2 = np.cos(2 * np.pi * frecuencia * self.tiempo_simulacion + np.pi)
        
        oscilacion = np.zeros(12)
        amp_cadera = 0.4
        amp_rodilla = 0.4 
        
        # Diagonal 1 (FR y RL)
        oscilacion[1] = fase_cadera_1 * amp_cadera    
        oscilacion[10] = fase_cadera_1 * amp_cadera   
        oscilacion[2] = fase_rodilla_1 * amp_rodilla  
        oscilacion[11] = fase_rodilla_1 * amp_rodilla 
        
        # Diagonal 2 (FL y RR)
        oscilacion[4] = fase_cadera_2 * amp_cadera    
        oscilacion[7] = fase_cadera_2 * amp_cadera    
        oscilacion[5] = fase_rodilla_2 * amp_rodilla  
        oscilacion[8] = fase_rodilla_2 * amp_rodilla  

        angulos_actuales = self.datos.qpos[7:19]
        velocidades_actuales = self.datos.qvel[6:18]
        
        # FUSIÓN: Postura Base + Ritmo Matemático + Micro-ajustes de la IA (Residuo)
        angulos_deseados = self.postura_nominal + oscilacion + (action * 0.3)
        
        error_posicion = angulos_deseados - angulos_actuales
        torques = (self.Kp * error_posicion) - (self.Kd * velocidades_actuales)

        self.datos.ctrl[:12] = torques
        mujoco.mj_step(self.modelo, self.datos)
        
        obs = self._get_obs()
        
        # --- REWARD SHAPING (Anti-Moonwalk) ---
        v_x = self.datos.qvel[0]
        
        if v_x > 0:
            recompensa_avance = v_x * 10.0
            bono_supervivencia = 1.0
        else:
            # Si va hacia atrás o resbala, el bono desaparece y se le restan puntos
            recompensa_avance = v_x * 10.0 
            bono_supervivencia = 0.0 
            
        reward = recompensa_avance + bono_supervivencia
        
        altura_torso = self.datos.qpos[2]
        terminated = bool(altura_torso < 0.22)
        
        if terminated:
            reward -= 10.0
            
        return obs, float(reward), terminated, False, {}

env = Go2Env()

# --- 3. GESTOR DE ENTRENAMIENTO / INFERENCIA ---
ruta_zip = f"{NOMBRE_MODELO}.zip"

if os.path.exists(ruta_zip) and not FORZAR_REENTRENAMIENTO:
    print("\n" + "="*50)
    print(f"✅ ¡Cerebro encontrado en {ruta_zip}!")
    print("Saltando entrenamiento. Preparando inferencia...")
    print("="*50 + "\n")
    modelo_ppo = PPO.load(NOMBRE_MODELO, env=env)
    
else:
    print("\n" + "="*50)
    print("🚀 INICIANDO ENTRENAMIENTO (CPG + APRENDIZAJE RESIDUAL)")
    print(f"Los pesos se guardarán en: {NOMBRE_MODELO}")
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
    
    # 500k pasos suelen ser suficientes aquí porque la matemática hace el 80% del trabajo
    modelo_ppo.learn(total_timesteps=100000, tb_log_name="PPO_Go2_CPG_Robusto")
    modelo_ppo.save(NOMBRE_MODELO)
    print("\n¡Entrenamiento completado y guardado!\n")

# --- 4. INFERENCIA VISUAL ---
obs, info = env.reset()

print("="*50)
print(" LANZANDO VISUALIZADOR MUJOCO")
print("Cierra la ventana gráfica para detener el script.")
print("="*50)

with mujoco.viewer.launch_passive(env.modelo, env.datos) as viewer:
    while viewer.is_running():
        # En esta arquitectura, determinista=True es vital para ver el resultado limpio
        action, _states = modelo_ppo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        viewer.sync()
        time.sleep(env.modelo.opt.timestep)
        
        if terminated or truncated:
            obs, info = env.reset()
