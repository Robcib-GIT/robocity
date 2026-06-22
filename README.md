# Robocity Course - Quadruped Environment and RL

Welcome to the course repository! Here you will find all the necessary scripts, models, and tools to simulate and control quadruped robots using ROS 2 (Humble), MuJoCo, and Reinforcement Learning.

## 🛠️ Automatic Installation

To make the setup easier, we have prepared a script that will automatically install Miniconda, Mamba, create the isolated environment (`robocity_cuadruped_robot`), and install all necessary ROS and Python dependencies.

Open a terminal in Ubuntu and run the following commands:

1. Clone this repository to your computer:
```bash
git clone https://github.com/Robcib-GIT/robocity.git
cd robocity
```

2. Grant execution permissions to the script and run it:
```bash
chmod +x install_robocity.sh
./install_robocity.sh
```

3. Once finished, **close your terminal and open a new one** for the changes to take effect.

---

## 🚀 Launching Base Codes

Before running any script, make sure your environment is activated:
```bash
conda activate robocity_cuadruped_robot
```

Next, open different terminal tabs (activating the environment in each one) to launch the base modules:

* **1. Environment Launch:**
```bash
cd ~/robocity_cuadruped_robot/base && python3 inicio_ros.py 
```

* **2. Locomotion Node:**
```bash
cd ~/robocity_cuadruped_robot/base && python3 locomotion_node.py
```

* **3. Teleoperation (Manual Control):**
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

* **4. Autonomous Navigation:**
```bash
cd ~/robocity_cuadruped_robot/base && python3 navigation.py
```

---

## 🧠 Launching RL (Reinforcement Learning) Codes

For the reinforcement learning section, run the following scripts depending on the case study. Make sure your environment is activated:

* **Case 1 - Environment:**
```bash
cd ~/robocity_cuadruped_robot/scripts/caso_1_entorno && python3 inicio.py 
```

* **Case 2 - Failures (Falling Forward & Vibration):**
  * *Training 1:*
  ```bash
  cd ~/robocity_cuadruped_robot/scripts/caso_2_fallos/cae_adelante && python3 entrenamiento.py
  ```
  * *Training Evolution (TensorBoard):*
  ```bash
  cd ~/robocity_cuadruped_robot/scripts/caso_2_fallos/cae_adelante && tensorboard --logdir ./logs/
  ```
  * *Inference (1 Million Epoch Model):*
  ```bash
  cd ~/robocity_cuadruped_robot/scripts/caso_2_fallos/cae_adelante && python3 cae_adelante.py
  ```
  * *Inference (Vibration Model):*
  ```bash
  cd ~/robocity_cuadruped_robot/scripts/caso_2_fallos/vibracion && python3 vibracion.py
  ```

* **Case 3 - Mobility (Terrains and Refinement):**
```bash
cd ~/robocity_cuadruped_robot/scripts/caso_3_movilidad && python3 inferencia.py
```
> **Note:** You can try changing the scene to `"scene_irregular.xml"` or changing the robot type to `"anybotics_anymal_c"` inside the script.

* **Case 4 - Robustness (GCPM + Residual):**
  * *Training and Inference:*
  ```bash
  cd ~/robocity_cuadruped_robot/scripts/caso_4_robustez && python3 train_inference.py
  ```
  * *Continue Training:*
  ```bash
  cd ~/robocity_cuadruped_robot/scripts/caso_4_robustez && python3 continuar_entrenamiento.py
  ```

---

## ⚠️ Potential Environment Errors

If you experience issues with your `.bashrc` file or ROS path conflicts, run the following cleanup commands in your terminal:
```bash
source $CONDA_PREFIX/setup.bash

unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset ROS_PACKAGE_PATH
unset LD_LIBRARY_PATH
```
