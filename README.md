# Robocity SummerSchool - Quadruped Robots Environment and RL

Welcome to the course repository! Here you will find all the necessary scripts, models, and tools to simulate and control quadruped robots using ROS 2 (Humble), MuJoCo, and Reinforcement Learning.

## 🛠️ Manual Installation

Please open a terminal in Ubuntu and strictly follow these steps one by one to set up your environment.

### 1. Install Miniconda
Download and install Miniconda (Python environment manager):

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    source $HOME/miniconda3/bin/activate

### 2. Install Mamba
Install Mamba using the classic solver and verbose mode to avoid installation freezes:

    conda install mamba -n base -c conda-forge --solver=classic -vvv

### 3. Create the Workspace and Environment
Create a specific folder for the course and the isolated Conda environment:

    mkdir -p ~/robocity_cuadruped_robot
    mamba create -n robocity_cuadruped_robot python=3.10.12 -y
    conda activate robocity_cuadruped_robot

### 4. Install ROS 2 (Humble)
Configure the required channels and install the ROS 2 packages:

    conda config --env --add channels conda-forge
    conda config --env --add channels robostack-staging
    conda config --env --set channel_priority strict

    mamba install ros-humble-ros-base ros-humble-sensor-msgs ros-humble-geometry-msgs ros-humble-tf2-ros ros-humble-rviz2 ros-humble-teleop-twist-keyboard typeguard -y

### 5. Install Python Dependencies
Install the required libraries for Reinforcement Learning and Simulation:

    pip install mujoco==3.6.0 numpy==2.2.6 "stable-baselines3[extra]" "gymnasium[mujoco]" tensorboard gTTS moviepy imageio mediapy

### 6. Download the Course Repositories
Navigate to your workspace, download the course files, and the MuJoCo Menagerie models:

    git clone https://github.com/Robcib-GIT/robocity.git robocity_cuadruped_robot
    cd ~/robocity_cuadruped_robot
    git clone https://github.com/google-deepmind/mujoco_menagerie.git

*(Note: Remember to replace the unitree_go2 folder inside mujoco_menagerie with the modified one provided in the course repository).*

---

## 🚀 Launching Base Codes

Before running any script, make sure your environment is activated:

    conda activate robocity_cuadruped_robot

Next, open different terminal tabs (activating the environment in each one) to launch the base modules:

* **1. Environment Launch:**

    cd ~/robocity_cuadruped_robot/base && python3 inicio_ros.py 

* **2. Locomotion Node:**

    cd ~/robocity_cuadruped_robot/base && python3 locomotion_node.py

* **3. Teleoperation (Manual Control):**

    ros2 run teleop_twist_keyboard teleop_twist_keyboard

* **4. Autonomous Navigation:**

    cd ~/robocity_cuadruped_robot/base && python3 navigation.py

---

## 🧠 Launching RL (Reinforcement Learning) Codes

For the reinforcement learning section, run the following scripts depending on the case study. Make sure your environment is activated:

* **Case 1 - Environment:**

        cd ~/robocity_cuadruped_robot/scripts/caso_1_entorno && python3 inicio.py 

* **Case 2 - Failures (Falling Forward & Vibration):**

  * *Training 1:*

        cd ~/robocity_cuadruped_robot/scripts/caso_2_fallos/cae_adelante && python3 entrenamiento.py

  * *Training Evolution (TensorBoard):*

        cd ~/robocity_cuadruped_robot/scripts/caso_2_fallos/cae_adelante && tensorboard --logdir ./logs/

  * *Inference (1 Million Epoch Model):*

        cd ~/robocity_cuadruped_robot/scripts/caso_2_fallos/cae_adelante && python3 cae_adelante.py

  * *Inference (Vibration Model):*

        cd ~/robocity_cuadruped_robot/scripts/caso_2_fallos/vibracion && python3 vibracion.py

* **Case 3 - Mobility (Terrains and Refinement):**

    cd ~/robocity_cuadruped_robot/scripts/caso_3_movilidad && python3 inferencia.py

  > **Note:** You can try changing the scene to "scene_irregular.xml" or changing the robot type to "anybotics_anymal_c" inside the script.

* **Case 4 - Robustness (GCPM + Residual):**

  * *Training and Inference:*

        cd ~/robocity_cuadruped_robot/scripts/caso_4_robustez && python3 train_inference.py

  * *Continue Training:*

        cd ~/robocity_cuadruped_robot/scripts/caso_4_robustez && python3 continuar_entrenamiento.py

---


## 🛠️ Manual Installation (macOS)

Please open a terminal in macOS and strictly follow these steps one by one to set up your environment.

### 1. Install Miniconda
On Mac, the wget command is not included by default, so we will use curl. Depending on your Mac's chip, run the corresponding commands:

For Mac with Apple Silicon (M1, M2, M3...):

    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3
    source $HOME/miniconda3/bin/activate

For Mac with an Intel processor:

    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3
    source $HOME/miniconda3/bin/activate

### 2. Install Mamba
Install mamba in the base environment using the classic solver to avoid installation freezes:

    conda install mamba -n base -c conda-forge --solver=classic -y

### 3. Create the Workspace and Environment
Create the Conda environment with the exact Python version required for the course:

    mkdir -p ~/robocity_cuadruped_robot
    mamba create -n robocity_cuadruped_robot python=3.10.12 -y
    conda activate robocity_cuadruped_robot

### 4. Install ROS 2 (Humble)
RoboStack natively supports ROS 2 Humble on macOS. Configure the conda channels and install it:

    conda config --env --add channels conda-forge
    conda config --env --add channels robostack-staging
    conda config --env --set channel_priority strict

    mamba install ros-humble-ros-base ros-humble-sensor-msgs ros-humble-geometry-msgs ros-humble-tf2-ros ros-humble-rviz2 ros-humble-teleop-twist-keyboard typeguard -y

### 5. Install Python Dependencies
Install the MuJoCo physics simulator and the Reinforcement Learning libraries:

    pip install mujoco==3.6.0 numpy==2.2.6 "stable-baselines3[extra]" "gymnasium[mujoco]" tensorboard gTTS moviepy imageio mediapy

### 6. Download the Course Repositories
Download the environment files and simulation models. (If you haven't used git before, running this might prompt your Mac to automatically install the "Command Line Tools", click accept):

    cd ~
    git clone https://github.com/Robcib-GIT/robocity.git robocity_cuadruped_robot_temp
    cp -r robocity_cuadruped_robot_temp/* ~/robocity_cuadruped_robot/
    rm -rf robocity_cuadruped_robot_temp
    cd ~/robocity_cuadruped_robot
    git clone https://github.com/google-deepmind/mujoco_menagerie.git

(Note: Remember to replace the unitree_go2 folder inside mujoco_menagerie with the modified folder included in the base course files).

### 🚀 How to Run the Code on Mac
From here on out, the workflow is identical to Linux. The only detail to keep in mind is that every time you open a new Terminal tab, you must activate conda and your environment before running any Python scripts or ROS commands:

    source $HOME/miniconda3/bin/activate
    conda activate robocity_cuadruped_robot

Once activated, you can run the different course nodes in separate tabs. For example, to launch the main environment:

    cd ~/robocity_cuadruped_robot/base
    mjpython inicio_ros.py


## ⚠️ Potential Environment Errors

**1. Mamba not found during ROS installation:**
If the terminal throws a "command not found" error for Mamba (or suggests installing `samba` instead), you can bypass it by forcing Conda to run Mamba directly from the base environment. Use this command instead of the standard ROS installation command:

    conda run -n base mamba install -n robocity_cuadruped_robot ros-humble-ros-base ros-humble-sensor-msgs ros-humble-geometry-msgs ros-humble-tf2-ros ros-humble-rviz2 ros-humble-teleop-twist-keyboard typeguard -y

**2. ROS path conflicts or .bashrc issues:**
If you experience issues with your ROS paths, run the following cleanup commands in your terminal:

    source $CONDA_PREFIX/setup.bash
    unset PYTHONPATH
    unset AMENT_PREFIX_PATH
    unset CMAKE_PREFIX_PATH
    unset ROS_PACKAGE_PATH
    unset LD_LIBRARY_PATH
