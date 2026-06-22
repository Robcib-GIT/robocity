#!/bin/bash
set -e

# =========================
# 1. Miniconda (solo si no existe)
# =========================
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Instalando Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
else
    echo "Miniconda ya existe, saltando..."
fi

source $HOME/miniconda3/bin/activate

# =========================
# 2. Mamba
# =========================
conda install mamba -n base -c conda-forge -y

# =========================
# 3. Entorno
# =========================
if conda info --envs | grep -q "robocity_cuadruped_robot"; then
    echo "Entorno ya existe, saltando creación..."
else
    mamba create -n robocity_cuadruped_robot python=3.10.12 -y
fi

conda activate robocity_cuadruped_robot

# =========================
# 4. Canales ROS
# =========================
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --set channel_priority strict

# =========================
# 5. ROS
# =========================
mamba install -y \
ros-humble-ros-base \
ros-humble-sensor-msgs \
ros-humble-geometry-msgs \
ros-humble-tf2-ros \
ros-humble-rviz2 \
ros-humble-teleop-twist-keyboard \
typeguard

# =========================
# 6. Python stack
# =========================
pip install \
mujoco==3.6.0 \
numpy==2.2.6 \
"stable-baselines3[extra]" \
"gymnasium[mujoco]" \
tensorboard \
gTTS \
moviepy \
imageio \
mediapy

# =========================
# 7. Repo
# =========================
mkdir -p ~/robocity_cuadruped_robot
cd ~/robocity_cuadruped_robot

if [ ! -d "mujoco_menagerie" ]; then
    git clone https://github.com/google-deepmind/mujoco_menagerie.git
else
    echo "mujoco_menagerie ya existe, saltando..."
fi

# =========================
# 8. ROS fix automático
# =========================
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'source $CONDA_PREFIX/setup.bash' > $CONDA_PREFIX/etc/conda/activate.d/ros.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/ros.sh

echo "Setup completado correctamente."
