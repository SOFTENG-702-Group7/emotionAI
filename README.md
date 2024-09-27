# HCI Group 7 Project - Enhancing Group Cognition in Remote Meetings Using AI-Mediated Feedback

# Project Overview
This project is an emotion recognition tool designed to assist in remote meetings. Since we are unable to integrate the project into existing software such as Zoom, Teams, etc., we created this tool to help users recognize the facial emotions of participants by having them face the computer screen. Therefore, the prerequisite for using this tool is that another laptop is needed to join the meeting. Then, the meeting screen should be positioned in front of the camera of the computer running this project for optimal use.

# The CNN model used in this project has an accuracy of 77.5%. If higher accuracy is required, you can retrain the model in the 'model_training' folder.

# Features
1.Read the user's facial emotions through the camera.
2.Support for Multiple Emotions: Recognizes multiple emotions such as happiness, anger, surprise, fear, and neutral states.

# Environment Setup

1. Install anaconda
    a. Download for your operating system < https://www.anaconda.com/download >
    b. Follow installation instructions. If you are on linux, you need to make the anaconda.sh file executable, then run “./anaconda.sh” in the terminal.
    c. In the Anaconda prompt (windows) or a new terminal (linux/osx), create a new conda environment conda create --name emotionAI python=3.9
    d. Activate your environment conda activate emotionAI. 
    # You will need to repeat this step every time you want to activate your environment.

2. Install Pytorch
    a. Navigate to < https://pytorch.org/ >
    b. Under the Install PyTorch section, select the stable build, the OS you are using, Conda for type of package and Python for type of language. If you have a Nvidia GPU, select CUDA 11.8 for the compute platform, otherwise select CPU.
    c. Copy and paste the command into your anaconda prompt/terminal. Ensure you are still in the emotionAI environment. This will take some time.

3. Install remaining packages
    a. In your emotionAI environment, use pip to install the correct versions of opencv, ultralytics, numpy and other packges:
        pip install numpy==1.23.5 scipy==1.9.3 opencv-python-headless torch torchvision
        pip install ultralytics
        pip install timm
4. Verify installation
    a. In your conda environment, enter the following commands:
        python
        import torch
    b. There should be no errors.
    c. ONLY if you installed pytorch with CUDA, torch.cuda.is_available() should return True. Skip this step if you did not install with cuda or do not have an Nvidia GPU.
    d. Type exit() to exit the python interpreter.

