# DualUAV


## Description
This project aims to design, implement, and evaluate a simulated dual-Unmanned Aerial Vehicle (UAV, drone) system where two drones collaborate to detect, share, and verify objects using computer vision and wireless communication. By leveraging multiple viewpoints, the system will improve detection accuracy, valuable for search and rescue (SAR) operations in challenging environments like dense forests or disaster zones. This will involve developing a lightweight object detection model to be implemented within the two drones. Communication protocols and drone positioning will be optimised to enhance reliability. The expected output is a functional prototype that demonstrates improved accuracy through coordinated UAV object-verification, which will be quantitatively evaluated within a controlled simulated environment.


## Pipeline
Airsim https://github.com/microsoft/AirSim

PX4 https://github.com/PX4/PX4-Autopilot

QGroundControl https://qgroundcontrol.com/


## Installation & Setup

### AirSim & Unreal Engine 4.27.2
Clone the Microsoft AirSim repository into your user documents folder and switch to the latest stable release using:
```
git clone https://github.com/microsoft/AirSim.git
git checkout [stable_branch_name]
```

Using Visual Studio Community 2022, go to `Tools > Command Line > Developer Command Prompt`, and navigate to your newly created AirSim repository "AirSim". Run the `build.cmd` Windows Command by typing exactly that in your AirSim directory. Once done, navigate into the `Unreal > Environments > Blocks` folder within AirSim, and double click the `Blocks.sln` solution file. Right click the Blocks folder and enable `Set as Startup Project`, ensuring these debugger configurations:
```
DebugGame Editor
Win64
Local Windows Debugger
```

Make sure Unreal Engine 4.27.2 (UE) is installed. The reason for this version is because it provides the best stability and support when it comes to using Unreal Engine with AirSim.

Back in Visual Studio, debug and confirm "Yes" to build the project to launch. After which, ensure `GameMode Overide` is set to `AirSimGameMode` in the `World Settings` tab on the right-hand side to enable the car and drone spawn feature. Press "Play" to launch the simulator, where "Yes" will spawn a car, and "No" will spawn a drone.

### PX4
Clone the PX4 repository into your user documents folder and switch to the latest stable release using:
```
git clone https://github.com/PX4/PX4-Autopilot.git
git checkout [stable_branch_name]
```

Using the Windows Subsystem for Linux (WSL) command line, navigate to your newly created PX4 repository "PX4-Autopilot" and into `/Tools/setup` to run `./ubuntu.sh` and build PX4. 

In the AirSim `settings.json` file, ensure your `ControlIP` is set to the Linux system IP address, which can be checked in WSL by using the command `ip a` and is commonly located under `eth0`. E.g., `172.17.49.163`. Ensure your `LocalHostIp` is set to what Linux appears as on Windows, which can be checked in your Windows Terminal by using the command `ipconfig`, where it could be located under, for example, `Ethernet adapter vEthernet (WSL (Hyper-V firewall))`. E.g., `172.17.48.1`.

On WSL, edit your Linux environment variables by using `nano ~/.bashrc`. Scroll to the very bottom and enter `export PX4_SIM_HOST_ADDR=172.17.48.1`, where the IP is your `LocalHostIp`. This will allow PX4 to work with AirSim/UE through WSL, as PX4 is native on Linux, but not so much on Windows.

Navigate back to the main PX4 directory and run `make px4_sitl_default none_iris` to run PX4. This command means:
```
make > create a profile (named px4_sitl_default)
sitl > software in the loop
none_iris > force our own simulator (use UE)
```



### QGroundControl
Download the executable and run it, then complete the setup wizard. Then in `Application Settings > Comm Links`, under `Links`:
1. Add two separate connections for UDP: 14580, and 14581.
2. For both connections, under `Server Addresses`, add your `ControlIp`.


### Pipeline Integration
1. Connect the two drones links in QGroundControl.
2. Run AirSim simulation.
3. Run PX4.
4. Arm the drones in QGroundControl, and input manual control in AirSim simulator. (Or, have them autonomously follow a flight path in QGroundControl)
