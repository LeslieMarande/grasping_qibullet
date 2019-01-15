#!/usr/bin/env python
# coding: utf-8

import pybullet
import threading
import pybullet_data
from qibullet.pepper_virtual import PepperVirtual


class SimulationManager:
    """
    Class allowing to handle the different parameters of a pybullet simulation
    """

    def __init__(self):
        """
        Constructor
        """
        pass

    def launchSimulation(self, gui=True):
        """
        Launches a simulation instance

        Parameters:
            gui - Boolean, if True the simulation is launched with a GUI, and
            with no GUI otherwise

        Returns:
            physics_client - The id of the simulation client created
        """
        if gui:
            physics_client = pybullet.connect(pybullet.GUI)
            pybullet.setRealTimeSimulation(1, physicsClientId=physics_client)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,
                0,
                physicsClientId=physics_client)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                0,
                physicsClientId=physics_client)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                0,
                physicsClientId=physics_client)
        else:
            physics_client = pybullet.connect(pybullet.DIRECT)
            threading.Thread(
                target=self._stepSimulation,
                args=[physics_client]).start()

        pybullet.setGravity(0, 0, -9.81, physicsClientId=physics_client)
        return physics_client

    def resetSimulation(self, physics_client):
        """
        Resets the simulated instance corresponding to the physics client id.
        All of the objects loaded in the simulation will be destroyed, but the
        instance will still be running
        """
        pybullet.resetSimulation(physicsClientId=physics_client)

    def stopSimulation(self, physics_client):
        """
        Stops the simulated instance corresponding to the physics_client id

        Parameters:
            physics_client - The id of the simulated instance to be stopped
        """
        pybullet.disconnect(physicsClientId=physics_client)

    def spawnPepper(
            self,
            physics_client,
            translation,
            quaternion,
            spawn_ground_plane=False):
        """
        Loads a Pepper model in the simulation

        Parameters:
            physics_client - The id of the simulated instance in which the
            robot is supposed to be spawned
            translation - List containing 3 elements, the spawning translation
            [x, y, z] in the WORLD frame
            quaternions - List containing 4 elements, the spawning rotation as
            a quaternion [x, y, z, w] in the WORLD frame
            spawn_ground_plane - If True, the pybullet_data ground plane will
            be spawned

        Returns:
            pepper - A PepperVirtual object, the Pepper simulated instance
        """
        pepper = PepperVirtual()

        if spawn_ground_plane:
            pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet.loadMJCF(
                "mjcf/ground_plane.xml",
                physicsClientId=physics_client)

        pepper.loadRobot(
            translation,
            quaternion,
            physicsClientId=physics_client)

        return pepper

    def _stepSimulation(self, physics_client):
        """
        INTERNAL METHOD: Steps the simulated instance corresponding to the
        physics_client id

        Parameters:
            physics_client - The id of the simulated instance to be stepped
        """
        try:
            while True:
                pybullet.stepSimulation(physicsClientId=client_id)
                time.sleep(0.001)
        except Exception:
            pass
