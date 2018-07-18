## Learning to grasp unknown objects with a robotic arm from experience
###### Luca Monorchio's master thesis, July 2018

#### Abstract
The evolution of technology opened new revolutionary scenarios in the field of robotics,
making it underhand the human presence in favor of increasingly efficient and automated
systems. The use of robots capable of grasping complicated, and heavy, objects pushes
the industrial sector towards automatic palletizing, replacing human labour as far as possible.
The main reasons are purely economic, aimed to save time and production, which
are essential in this field. Robotics grasping research allows machines understand which
is the best way to grasp objects independently, and the choice of the best grasping pose.
Given a CAD model, typical approaches are based on estimating the pose of the object
by matching its model. Hence, the grasping pose is chosen according to the position and
orientation in which the object has been located. We are facing with a clear problem of
computer vision, a fundamental element to allow the robot to identify the region in the
space where the gripper will perform the grasping.
In the last years, deep learning based systems have reached impressive results in robot
grasping. Using the resources made available in the RoCoCo laboratory at Sapienza,
University of Rome, this thesis aims to solve the grasping problem of unknown objects
by employing a self-supervised, data driven learning approach. Given an RGB image I
and the corresponding depth map D acquired by a Microsoft Kinect2 sensor, we aim to
estimate the best grasping position for a 6 degrees of freedom (d.o.f.) anthropomorphic
robot manipulator equipped with a vacuum gripper. Among other contributions, this
work introduces a new set of data composed of 4000 acquisitions in which each acquisition
is related to a robot that try to grasp an object. We introduce a custom Convolutional
Neural Network (CNN) that has been trained with the acquired dataset.
We report several experiments performed by using known (i.e., object included in the
training dataset) and unknown objects, showing that our system is able to effectively
learn good grasping positions.
