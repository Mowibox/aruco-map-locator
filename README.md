# aruco-map-locator

This repo contains the ROS 2 code for locating and estimating a robot position on a map using ArUco tags.

![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)
![ROS 2](https://img.shields.io/badge/ROS2-Humble-blue?logo=ros)
![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-33cc55?logo=opencv)
![Issues](https://img.shields.io/github/issues/Mowibox/aruco-map-locator)

<p align="center">
  <img alt="aruco_map_locator" src="https://github.com/user-attachments/assets/a4c17f62-d4c8-48ce-b297-6aa309cdb48d"/>
</p>

## Table of contents

| Section                               | Description                                                               |
| ------------------------------------- | ------------------------------------------------------------------------- |
| [Project overview](#project-overview) | General description of the ArUco-based localization system                |
| [Authors](#authors)                   | Main contributors information                                             |
| [Documentation](#documentation)       | Links to detailed wiki and presentation materials                         |
| [Contributions](#contributions)       | How to contribute to the repository                                       |
| [References](#references)             | Scientific references                                                     |
| [License](#license)                   | Licensing information                                                     |

## Project overview

Precise localization of its environment and dynamic obstacle detection are crucial challenges in autonomous mobile robotics. These issues can be explored in competitions such as the French Robotics Cup. In this repository, I propose an approach to robot localization based on fiducial markers (ArUco tags). The aim is to determine the precise position of a robot in the game space, as well as to estimate the position of obstacles in real time (game elements and opposing robots) [[1]](#references).

## Authors

[Ousmane THIONGANE](https://mowibox.github.io)

## Documentation

 More details about this project are specified in the [wiki.](https://github.com/Mowibox/aruco-map-locator/wiki). A short presentation is also available in the ['docs/'](https://github.com/Mowibox/aruco-map-locator/tree/main/docs) folder.

## Contributions

Contributions are always welcome!

* Report Issues: Found a bug or have a feature request? Create a new issue [here.](https://github.com/Mowibox/aruco-map-locator/issues/new/choose)
* Fix Bugs & Add Features: Find out where you can lend a hand by checking out [existing issues.](https://github.com/Mowibox/aruco-map-locator/issues)

## References

> * [1] Eric Marchand, Hideaki Uchiyama, Fabien Spindler. Pose Estimation for Augmented Reality: A Hands-On Survey. IEEE Transactions on Visualization and Computer Graphics, 2016, 22 (12), pp.2633 - 2651. ⟨10.1109/TVCG.2015.2513408⟩. ⟨hal-01246370⟩. https://ieeexplore.ieee.org/document/7368948

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Mowibox/aruco-map-locator/blob/main/LICENSE) file for more details.
