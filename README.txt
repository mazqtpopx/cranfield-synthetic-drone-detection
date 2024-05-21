This code is largely based on 
https://github.com/pytorch/vision/tree/main/references/detection


To run this code, run the command:
python3 main.py

or, for multi-gpu training:
torchrun main.py

(Note that the evaluation can only be done in single-GPU mode)

Testing
We use 3 real-world datasets: MAV-Vid, Drone-vs-Bird, Anti-UAV. 

To perform the COCO evaluation, UAVDetectionTeackingBenchmark https://github.com/KostadinovShalon/UAVDetectionTrackingBenchmark utils is used to process the original videos to images. The coco files provided by UAVDetectionTeackingBenchmark for the benchmarking are used. Use [the scripts in utils] to output the individual video frames as jpeg images to a directory. 

Download the mav-vid, drone-vs-bird, and anti-uav datasets and output them to the datasets directory. To follow the config 
datasets
anti-uav, drone-vs-bird, mav-vid, multi-drones (multi-drones can be downloaded from [add link when embargo lifted])

1. Update the dataset IMG_DIR paths in config.py:
REPOS_DIR = "" - the repos directory (this should condain a datasets directory (i.e. containing the datasets) and a cranfield-synthetic-drone-vs-bird directory (i.e. this project)) 




Before 