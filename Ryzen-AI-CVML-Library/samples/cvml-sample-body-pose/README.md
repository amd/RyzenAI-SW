# `cvml-sample-body-pose`

This sample demonstrates the implementation of AMD's Body Pose feature.
It detects up to 6 people per frame and returns 17 keypoints per person (bounding box, landmark coordinates, and confidence scores) using the COCO-Pose keypoint convention.
Results are visually overlaid on the output image or video.

## Usage

```sh
cvml-sample-body-pose.exe [-i path_to_image/video] [-o output image/video filename] [-h]
Options
-i: Run body pose on the given image or video. (Optional)
-o: Specify output image or video file name e.g., .mp4 or .jpg. (Optional)
-h: Show usage.
If no arguments are provided, the application attempts to capture input from camera index 0.

Examples
Run the sample with an image input without output file:
cvml-sample-body-pose.exe -i my_image.jpg

Run the sample with a video input and save the result to an output video file:
cvml-sample-body-pose.exe -i my_video.mp4 -o output_video.mp4

Run the sample to capture the camera feed and save the result to a video file:
cvml-sample-body-pose.exe -o output_video.mp4

Note
If the user runs the application without any arguments, it will use the camera as an input.
