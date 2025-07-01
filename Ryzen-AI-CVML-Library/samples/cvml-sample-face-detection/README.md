# `cvml-sample-face-detection`

This sample demonstrates the implementation of AMD's Face Detection feature. 
It detects faces along with five landmarks(two eyes, nose and two edges of the mouth) and confidence scores in the input image, video, or camera.
Then visually represents the results in output images or video.

## Usage

```sh
cvml-sample-face-detection.exe [-i path_to_image/video] [-o output image/video filename] [-m fd_model] [-h]
Options
-i: Run face detection on the given image or video. (Optional)
-o: Specify output image or video file name e.g., .mp4 or .jpg. (Optional)
-m: Specify face detection model (precise/fast). Fast is the default. (Optional)
-h: Show usage.
If no arguments are provided, the application attempts to capture input from camera index 0.

Examples
Run the sample with an image input without output file:
cvml-sample-face-detection.exe -i my_image.jpg

Run the sample with a video input and save the result to an output video file:
cvml-sample-face-detection.exe -i my_video.mp4 -o output_video.mp4

Run the sample to capture the camera feed using the "precise" model and save the result to a video file:
cvml-sample-face-detection.exe -m precise -o output_video.mp4

Note
If the user runs the application without any arguments, it will use the camera as an input.

