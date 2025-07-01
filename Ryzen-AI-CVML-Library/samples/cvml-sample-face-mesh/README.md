# `cvml-sample-facemesh`

This sample demonstrates the implementation of AMD's Face Mesh feature. 
It detects faces in the input image or video and computes face landmarks and head pose information, and then visually represents the results in output images or video.

## Usage

```sh
cvml-sample-facemesh.exe [-i path_to_video/image] [-h] [-o output image/video filename] [-?]
Options
-i: Specify an input image or video.
-o: Specify output image or video file name e.g., .mp4 or .jpg.
-fd: Specify face detection model (precise/fast). Precise is the default. (Optional)
-h: Show usage.
-?: Show usage.


Examples:

Run the sample with a camera input without output file:
cvml-sample-facemesh.exe

Run the sample with a video input without output file:
cvml-sample-facemesh.exe -i my_video.mp4

Run the sample with an image input and save the result to an image file:
cvml-sample-facemesh.exe -i my_image.jpg -o output_image.jpg

Run the sample to capture the camera feed and save the result to a video file:
cvml-sample-facemesh.exe -o output_video.mp4

Note
If the user runs the application without any arguments, it will use the camera as an input.
