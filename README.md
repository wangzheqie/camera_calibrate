## Stereo Camera Calibrate

* The calibration program that save data in each step
* The stereo camera calibration that use Halcon Circle Board
* You should change your `objp` in line 40 and 41

## Test Dependences

* Tested with python 2.7 and OpenCV 2.4.13

## Run

1. Put your images to the `input/cam_left` and `input/cam_right` with the same name
2. Change your camera and board information in the `main()` context
3. Run the functions in `main()` step by step
4. The calculation data will be saved after runing each single function, and may be used by other steps

