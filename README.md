# Parallel Canny Edge Detection

## Description
This project implements the Canny edge detection algorithm in both sequential and parallel versions using OpenMP. It compares the performance of the sequential and parallel implementations across different problem sizes and numbers of processing elements.

## Files
1. `canny_cuda.cpp`: The main C++ source code file containing implementations of sequential and parallel Canny edge detection algorithms using OpenMP.
2. `Butterfly.jpg`: Sample input image for testing the edge detection algorithms.

## Dependencies
1. OpenCV: Required for image processing operations.
2. Gnuplot: Used for generating performance plots.

## Usage
1. Ensure that OpenCV and Gnuplot are installed on your system.
2. Compile the C++ source code using an appropriate compiler with OpenMP support. For example:
   ```
   g++ -o canny_cuda canny_cuda.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lgomp
   ```
3. Run the compiled executable:
   ```
   ./canny_cuda
   ```
4. The program will perform edge detection on the sample image and generate performance plots.

## Output
1. `speedup.png`: Plot showing Speed Up vs Number of Processing Elements.
2. `parallelefficiency.png`: Plot showing Parallel Efficiency vs Number of Processing Elements.

## Notes
- Ensure that the input image (`Butterfly.jpg`) is present in the same directory as the executable.
- Adjust parameters such as high and low thresholds for edge detection as needed in the source code.
- Experiment with different problem sizes and numbers of processing elements to observe performance variations.
