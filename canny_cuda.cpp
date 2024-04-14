#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <omp.h>

// Include the Gnuplot library
#include "gnuplot-iostream.h"

using namespace cv;

// Function to perform non-maximum suppression
void nonMaxSuppression(const Mat& gradientMagnitude, const Mat& gradientDirection, Mat& nonMaxSuppressed) {
    int rows = gradientMagnitude.rows;
    int cols = gradientMagnitude.cols;
    
    nonMaxSuppressed = Mat(rows, cols, CV_8UC1, Scalar(0));

    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            float magnitude = gradientMagnitude.at<float>(y, x);
            float direction = gradientDirection.at<float>(y, x);

            float grad1, grad2;

            // Determine neighbor pixel coordinates based on gradient direction
            if (direction < 0) {
                direction += M_PI;
            }

            if (direction <= M_PI / 8 || direction > 7 * M_PI / 8) {
                grad1 = gradientMagnitude.at<float>(y, x + 1);
                grad2 = gradientMagnitude.at<float>(y, x - 1);
            } else if (direction <= 3 * M_PI / 8) {
                grad1 = gradientMagnitude.at<float>(y - 1, x + 1);
                grad2 = gradientMagnitude.at<float>(y + 1, x - 1);
            } else if (direction <= 5 * M_PI / 8) {
                grad1 = gradientMagnitude.at<float>(y - 1, x);
                grad2 = gradientMagnitude.at<float>(y + 1, x);
            } else {
                grad1 = gradientMagnitude.at<float>(y - 1, x - 1);
                grad2 = gradientMagnitude.at<float>(y + 1, x + 1);
            }

            // Suppress non-maximum pixels
            if (magnitude >= grad1 && magnitude >= grad2) {
                nonMaxSuppressed.at<uchar>(y, x) = static_cast<uchar>(magnitude);
            }
        }
    }
}

// Function to perform hysteresis thresholding
void hysteresisThresholding(const Mat& nonMaxSuppressed, Mat& edges, double lowThreshold, double highThreshold) {
    int rows = nonMaxSuppressed.rows;
    int cols = nonMaxSuppressed.cols;

    edges = Mat(rows, cols, CV_8UC1, Scalar(0));

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            uchar pixelValue = nonMaxSuppressed.at<uchar>(y, x);

            if (pixelValue >= highThreshold) {
                edges.at<uchar>(y, x) = 255; // Strong edge
            } else if (pixelValue >= lowThreshold) {
                edges.at<uchar>(y, x) = 127; // Weak edge
            } else {
                edges.at<uchar>(y, x) = 0; // Suppress
            }
        }
    }
}

// Sequential Canny edge detection function
void sequentialCannyEdgeDetection(const Mat& image) {
    // Convert image to grayscale if it's not already
    Mat grayImage;
    if (image.channels() > 1) {
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    // Apply Gaussian blur to reduce noise
    Mat blurredImage;
    GaussianBlur(grayImage, blurredImage, Size(5, 5), 1.4, 1.4);

    // Compute gradients using Sobel operator
    Mat gradX, gradY;
    Sobel(blurredImage, gradX, CV_32F, 1, 0);
    Sobel(blurredImage, gradY, CV_32F, 0, 1);

	if (gradX.depth() != CV_32F && gradX.depth() != CV_64F) {
		std::cerr << "Error: Gradient matrices have unsupported depth." << std::endl;
		return;
	}

	// Convert depth if necessary
	if (gradX.depth() != CV_32F) {
		gradX.convertTo(gradX, CV_32F);
		gradY.convertTo(gradY, CV_32F);
	}

    // Convert depth if necessary
    if (gradX.depth() != CV_32F) {
        gradX.convertTo(gradX, CV_32F);
        gradY.convertTo(gradY, CV_32F);
    }

    // Compute gradient magnitude and direction
    Mat gradientMagnitude, gradientDirection;
    cartToPolar(gradX, gradY, gradientMagnitude, gradientDirection, true);

    // Non-maximum suppression
    Mat nonMaxSuppressed;
    nonMaxSuppression(gradientMagnitude, gradientDirection, nonMaxSuppressed);

    // Apply hysteresis thresholding
    Mat edges;
    double highThreshold = 150;
    double lowThreshold = 50;
    hysteresisThresholding(nonMaxSuppressed, edges, lowThreshold, highThreshold);
}

void parallelCannyEdgeDetection(const unsigned char* inputImage, unsigned char* outputImage, int width, int height) {

    // Check if indices are within image bounds
	#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
			// Compute gradient magnitude and orientation using Sobel operators
			// Sobel operators
			int grad_x = 0, grad_y = 0;

			// Define Sobel kernels
			const int sobelX[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
			const int sobelY[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

			// Apply Sobel operators
			for (int i = -1; i <= 1; ++i) {
				for (int j = -1; j <= 1; ++j) {
					int pixelX = min(max(x + i, 0), width - 1);
					int pixelY = min(max(y + j, 0), height - 1);
					int sobelIndex = (i + 1) * 3 + (j + 1);
					grad_x += inputImage[pixelY * width + pixelX] * sobelX[sobelIndex];
					grad_y += inputImage[pixelY * width + pixelX] * sobelY[sobelIndex];
				}
			}

			// Compute magnitude and direction
			float magnitude = sqrtf(grad_x * grad_x + grad_y * grad_y);
			float orientation = atan2(grad_y, grad_x) * 180 / M_PI;

			// Perform non-maximum suppression
			float gradientDirection = atan2(grad_y, grad_x);
			float grad1, grad2, grad3, grad4;
			float pixelValue = inputImage[y * width + x];

			if (gradientDirection < 0) {
				gradientDirection += M_PI;
			}

			if (gradientDirection <= M_PI / 8 || gradientDirection > 7 * M_PI / 8) {
				grad1 = inputImage[(y + 1) * width + x];
				grad2 = inputImage[(y - 1) * width + x];
			} else if (gradientDirection <= 3 * M_PI / 8) {
				grad1 = inputImage[(y + 1) * width + (x + 1)];
				grad2 = inputImage[(y - 1) * width + (x - 1)];
			} else if (gradientDirection <= 5 * M_PI / 8) {
				grad1 = inputImage[y * width + (x + 1)];
				grad2 = inputImage[y * width + (x - 1)];
			} else {
				grad1 = inputImage[(y - 1) * width + (x + 1)];
				grad2 = inputImage[(y + 1) * width + (x - 1)];
			}

			// Compare pixelValue with its neighbors and suppress non-maximum pixels
			if (pixelValue >= grad1 && pixelValue >= grad2) {
				outputImage[y * width + x] = pixelValue;
			} else {
				outputImage[y * width + x] = 0;
			}

			// Apply hysteresis thresholding
			// Note: You may need to use shared memory for intermediate results
			float highThreshold = 150.0f; // Example threshold values
			float lowThreshold = 50.0f;

			if (outputImage[y * width + x] >= highThreshold) {
				outputImage[y * width + x] = 255; // Strong edge
			} else if (outputImage[y * width + x] >= lowThreshold) {
				outputImage[y * width + x] = 127; // Weak edge
			} else {
				outputImage[y * width + x] = 0; // Suppress
			}
		}
    }
}

// Function to save Gnuplot commands to a file
void saveGnuplotScript(const std::string& filename, const std::string& commands) {
    std::ofstream scriptFile(filename);
    scriptFile << commands;
    scriptFile.close();
}

int main() {
    // Load input image
    Mat inputImage = imread("Butterfly.jpg", IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Unable to load input image." << std::endl;
        return 1;
    }

    // Define problem sizes (N) and numbers of processing elements (p)
    std::vector<int> problemSizes = {512, 1024, 2048};
    std::vector<int> processingElements = {1, 2, 4, 8};

    // Data structures to store running times
    std::vector<std::vector<double>> sequentialRunningTimes(problemSizes.size());
    std::vector<std::vector<double>> parallelRunningTimes(problemSizes.size());

    // Perform experiments for different problem sizes and processing elements
    for (int i = 0; i < problemSizes.size(); ++i) {
        int N = problemSizes[i];

        // Resize input image to desired size (N x N)
        Mat resizedInput;
        resize(inputImage, resizedInput, Size(N, N));

        // Sequential Canny edge detection
        auto startSeq = std::chrono::high_resolution_clock::now();
        Mat sequentialOutput;
        sequentialCannyEdgeDetection(resizedInput);
        auto endSeq = std::chrono::high_resolution_clock::now();
        double sequentialTime = std::chrono::duration<double>(endSeq - startSeq).count();

        // Store sequential running time
        sequentialRunningTimes[i].push_back(sequentialTime);

        for (int j = 0; j < processingElements.size(); ++j) {
            int p = processingElements[j];

            // Allocate memory for output image on the host
            Mat parallelOutput(N, N, CV_8UC1);

            // Launch parallel Canny edge detection
            auto startParallel = std::chrono::high_resolution_clock::now();
			parallelCannyEdgeDetection(resizedInput.data, parallelOutput.data, N, N);
			auto endParallel = std::chrono::high_resolution_clock::now();
            double parallelTime = std::chrono::duration<double>(endParallel - startParallel).count();

            // Store parallel running time
            parallelRunningTimes[i].push_back(parallelTime);

            // Save the output image
            std::string outputFilename = "output_image_" + std::to_string(N) + "_p" + std::to_string(p) + ".jpg";
            imwrite(outputFilename, parallelOutput);
        }
    }

    // Calculate speed up and parallel efficiency
    std::vector<std::vector<double>> speedUp(problemSizes.size(), std::vector<double>(processingElements.size(), 0));
    std::vector<std::vector<double>> parallelEfficiency(problemSizes.size(), std::vector<double>(processingElements.size(), 0));

    for (int i = 0; i < problemSizes.size(); ++i) {
        for (int j = 0; j < processingElements.size(); ++j) {
            speedUp[i][j] = sequentialRunningTimes[i][0] / parallelRunningTimes[i][j];
            parallelEfficiency[i][j] = (speedUp[i][j] / processingElements[j]) * 100;
        }
    }

    // Store data to files for Gnuplot
    std::ofstream speedUpFile("speedup.dat");
    for (int i = 0; i < problemSizes.size(); ++i) {
        for (int j = 0; j < processingElements.size(); ++j) {
            speedUpFile << problemSizes[i] << " " << processingElements[j] << " " << speedUp[i][j] << std::endl;
        }
    }
    speedUpFile.close();

    std::ofstream parallelEfficiencyFile("parallelefficiency.dat");
    for (int i = 0; i < problemSizes.size(); ++i) {
        for (int j = 0; j < processingElements.size(); ++j) {
            parallelEfficiencyFile << problemSizes[i] << " " << processingElements[j] << " " << parallelEfficiency[i][j] << std::endl;
        }
    }
    parallelEfficiencyFile.close();

     // Define Gnuplot commands to plot data
    std::string speedUpCommands = R"(
        set terminal png
        set output 'speedup.png'
        set title 'Speed Up vs Number of Processing Elements'
        set xlabel 'Number of Processing Elements (p)'
        set ylabel 'Speed Up'
        set grid
        plot 'speedup.dat' using 1:2:3 with lines
    )";

    std::string parallelEfficiencyCommands = R"(
        set terminal png
        set output 'parallelefficiency.png'
        set title 'Parallel Efficiency vs Number of Processing Elements'
        set xlabel 'Number of Processing Elements (p)'
        set ylabel 'Parallel Efficiency (%)'
        set zlabel 'Problem Size (N)'
        set grid
        splot 'parallelefficiency.dat' using 1:2:3 with lines
    )";

    // Save Gnuplot commands to script files
    saveGnuplotScript("speedup_script.gp", speedUpCommands);
    saveGnuplotScript("parallelefficiency_script.gp", parallelEfficiencyCommands);

    // Execute Gnuplot commands to generate plots
    system("gnuplot speedup_script.gp");
    system("gnuplot parallelefficiency_script.gp");

    return 0;
}
