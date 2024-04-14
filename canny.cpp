#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;

int main() {
    // Read input image
    Mat image = imread("Butterfly.jpg", 0); // Read as grayscale

    if (image.empty()) {
        std::cerr << "Error: Unable to load input image." << std::endl;
        return 1;
    }

    // Gaussian smoothing
    Mat smoothed;
    GaussianBlur(image, smoothed, Size(3, 3), 0, 0);

    // Gradient calculation
    Mat grad_x, grad_y;
    Sobel(smoothed, grad_x, CV_32F, 1, 0);
    Sobel(smoothed, grad_y, CV_32F, 0, 1);
    Mat gradient, angle;
    magnitude(grad_x, grad_y, gradient);
    phase(grad_x, grad_y, angle, true);

    // Non-maximum suppression
    Mat nonMaxSuppressed = Mat::zeros(gradient.size(), CV_32F);
    for (int i = 1; i < gradient.rows - 1; ++i) {
        for (int j = 1; j < gradient.cols - 1; ++j) {
            float q = gradient.at<float>(i, j - 1);
            float r = gradient.at<float>(i, j + 1);
            float s = gradient.at<float>(i - 1, j);
            float t = gradient.at<float>(i + 1, j);
            float current = gradient.at<float>(i, j);
            float angleVal = angle.at<float>(i, j) * (180 / CV_PI);

            if ((angleVal > -22.5 && angleVal <= 22.5) || (angleVal <= -157.5 || angleVal > 157.5)) {
                if (current > q && current > r)
                    nonMaxSuppressed.at<float>(i, j) = current;
            } else if ((angleVal > 22.5 && angleVal <= 67.5) || (angleVal <= -112.5 || angleVal > -157.5)) {
                if (current > s && current > r)
                    nonMaxSuppressed.at<float>(i, j) = current;
            } else if ((angleVal > 67.5 && angleVal <= 112.5) || (angleVal <= -67.5 || angleVal > -112.5)) {
                if (current > s && current > t)
                    nonMaxSuppressed.at<float>(i, j) = current;
            } else if ((angleVal > 112.5 && angleVal <= 157.5) || (angleVal <= -22.5 || angleVal > -67.5)) {
                if (current > q && current > t)
                    nonMaxSuppressed.at<float>(i, j) = current;
            }
        }
    }

    // Hysteresis thresholding
    float lowThreshold = 50;
    float highThreshold = 150;
    Mat strongEdges = Mat::zeros(gradient.size(), CV_8U);
    Mat weakEdges = Mat::zeros(gradient.size(), CV_8U);
    for (int i = 0; i < nonMaxSuppressed.rows; ++i) {
        for (int j = 0; j < nonMaxSuppressed.cols; ++j) {
            if (nonMaxSuppressed.at<float>(i, j) > highThreshold) {
                strongEdges.at<uchar>(i, j) = 255;
            } else if (nonMaxSuppressed.at<float>(i, j) > lowThreshold) {
                weakEdges.at<uchar>(i, j) = 255;
            }
        }
    }
    // Edge extension
    Mat edges;
    bitwise_or(strongEdges, weakEdges, edges);
   // Performance measurement
    auto start = std::chrono::high_resolution_clock::now();
    // Perform Canny edge detection steps
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
    // Save the output image
    imwrite("output.jpg", edges);
    // Display the output image
    std::cout << "Output image saved as output_image.jpg" << std::endl;
    return 0;
}
