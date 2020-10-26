#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace dnn;
using namespace std;


int main(int argc, char** argv)
{
    // Parse command line arguments
    String model_name=argv[1];
    String background_path=argv[2]; 
    String mask_edge=argv[3];

    // Configure model and backend settings
    int backendId = 0; // Auto
    int targetId = 1;  // OpenCL
    int inpWidth = 320;
    int inpHeight = 320;

    // Read the onnx segmentation model
    Net net = readNetFromONNX(model_name);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);

    // Create a window for display
    static const string segwin = "SINet Portrait Segmentation";
    namedWindow(segwin, WINDOW_NORMAL);

    // Capture frames from camera
    VideoCapture cap;
    cap.open(0); // Camera
    
    // Load and preprocess the background image
    Mat background = imread(background_path, IMREAD_COLOR);
    resize(background, background, Size(320, 320));
    background.convertTo(background, CV_32FC3);
    Mat bg=background.clone();
    Mat frame;

    // Process video frames
    Mat blob;
    while (waitKey(1) < 0)
    {
        // Read frames from camera
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }
        
        // Resize the input frames
        resize(frame, frame, Size(320, 320));
        frame.convertTo(frame, CV_32FC3);
        Mat orig_frame = frame.clone();
        
        // Normalize the image using mean and standard deviation
        subtract(frame, Scalar(102.890434, 111.25247, 126.91212), frame);
        multiply(frame, Scalar(1.0 / 62.93292,  1.0 / 62.82138, 1.0 / 66.355705), frame);
 
        // Convert image to blob and set the network input
        blobFromImage(frame, blob, 1.0/255.0, Size(inpWidth, inpHeight));
        net.setInput(blob);

        //Run the network and retrieve the outputs
        Mat score = net.forward();
        Mat bgval = Mat(score.size[2], score.size[3], CV_32F, score.data);
         
        // Generate background mask and crop background image
        Mat bgcrop;
        Mat bgmask = Mat(320, 320, CV_32FC3, Scalar(0.0,0.0,0.0));
 
        // Choose edge smoothing for alpha blending
        if ( mask_edge == "smooth" ) {
           cvtColor(bgval, bgval, COLOR_GRAY2BGR);
           bgmask=bgval; }
        else
           bgmask.setTo(1.0, bgval>0.5);
        multiply(bgmask, background, bgcrop);

        // Generate foreground mask and crop foreground image
        Mat fgcrop;
        Mat fgmask = Mat(320, 320, CV_32FC3, Scalar(1.0,1.0,1.0));
        subtract(fgmask, bgmask, fgmask);
        multiply(fgmask, orig_frame, fgcrop);

        // Add cropped fg and bg regions
        Mat result;
        add(bgcrop,fgcrop,result);
        result.convertTo(result, CV_8UC3);

        // Put efficiency information.
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time: %.2f ms", t);
        putText(result, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
 
        // Display output in window
        imshow(segwin, result);
    }
    return 0;
}

/**
Compile:  g++ sinet.cpp `pkg-config opencv --cflags --libs`
Run: ./a.out SINet_Softmax.onnx whitehouse.jpeg smooth
Args => model_path background_path mask_edge
Tested: Ubuntu 18.04, OpenCV 4.5
**/
