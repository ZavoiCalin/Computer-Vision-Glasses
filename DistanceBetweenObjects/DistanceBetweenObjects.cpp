#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>

// Function to load video and execute depth inference

int loadVideo(Ort::Session& session) {
    std::string videoPath = "Resources\\challenge_color_848x480.mp4"; // path to your MP4 file
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame; // read the next frame
        if (frame.empty()) break; // end of video

        // preprocessing the frame

        cv::Mat img;
        cv::resize(frame, img, cv::Size(518, 518));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1.0 / 255.0);

        // convert HWC to NCHW
        std::vector<float> input(1 * 3 * 518 * 518);
        std::vector<int64_t> shape = { 1, 3, 518, 518 };

        cv::Mat channels[3];
        cv::split(img, channels);

        int hw = 518 * 518;

        for (int i = 0; i < 3; i++) {
            memcpy(input.data() + i * hw,
                channels[i].data,
                hw * sizeof(float)
            );
        }

        // ONNX runtime inference
        Ort::MemoryInfo mem =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem, input.data(), 
            input.size(), 
            shape.data(), 
            shape.size()
        );


        const char* input_names[] = { "input" };
        const char* output_names[] = { "depth" };

        auto output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_names, &input_tensor, 1, output_names, 1
        );

        // process the depth output

        float* depth_ptr = output_tensors[0].GetTensorMutableData<float>();
        cv::Mat depth(518, 518, CV_32F, depth_ptr);

        cv::Mat depth_vis;
        //cv::normalize(depth, depth_vis, 0, 255, cv::NORM_INF);
        cv::normalize(depth, depth_vis, 0, 255, cv::NORM_MINMAX);
        depth_vis.convertTo(depth_vis, CV_8U);


        //cv::imshow("Video", frame);

        cv::imshow("Depth", depth_vis);

        // Wait 30 ms or until ESC key is pressedWW
        if (cv::waitKey(30) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();

	return 0;
}

void printCWD() {
    std::cout << "Current working directory: "
        << std::filesystem::current_path() << std::endl;
}

int main() {
    
    //loadVideo(); 
    //printCWD();

    // ONNX Runtime Init

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DepthAI");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    // enable CUDA support 

    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    cuda_options.gpu_mem_limit = 512 * 1024 * 1024; // GTX 950M GPU
    session_options.AppendExecutionProvider_CUDA(cuda_options);



    Ort::Session session(
        env,
        L"models/depth_anything_v2_vits.onnx",
        session_options
    );

    

    // run the ONNX session

    loadVideo(session);

    return 0;
}
