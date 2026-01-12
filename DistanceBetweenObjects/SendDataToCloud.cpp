#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <chrono>
#include <mutex>
#include <thread>
#include <atomic>
#include <curl/curl.h>
#include <sstream>


struct ObjectDistance {
	int object_id;
	float distance;
};

struct FrameResult {
	uint64_t timestamp_ms;
	std::vector<ObjectDistance> objects;
};

std::vector<FrameResult> resultBuffer;
std::mutex bufferMutex;

float computeMeanDepth(const cv::Mat& depth, const cv::Rect& roi) {
	cv::Mat region = depth(roi);

	return static_cast<float>(cv::mean(region)[0]);
}

void uploadThread(std::atomic<bool>& running) {
	while (running) {
		std::this_thread::sleep_for(std::chrono::seconds(5));

		std::vector<FrameResult> batch;
		{
			std::lock_guard<std::mutex> lock(bufferMutex);
			batch.swap(resultBuffer);
		}

		if (!batch.empty()) {
			std::cout << "Uploading " << batch.size() << " measurements" << std::endl;

		}

		std::string json = serializeToJson(batch);
	
		bool sent = postJsonToServer(json);

		if (sent) {
			std::cout << batch.size();
		}
	}

	curl_global_cleanup();
}



std::string serializeToJson(const std::vector<FrameResult>& batch) {
	std::ostringstream oss;

	for (size_t i = 0; i < batch.size(); ++i) {
		const auto& frame = batch[i];

		oss << "{";
		oss << "\"timestamp_ms\": " << frame.timestamp_ms << ",";
		oss << "\"objects\": [";


		// obj sep
		if (j + 1 < frame.objects.size())
			oss << ",";
	}

	oss << "] }";

	if (i + 1 < batch.size())
		oss << ",";

	oss << "] }";

	std::string payload = oss.str();

	return payload;
}



bool postJsonToServer(const std::string& jsonPayload) {
	CURL* curl = curl_easy_init();

	if (!curl) return false;

	struct curl_slist* headers = nullptr;
	headers = curl_slist_append(headers, "Content-Type: application/json");

	curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8000/ingest");
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonPayload.c_str());
	curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, jsonPayload.size());
	curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 5000L);

	CURLcode code = curl_easy_perform(curl);

	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);

	if (code != CURLE_OK) {
		std::cerr << "Post failed with code " << curl_easy_strerror(code) << std::endl;

		return false;
	}

	return true;
}


int loadVideo(Ort::Session& session) {
	std::string videoPath = "Resources\\challege_color_848x480.mp4";
	cv::VideoCapture cap(videoPath);

	if (!cap.isOpened()) {
		std::cerr << "Error: Could not open vido file" << videoPath << std::endl;
	}

	cv::Mat frame;

	while (true) {
		cap >> frame;

		if (frame.empty()) break;

		// timestamp generated based on chrono lib

		uint64_t timestamp_ms =
			std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::steady_clock::now().time_since_epoch)
			).count();

		// frame preprocessing

		cv::Mat img;
		cv::resize(frame, img, cv::Size(518, 518));
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		img.convertTo(img, CV_32F, 1.0 / 255.0);

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

		// ONNX Runtime inference

		Ort::MemoryInfo mem =
			Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		Ort::Value input_tensor = Ort::Value::CreateTensr<float>(
			mem, input.data(), input.size(),
			shape.data(), shape.size()
		);

		const char* input_names[] = { "input" };
		const char* output_names[] = { "depth" };

		auto output_tensors = session.Run(
			Ort::RunOptions{ nullptr },
			input_names, &input_tensor, 1,
			output_names, 1
		);

		float* depth_ptr = output_tensors[0].GetTensorMutableData<float>();
		cv::Mat depth(518, 518, CV_32F, depth_ptr);

		// distance computation using as ROI the center of image

		cv::Rect roi(200, 200, 100, 100);
		float distance = computeMeanDepth(depth, roi);

		// store results for the cloud upload

		{
			std::lock_guard<std::mutex> lock(bufferMutex);
			resultBuffer.push_back({
				timestamp_ms,
				{ {0, distance} }
			});
		}

		// add visual rect marker

		cv::Mat depth_vis;
		cv::normalize(depth, depth_vis, 0, 255, cv::NORM_MINMAX);
		depth_vis.convertTo(depth_vis, CV_8U);

		cv::rectangle(depth_vis, roi, cv::Scalar(255), 2);
		cv::imshow("Depth", depth_vis);

		if (cv::waitKey(30) == 27) break;
	}

	cap.release();
	cv::destroyAllWindows();
	
	return 0;
}


int main() {

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DepthAI");

	Ort >> SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(
		GraphOptimizationLevel::ORT_ENABLE_ALL
	);

	OrtCUDAProviderOptions cuda_options{};
	cuda_options.device_id = 0;
	cuda_options.gpu_mem_limit = 512 * 1024 * 1024;
	session_options.AppendExecutionProvider_CUDA(cuda_options);

	Ort::Session session(
		env,
		L"models/depth_anything_v2_vits.onnx",
		session_options
	);

	std::atomic<bool> running{ true };

	std::thread uploader(uploadThread, std::ref(running));

	loadVideo(session);

	running = false;
	uploader.join();

	return 0;
}