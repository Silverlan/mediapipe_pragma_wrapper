/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2023 Silverlan
 */

#define DLLMPW_EX
#include "mediapipe_pragma_wrapper.h"
#pragma warning(disable:4996)
 // Marked as deprecated but is still used internally by mediapipe, so we'll keep using it as well for now...
#include "mediapipe/framework/api2/builder.h"
#pragma warning(default:4996)

#include "absl/flags/flag.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/utils/gate.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/util/resource_util_custom.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include <opencv2/opencv.hpp>
#include <string_view>
#include <fstream>
#include <iostream>

static bool g_initialized = false;
std::shared_ptr< mpw::MotionCaptureManager> mpw::MotionCaptureManager::Create(SourceType type, const Source& source, std::string& outErr)
{
	if (!g_initialized)
	{
		outErr = "mpw has not bee initialized!";
		return nullptr;
	}
	auto manager = std::shared_ptr< MotionCaptureManager>{ new MotionCaptureManager{} };
	manager->m_source = source;
	manager->m_sourceType = type;
	::mediapipe::api2::builder::Graph graph {};
	using TPayload = decltype(graph[::mediapipe::api2::Input<mediapipe::Image>("")]);
	TPayload imgInput = graph[::mediapipe::api2::Input<mediapipe::Image>("IMAGE")];
	imgInput.SetName("image");
	if (!manager->CreateFaceLandmarkerTask(graph, &imgInput, outErr))
	{
		outErr = "Failed to create face landmarker task: " + outErr;
		return nullptr;
	}
	if (!manager->CreatePoseLandmarkerTask(graph, &imgInput, outErr))
	{
		outErr = "Failed to create pose landmarker task: " + outErr;
		return nullptr;
	}
	if (!manager->CreateHandLandmarkerTask(graph, &imgInput, outErr))
	{
		outErr = "Failed to create hand landmarker task: " + outErr;
		return nullptr;
	}

	auto taskRunner = mediapipe::tasks::core::TaskRunner::Create(
		graph.GetConfig(),
		absl::make_unique<mediapipe::tasks::core::MediaPipeBuiltinOpResolver>());

	if (!taskRunner.ok())
	{
		outErr = std::string{ taskRunner.status().message() };
		return nullptr;
	}
	manager->m_taskRunner.taskRunner = std::shared_ptr<mediapipe::tasks::core::TaskRunner>{ std::move(taskRunner.value()) };

	auto result = manager->InitializeSource(outErr);
	if (!result)
		return nullptr;
	manager->InitializeThreads();
	return manager;
}
std::shared_ptr< mpw::MotionCaptureManager> mpw::MotionCaptureManager::CreateFromImage(const std::string& source, std::string& outErr)
{
	return Create(SourceType::Image, source, outErr);
}
std::shared_ptr< mpw::MotionCaptureManager> mpw::MotionCaptureManager::CreateFromVideo(const std::string& source, std::string& outErr)
{
	return Create(SourceType::Video, source, outErr);
}
std::shared_ptr< mpw::MotionCaptureManager> mpw::MotionCaptureManager::CreateFromCamera(CameraDeviceId deviceId, std::string& outErr)
{
	return Create(SourceType::Camera, deviceId, outErr);
}

mpw::MotionCaptureManager::MotionCaptureManager()
	: m_source{ "" }, m_tFrameStart{}
{}

extern absl::Flag<std::string> FLAGS_resource_root_dir;
static const char* g_dataPath = "/";
mpw::MpTaskRunner& mpw::get_mp_task_runner(TaskRunner& taskRunner) { return *static_cast<mpw::MpTaskRunner*>(taskRunner.taskRunner.get()); }
void mpw::init(const char* rootPath, const char* dataPath) {
	g_initialized = true;
	g_dataPath = dataPath;
	/*FLAGS_logtostderr = true;
	FLAGS_log_dir = rootPath;
	google::SetLogDestination(google::GLOG_INFO, (std::string{rootPath} + "/info.log").c_str());
	google::InitGoogleLogging((std::string{rootPath}).c_str());*/
	std::string srootPath = rootPath;
	absl::SetFlag(&FLAGS_resource_root_dir, srootPath);
	/*mediapipe::SetCustomGlobalResourceProvider([srootPath](const std::string& path, std::string* output, bool read_as_binary) {
		auto newPath = srootPath + path;
		return mediapipe::file::GetContents(newPath, output, read_as_binary);
	});*/
}

void mpw::MotionCaptureManager::LockResultData()
{
	m_resultData.resultDataMutex.lock();
}
void mpw::MotionCaptureManager::UnlockResultData()
{
	m_resultData.resultDataMutex.unlock();
}
size_t mpw::MotionCaptureManager::GetBlendShapeCollectionCount() const
{
	return m_resultData.dataSet.blendShapeCoefficientLists.size();
}
bool mpw::MotionCaptureManager::GetBlendShapeCoefficient(size_t collectionIndex, BlendShape blendShape, float& outCoefficient) const
{
	if (collectionIndex >= m_resultData.dataSet.blendShapeCoefficientLists.size())
		return false;
	auto& coefficients = m_resultData.dataSet.blendShapeCoefficientLists[collectionIndex];
	auto iblendShape = static_cast<std::underlying_type_t<BlendShape>>(blendShape);
	if (iblendShape >= coefficients.size())
		return false;
	outCoefficient = coefficients[iblendShape];
	return true;
}
bool mpw::MotionCaptureManager::GetBlendShapeCoefficients(size_t collectionIndex, std::vector<float>& outCoefficients) const
{
	if (collectionIndex >= m_resultData.dataSet.blendShapeCoefficientLists.size())
		return false;
	outCoefficients = m_resultData.dataSet.blendShapeCoefficientLists[collectionIndex];
}
void mpw::MotionCaptureManager::GetBlendShapeCoefficientLists(std::vector<std::vector<float>>& outCoefficientLists) const {
	outCoefficientLists = m_resultData.dataSet.blendShapeCoefficientLists;
}

size_t mpw::MotionCaptureManager::GetPoseCollectionCount() const
{
	return m_resultData.dataSet.poseLandmarkLists.size();
}
bool mpw::MotionCaptureManager::GetPoseWorldLandmark(size_t collectionIndex, PoseLandmark poseLandmark, LandmarkData& outLandmarkData) const
{
	if (collectionIndex >= m_resultData.dataSet.poseLandmarkLists.size())
		return false;
	auto& poses = m_resultData.dataSet.poseLandmarkLists[collectionIndex];
	auto iposeLandmark = static_cast<std::underlying_type_t<PoseLandmark>>(poseLandmark);
	if (iposeLandmark >= poses.size())
		return false;
	outLandmarkData = poses[iposeLandmark];
	return true;
}
bool mpw::MotionCaptureManager::GetPoseWorldLandmarks(size_t collectionIndex, std::vector<LandmarkData>& outLandmarks) const
{
	if (collectionIndex >= m_resultData.dataSet.poseLandmarkLists.size())
		return false;
	outLandmarks = m_resultData.dataSet.poseLandmarkLists[collectionIndex];
}
void mpw::MotionCaptureManager::GetPoseWorldLandmarkLists(std::vector<std::vector<LandmarkData>>& outLandmarks) const {
	outLandmarks = m_resultData.dataSet.poseLandmarkLists;
}

size_t mpw::MotionCaptureManager::GetHandCollectionCount() const
{
	return m_resultData.dataSet.handLandmarkLists.size();
}
bool mpw::MotionCaptureManager::GetHandWorldLandmark(size_t collectionIndex, HandLandmark handLandmark, LandmarkData& outLandmarkData) const
{
	if (collectionIndex >= m_resultData.dataSet.handLandmarkLists.size())
		return false;
	auto& poses = m_resultData.dataSet.handLandmarkLists[collectionIndex];
	auto ihandLandmark = static_cast<std::underlying_type_t<HandLandmark>>(handLandmark);
	if (ihandLandmark >= poses.size())
		return false;
	outLandmarkData = poses[ihandLandmark];
	return true;
}
bool mpw::MotionCaptureManager::GetHandWorldLandmarks(size_t collectionIndex, std::vector<LandmarkData>& outLandmarks) const
{
	if (collectionIndex >= m_resultData.dataSet.handLandmarkLists.size())
		return false;
	outLandmarks = m_resultData.dataSet.handLandmarkLists[collectionIndex];
}
void mpw::MotionCaptureManager::GetHandWorldLandmarkLists(std::vector<std::vector<LandmarkData>>& outLandmarks) const {
	outLandmarks = m_resultData.dataSet.handLandmarkLists;
}

std::optional<std::string> mpw::MotionCaptureManager::GetLastError() const {
	return m_resultData.dataSet
		.errorMessage;
}

// Helper function to construct NormalizeRect proto.
static mediapipe::NormalizedRect MakeNormRect(float x_center, float y_center, float width,
	float height, float rotation) {
	mediapipe::NormalizedRect pose_rect;
	pose_rect.set_x_center(x_center);
	pose_rect.set_y_center(y_center);
	pose_rect.set_width(width);
	pose_rect.set_height(height);
	pose_rect.set_rotation(rotation);
	return pose_rect;
}

bool mpw::MotionCaptureManager::Process(std::string& outErr)
{
	auto& blendShapeCoefficientLists = m_resultData.tmpDataSet.blendShapeCoefficientLists;
	blendShapeCoefficientLists.clear();

	auto& poseLandmarkLists = m_resultData.tmpDataSet.poseLandmarkLists;
	poseLandmarkLists.clear();

	auto& handLandmarkLists = m_resultData.tmpDataSet.handLandmarkLists;
	handLandmarkLists.clear();

	m_resultData.tmpDataSet.errorMessage = {};

	mediapipe::Image* image;
	if (!UpdateFrame(outErr, &image))
		return false;
	// Process blend shapes
	auto& taskRunner = get_mp_task_runner(m_taskRunner);
	auto t = std::chrono::steady_clock::now();
	auto outputPackets = taskRunner.Process(
		{ {"image", mediapipe::MakePacket<mediapipe::Image>(*image)},

		// Documentation states that this input is optional, but that appears to be false
		{"norm_rect",
		mediapipe::MakePacket<mediapipe::NormalizedRect>(MakeNormRect(0.5, 0.5, 1.0, 1.0, 0))}
		});
	auto dt = std::chrono::steady_clock::now() - t;
	if (!outputPackets.ok()) {
		outErr = "Failed to process face landmarker task: " + std::string{outputPackets.status().message()};
		return false;
	}
	m_packetMap = std::move(outputPackets.value());
	auto& packetBlendshapes = m_packetMap["blendshapes"];
	if (!packetBlendshapes.IsEmpty()) {
		auto& packetBlendShapeLists = packetBlendshapes.Get<std::vector<mediapipe::ClassificationList>>();

		blendShapeCoefficientLists.resize(packetBlendShapeLists.size());
		for (auto i = decltype(packetBlendShapeLists.size()){0u}; i < packetBlendShapeLists.size(); ++i) {
			auto& packetBlendShapeList = packetBlendShapeLists[i];
			auto& coefficientList = blendShapeCoefficientLists[i];
			coefficientList.resize(packetBlendShapeList.classification_size());
			for (auto j = decltype(coefficientList.size()){0u}; j < coefficientList.size(); ++j)
				coefficientList[j] = packetBlendShapeList.classification(j).score();
		}
	}

	auto& packetPose = m_packetMap["pose_world_landmarks"];
	if (!packetPose.IsEmpty()) {
		auto& packetPoseLandmarkLists = packetPose.Get<std::vector<mediapipe::LandmarkList>>();

		poseLandmarkLists.resize(packetPoseLandmarkLists.size());
		for (auto i = decltype(packetPoseLandmarkLists.size()){0u}; i < packetPoseLandmarkLists.size(); ++i) {
			auto& packetPoseLandmarkList = packetPoseLandmarkLists[i];
			auto& poseLandmarkList = poseLandmarkLists[i];
			poseLandmarkList.resize(packetPoseLandmarkList.landmark_size());
			for (auto j = decltype(poseLandmarkList.size()){0u}; j < poseLandmarkList.size(); ++j)
			{
				auto& packetPoseLandmark = packetPoseLandmarkList.landmark(j);
				auto& poseLandmark = poseLandmarkList[j];
				assert(packetPoseLandmark.has_x() && packetPoseLandmark.has_y() && packetPoseLandmark.has_z() && packetPoseLandmark.has_presence() && packetPoseLandmark.has_visibility());
				poseLandmark.pos = {
					packetPoseLandmark.x(),
					packetPoseLandmark.y(),
					packetPoseLandmark.z()
				};
				poseLandmark.presence = packetPoseLandmark.presence();
				poseLandmark.visibility = packetPoseLandmark.visibility();
			}
		}
	}

	auto& packetHand = m_packetMap["hand_world_landmarks"];
	if (!packetHand.IsEmpty()) {
		auto& packetHandLandmarkLists = packetHand.Get<std::vector<mediapipe::LandmarkList>>();

		handLandmarkLists.resize(packetHandLandmarkLists.size());
		for (auto i = decltype(packetHandLandmarkLists.size()){0u}; i < packetHandLandmarkLists.size(); ++i) {
			auto& packetHandLandmarkList = packetHandLandmarkLists[i];
			auto& handLandmarkList = handLandmarkLists[i];
			handLandmarkList.resize(packetHandLandmarkList.landmark_size());
			for (auto j = decltype(handLandmarkList.size()){0u}; j < handLandmarkList.size(); ++j)
			{
				auto& packetHandLandmark = packetHandLandmarkList.landmark(j);
				auto& handLandmark = handLandmarkList[j];
				assert(packetHandLandmark.has_x() && packetHandLandmark.has_y() && packetHandLandmark.has_z() && packetHandLandmark.has_presence() && packetHandLandmark.has_visibility());
				handLandmark.pos = {
					packetHandLandmark.x(),
					packetHandLandmark.y(),
					packetHandLandmark.z()
				};
				handLandmark.presence = packetHandLandmark.presence();
				handLandmark.visibility = packetHandLandmark.visibility();
			}
		}
	}

	auto& packetHandedness = m_packetMap["handedness"];
	if (!packetHandedness.IsEmpty()) {
		// m_resultData.handednessList = &packetHandedness.Get<std::vector<mediapipe::ClassificationList>>(); // TODO
	}
	return true;
}

bool mpw::MotionCaptureManager::Start(std::string& outErr)
{
	return StartNextFrame(outErr);
}

bool mpw::MotionCaptureManager::StartNextFrame(std::string& outErr)
{
	WaitForFrame();
	m_firstFrameStarted = true;
	m_frameComplete = false;
	std::unique_lock<std::mutex> lock{m_taskMutex};
	m_hasTask = true;
	m_taskCondition.notify_one();
	return true;
}

bool mpw::MotionCaptureManager::UpdateFrame(std::string& outErr, mediapipe::Image** outImg)
{
	*outImg = nullptr;
	if (!m_inputData)
	{
		outErr = "Invalid input data!";
		return false;
	}
	if (m_sourceType == SourceType::Image) {
		*outImg = static_cast<ImageInputData&>(*m_inputData).image.get();
		return true;
	}
	auto& streamInputData = static_cast<StreamInputData&>(*m_inputData);
	auto& cap = *streamInputData.capture;
	cv::Mat frameBgr;
	if (!cap.read(frameBgr)) {
		outErr = "Failed to read frame.";
		return false;
	}

	// Convert frame from BGR to RGB
	cv::Mat frameRgb;
	cv::cvtColor(frameBgr, frameRgb, cv::COLOR_BGR2RGB);

	auto imageFormat = mediapipe::ImageFormat::SRGB;
	auto* data = frameRgb.data;
	auto width = frameRgb.cols;
	auto height = frameRgb.rows;

	if (data == nullptr) {
		outErr = "Invalid frame input data.";
		return false;
	}
	if (!mediapipe::ImageFormat::Format_IsValid(imageFormat)) {
		outErr = "Invalid image format: " + std::to_string(imageFormat);
		return false;
	}

	auto input_frame_for_input = std::make_shared<mediapipe::ImageFrame>();
	auto mp_image_format = static_cast<mediapipe::ImageFormat::Format>(imageFormat);
	input_frame_for_input->CopyPixelData(mp_image_format, width, height, data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

	auto& img = *static_cast<StreamInputData&>(*m_inputData).currentFrameImage;
	img = { input_frame_for_input };
	*outImg = &img;
	return true;
}

bool mpw::MotionCaptureManager::InitializeSource(std::string& outErr) {
	//# copy_mp_files(mediapipe_pragma_wrapper_root + "/mediapipe", mediapipe_root + "/mediapipe")
	//	set_num_poses
	m_inputData = nullptr;
	if (m_sourceType == SourceType::Image) {
		auto imgInputData = std::make_unique< ImageInputData>();
		auto image = mediapipe::tasks::vision::DecodeImageFromFile(std::get<std::string>(m_source));
		if (!image.ok())
		{
			outErr = "Failed to decode image: " + std::string{image.status().message()};
			return false;
		}
		imgInputData->image = std::make_shared<mediapipe::Image>(std::move(*image));
		m_inputData = std::move(imgInputData);
		return true;
	}

	auto streamInputData = std::make_unique< StreamInputData>();
	streamInputData->capture = std::make_shared<cv::VideoCapture>();
	streamInputData->currentFrameImage = std::make_shared<mediapipe::Image>();
	auto& cap = *streamInputData->capture;
	if (m_sourceType == SourceType::Video)
	{
		auto& videoPath = std::get<std::string>(m_source);
		try
		{
			cap.open(videoPath);
		}
		catch (const cv::Exception& e) {
			outErr = "Failed to open video '" + videoPath + "': " + std::string{e.what()};
			return false;
		}
		if (!cap.isOpened()) {
			outErr = "Failed to open video '" + videoPath + "'.";
			return false;
		}
	}
	else
	{
		auto camId = std::get<mpw::CameraDeviceId>(m_source);
		try
		{
			cap.open(camId);
		}
		catch (const cv::Exception& e) {
			outErr = "Failed to open camera device " + std::to_string(camId) + ": " + std::string{e.what()};
			return false;
		}
		if (!cap.isOpened()) {
			outErr = "Failed to open camera device " + std::to_string(camId) + ".";
			return false;
		}
	}

	m_inputData = std::move(streamInputData);
	return true;
}

bool mpw::MotionCaptureManager::CreateFaceLandmarkerTask(::mediapipe::api2::builder::Graph& graph, void* pimgInput, std::string& outErr) {
	using TPayload = decltype(graph[::mediapipe::api2::Input<mediapipe::Image>("")]);
	auto& imgInput = *static_cast<TPayload*>(pimgInput);

	auto& faceLandmarker = graph.AddNode("mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph");

	auto* options = &faceLandmarker.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>();
	options->mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", g_dataPath, m_faceLandmarkerModel));
	options->mutable_face_detector_graph_options()->set_num_faces(1);
	options->mutable_base_options()->set_use_stream_mode(true);

	imgInput >>
		faceLandmarker.In("IMAGE");

	faceLandmarker.Out("BLENDSHAPES").SetName("blendshapes") >>
		graph[::mediapipe::api2::Output< std::vector<mediapipe::ClassificationList>>("BLENDSHAPES")];
	return true;
}
bool mpw::MotionCaptureManager::CreatePoseLandmarkerTask(::mediapipe::api2::builder::Graph& graph, void* pimgInput, std::string& outErr) {
	using TPayload = decltype(graph[::mediapipe::api2::Input<mediapipe::Image>("")]);
	auto& imgInput = *static_cast<TPayload*>(pimgInput);

	auto& poseLandmarker = graph.AddNode(
		"mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph");

	auto* options = &poseLandmarker.GetOptions<mediapipe::tasks::vision::pose_landmarker::proto::PoseLandmarkerGraphOptions>();
	options->mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", g_dataPath, m_poseLandmarkerModel));
	options->mutable_pose_detector_graph_options()->set_num_poses(1);
	options->mutable_base_options()->set_use_stream_mode(true);

	imgInput >>
		poseLandmarker.In("IMAGE");
	graph[::mediapipe::api2::Input<mediapipe::NormalizedRect>("NORM_RECT")].SetName("norm_rect") >>
		poseLandmarker.In("NORM_RECT");

	poseLandmarker.Out("WORLD_LANDMARKS").SetName("pose_world_landmarks") >>
		graph[::mediapipe::api2::Output< std::vector<mediapipe::LandmarkList>>("POSE_WORLD_LANDMARKS")];
	return true;
}
bool mpw::MotionCaptureManager::CreateHandLandmarkerTask(::mediapipe::api2::builder::Graph& graph, void* pimgInput, std::string& outErr) {
	using TPayload = decltype(graph[::mediapipe::api2::Input<mediapipe::Image>("")]);
	auto& imgInput = *static_cast<TPayload*>(pimgInput);

	auto& handLandmarker = graph.AddNode(
		"mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph");

	auto* options = &handLandmarker.GetOptions<mediapipe::tasks::vision::hand_landmarker::proto::HandLandmarkerGraphOptions>();
	options->mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", g_dataPath, m_handLandmarkerModel));
	options->mutable_hand_detector_graph_options()->set_num_hands(2);
	options->mutable_base_options()->set_use_stream_mode(true);

	imgInput >>
		handLandmarker.In("IMAGE");
	handLandmarker.Out("WORLD_LANDMARKS").SetName("hand_world_landmarks") >>
		graph[::mediapipe::api2::Output<std::vector<mediapipe::LandmarkList>>("HAND_WORLD_LANDMARKS")];
	handLandmarker.Out("HANDEDNESS").SetName("handedness") >>
		graph[::mediapipe::api2::Output< std::vector<mediapipe::ClassificationList>>("HANDEDNESS")];
	return true;
}
void mpw::MotionCaptureManager::InitializeThreads()
{
	m_mainThread = std::thread{ [this]() {
		while (m_running) {
			{
				std::unique_lock<std::mutex> lock{m_taskMutex};
				m_taskCondition.wait(lock, [&]() {
					return !m_running || m_hasTask;
					});

				if (!m_running)
					break;

				m_tFrameStart = std::chrono::steady_clock::now();
				std::string err;
				auto result = Process(err);
				if (!result)
					m_resultData.tmpDataSet.errorMessage = std::move(err);
				else
					m_resultData.tmpDataSet.errorMessage = {};

				auto dt = std::chrono::steady_clock::now() - m_tFrameStart;

				std::unique_lock<std::mutex> lockComplete{m_frameCompleteMutex};
				m_resultData.resultDataMutex.lock();
				m_hasTask = false;
				m_frameComplete = true;
				m_resultData.dataSet.blendShapeCoefficientLists = std::move(m_resultData.tmpDataSet.blendShapeCoefficientLists);
				m_resultData.dataSet.poseLandmarkLists = std::move(m_resultData.tmpDataSet.poseLandmarkLists);
				m_resultData.dataSet.handLandmarkLists = std::move(m_resultData.tmpDataSet.handLandmarkLists);
				m_resultData.dataSet.errorMessage = std::move(m_resultData.tmpDataSet.errorMessage);
				++m_resultData.frameIndex;
				m_resultData.resultDataMutex.unlock();
				m_frameCompleteCondition.notify_all();
			}
			if (!m_running)
				break;
			if (m_autoAdvance) {
				std::string err;
				Process(err);
			}
		}
	} };
}

bool mpw::MotionCaptureManager::IsFrameComplete() const
{
	return m_frameComplete;
}
void mpw::MotionCaptureManager::WaitForFrame()
{
	if (!m_firstFrameStarted)
		return;
	std::unique_lock<std::mutex> lock{m_frameCompleteMutex};
	m_frameCompleteCondition.wait(lock, [&]() {
		return !m_running || m_frameComplete;
		});
}
mpw::MotionCaptureManager::~MotionCaptureManager() {
	m_running = false;
	m_taskCondition.notify_all();
	m_frameCompleteCondition.notify_all();
	m_mainThread.join();
}
