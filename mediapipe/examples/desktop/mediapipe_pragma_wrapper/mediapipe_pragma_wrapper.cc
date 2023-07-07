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
	if (!manager->CreateFaceLandmarkerTask(outErr))
	{
		outErr = "Failed to create face landmarker task: " + outErr;
		return nullptr;
	}
	if (!manager->CreatePoseLandmarkerTask(outErr))
	{
		outErr = "Failed to create pose landmarker task: " + outErr;
		return nullptr;
	}
	if (!manager->CreateHandLandmarkerTask(outErr))
	{
		outErr = "Failed to create hand landmarker task: " + outErr;
		return nullptr;
	}
	auto result = manager->Start(outErr);
	if (!result)
		return nullptr;
	return manager;
}
std::shared_ptr< mpw::MotionCaptureManager> mpw::MotionCaptureManager::CreateFromImage(const std::string& source, std::string& outErr)
{
	return Create(SourceType::Image, source,outErr);
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
	: m_source{ "" }
{}

extern absl::Flag<std::string> FLAGS_resource_root_dir;
static const char *g_dataPath = "/";
mpw::MpTaskRunner& mpw::get_mp_task_runner(TaskRunner& taskRunner) { return *static_cast<mpw::MpTaskRunner*>(taskRunner.taskRunner.get()); }
void mpw::init(const char* rootPath, const char* dataPath) {
	g_initialized = true;
	g_dataPath = dataPath;
	FLAGS_logtostderr = true;
	FLAGS_log_dir = rootPath;
	google::SetLogDestination(google::GLOG_INFO, (std::string{rootPath} + "/info.log").c_str());
	google::InitGoogleLogging((std::string{rootPath} + "/test_mediapipe_wrapper.exe").c_str());
	//google::SetLogDestination(ERROR, (std::string{rootPath} + "/error.log").c_str());
	//google::SetLogDestination(WARN, (std::string{rootPath} + "/warn.log").c_str());
	std::string srootPath = rootPath;
	absl::SetFlag(&FLAGS_resource_root_dir, srootPath);
	/*mediapipe::SetCustomGlobalResourceProvider([srootPath](const std::string& path, std::string* output, bool read_as_binary) {
		auto newPath = srootPath + path;
		return mediapipe::file::GetContents(newPath, output, read_as_binary);
	});*/
}

size_t mpw::MotionCaptureManager::GetBlendShapeCollectionCount() const
{
	if (!m_blendShapeResult.collections)
		return 0;
	return m_blendShapeResult.collections->size();
}
bool mpw::MotionCaptureManager::GetBlendShapeCoefficient(size_t collectionIndex, BlendShape blendShape, float& outCoefficient)
{
	if (collectionIndex >= m_blendShapeResult.collections->size())
		return false;
	auto& coefficients = (*m_blendShapeResult.collections)[collectionIndex];
	auto iblendShape = static_cast<std::underlying_type_t<BlendShape>>(blendShape);
	if (iblendShape >= coefficients.classification_size())
		return false;
	outCoefficient = coefficients.classification(iblendShape).score();
	return true;
}

size_t mpw::MotionCaptureManager::GetPoseCollectionCount() const
{
	if (!m_poseResult.landmarkLists)
		return 0;
	return m_poseResult.landmarkLists->size();
}
bool mpw::MotionCaptureManager::GetPoseWorldLandmarkPosition(size_t collectionIndex, PoseLandmark poseLandmark, std::array<float, 3>& outPosition, float& outPresence, float& outVisibility)
{
	if (collectionIndex >= m_poseResult.landmarkLists->size())
		return false;
	auto& poses = (*m_poseResult.landmarkLists)[collectionIndex];
	auto iposeLandmark = static_cast<std::underlying_type_t<PoseLandmark>>(poseLandmark);
	if (iposeLandmark >= poses.landmark_size())
		return false;
	auto& landmark = poses.landmark(iposeLandmark);
	assert(landmark.has_x() && landmark.has_y() && landmark.has_z() && landmark.has_presence() && landmark.has_visibility());
	outPosition = {
		landmark.x(),
		landmark.y(),
		landmark.z()
	};
	outPresence = landmark.presence();
	outVisibility = landmark.visibility();
	return true;
}

size_t mpw::MotionCaptureManager::GetHandCollectionCount() const
{
	if (!m_poseResult.landmarkLists)
		return 0;
	return m_poseResult.landmarkLists->size();
}
bool mpw::MotionCaptureManager::GetHandWorldLandmarkPosition(size_t collectionIndex, HandLandmark handLandmark, std::array<float, 3>& outPosition, float& outPresence, float& outVisibility)
{
	if (collectionIndex >= m_poseResult.landmarkLists->size())
		return false;
	auto& poses = (*m_poseResult.landmarkLists)[collectionIndex];
	auto ihandLandmark = static_cast<std::underlying_type_t<HandLandmark>>(handLandmark);
	if (ihandLandmark >= poses.landmark_size())
		return false;
	auto& landmark = poses.landmark(ihandLandmark);
	assert(landmark.has_x() && landmark.has_y() && landmark.has_z() && landmark.has_presence() && landmark.has_visibility());
	outPosition = {
		landmark.x(),
		landmark.y(),
		landmark.z()
	};
	outPresence = landmark.presence();
	outVisibility = landmark.visibility();
	return true;
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

bool mpw::MotionCaptureManager::ProcessImage(mediapipe::Image& image, std::string& outErr)
{
	{
		// Process blend shapes
		auto& taskRunner = get_mp_task_runner(m_faceLandmarker.taskRunner);
		auto outputPackets = taskRunner.Process(
			{ {"image", mediapipe::MakePacket<mediapipe::Image>(image)}
			});
		if (!outputPackets.ok()) {
			outErr = "Failed to process face landmarker task: " + std::string{outputPackets.status().message()};
			return false;
		}
		m_blendShapeResult.packetMap = std::move(outputPackets.value());
		m_blendShapeResult.collections = &m_blendShapeResult.packetMap["blendshapes"].Get<std::vector<mediapipe::ClassificationList>>();
	}
	{
		// Process pose
		auto& taskRunner = get_mp_task_runner(m_poseLandmarker.taskRunner);

		auto outputPackets = taskRunner.Process(
			{ {"image", mediapipe::MakePacket<mediapipe::Image>(image)},

			// Documentation states that this input is optional, but that appears to be false
   {"norm_rect",
			mediapipe::MakePacket<mediapipe::NormalizedRect>(MakeNormRect(0.5, 0.5, 1.0, 1.0, 0))}


			});
		if (!outputPackets.ok()) {
			outErr = "Failed to process pose landmarker task: " + std::string{outputPackets.status().message()};
			return false;
		}
		m_poseResult.packetMap = std::move(outputPackets.value());
		m_poseResult.landmarkLists = &m_poseResult.packetMap["world_landmarks"].Get<std::vector<mediapipe::LandmarkList>>();
	}
	{
		// Process hands
		auto& taskRunner = get_mp_task_runner(m_poseLandmarker.taskRunner);

		auto outputPackets = taskRunner.Process(
			{ {"image", mediapipe::MakePacket<mediapipe::Image>(image)},

			// Documentation states that this input is optional, but that appears to be false
   {"norm_rect",
			mediapipe::MakePacket<mediapipe::NormalizedRect>(MakeNormRect(0.5, 0.5, 1.0, 1.0, 0))}


			});
		if (!outputPackets.ok()) {
			outErr = "Failed to process pose landmarker task: " + std::string{outputPackets.status().message()};
			return false;
		}
		m_handResult.packetMap = std::move(outputPackets.value());
		m_handResult.landmarkLists = &m_handResult.packetMap["world_landmarks"].Get<std::vector<mediapipe::LandmarkList>>();
		// m_handResult.handednessList = &m_handResult.packetMap["handedness"].Get<std::vector<mediapipe::ClassificationList>>();
	}
	return true;
}

bool mpw::MotionCaptureManager::ProcessNextFrame(std::string& outErr)
{
	if (!m_inputData)
	{
		outErr = "Invalid input data!";
		return false;
	}
	if (m_sourceType == SourceType::Image)
		return ProcessImage(*static_cast<ImageInputData&>(*m_inputData).image,outErr);

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

	mediapipe::Image img { input_frame_for_input };
	return ProcessImage(img, outErr);
}

bool mpw::MotionCaptureManager::Start(std::string& outErr) {
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
	auto& cap = *streamInputData->capture;
	if (m_sourceType == SourceType::Video)
	{
		auto& videoPath = std::get<std::string>(m_source);
		try
		{
			cap.open(videoPath);
		}
		catch (const cv::Exception& e) {
			outErr = "Failed to open video '" + videoPath +"': " + std::string{e.what()};
			return false;
		}
		if (!cap.isOpened()) {
			outErr = "Failed to open video '" + videoPath +"'.";
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
			outErr = "Failed to open camera device " +std::to_string(camId) +": " + std::string{e.what()};
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

bool mpw::MotionCaptureManager::CreateFaceLandmarkerTask(std::string& outErr) {
	::mediapipe::api2::builder::Graph graph {};
	auto& faceLandmarker = graph.AddNode("mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph");

	auto* options = &faceLandmarker.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>();
	options->mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", g_dataPath, m_faceLandmarker.modelName));
	options->mutable_face_detector_graph_options()->set_num_faces(1);
	options->mutable_base_options()->set_use_stream_mode(true);

	graph[::mediapipe::api2::Input<mediapipe::Image>("IMAGE")].SetName("image") >>
		faceLandmarker.In("IMAGE");

	faceLandmarker.Out("BLENDSHAPES").SetName("blendshapes") >>
		graph[::mediapipe::api2::Output< std::vector<mediapipe::ClassificationList>>("BLENDSHAPES")];

	auto taskRunner = mediapipe::tasks::core::TaskRunner::Create(
		graph.GetConfig(),
		absl::make_unique<mediapipe::tasks::core::MediaPipeBuiltinOpResolver>());

	if (!taskRunner.ok())
	{
		outErr = std::string{taskRunner.status().message()};
		return false;
	}

	m_faceLandmarker.taskRunner.taskRunner = std::shared_ptr<mediapipe::tasks::core::TaskRunner>{ std::move(taskRunner.value()) };
	return taskRunner.ok();
}
bool mpw::MotionCaptureManager::CreatePoseLandmarkerTask(std::string& outErr) {
	::mediapipe::api2::builder::Graph graph {};

	auto& poseLandmarker = graph.AddNode(
		"mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph");

	auto* options = &poseLandmarker.GetOptions<mediapipe::tasks::vision::pose_landmarker::proto::PoseLandmarkerGraphOptions>();
	options->mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", g_dataPath, m_poseLandmarker.modelName));
	options->mutable_pose_detector_graph_options()->set_num_poses(1);
	options->mutable_base_options()->set_use_stream_mode(true);

	graph[::mediapipe::api2::Input<mediapipe::Image>("IMAGE")].SetName("image") >>
		poseLandmarker.In("IMAGE");
	graph[::mediapipe::api2::Input<mediapipe::NormalizedRect>("NORM_RECT")].SetName("norm_rect") >>
		poseLandmarker.In("NORM_RECT");

	poseLandmarker.Out("WORLD_LANDMARKS").SetName("world_landmarks") >>
		graph[::mediapipe::api2::Output< std::vector<mediapipe::LandmarkList>>("WORLD_LANDMARKS")];

	auto taskRunner = mediapipe::tasks::core::TaskRunner::Create(
		graph.GetConfig(),
		absl::make_unique<mediapipe::tasks::core::MediaPipeBuiltinOpResolver>());
	if (!taskRunner.ok())
	{
		outErr = std::string{taskRunner.status().message()};
		return false;
	}

	m_poseLandmarker.taskRunner.taskRunner = std::shared_ptr<mediapipe::tasks::core::TaskRunner>{ std::move(taskRunner.value()) };
	return taskRunner.ok();
}
bool mpw::MotionCaptureManager::CreateHandLandmarkerTask(std::string& outErr) {
	::mediapipe::api2::builder::Graph graph {};

	auto& handLandmarker = graph.AddNode(
		"mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph");

	auto* options = &handLandmarker.GetOptions<mediapipe::tasks::vision::hand_landmarker::proto::HandLandmarkerGraphOptions>();
	options->mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", g_dataPath, m_handLandmarker.modelName));
	options->mutable_hand_detector_graph_options()->set_num_hands(2);
	options->mutable_base_options()->set_use_stream_mode(true);

	graph[::mediapipe::api2::Input<mediapipe::Image>("IMAGE")].SetName("image") >>
		handLandmarker.In("IMAGE");

	handLandmarker.Out("WORLD_LANDMARKS").SetName("world_landmarks") >>
		graph[::mediapipe::api2::Output<std::vector<mediapipe::LandmarkList>>("WORLD_LANDMARKS")];
	handLandmarker.Out("HANDEDNESS").SetName("handedness") >>
		graph[::mediapipe::api2::Output< std::vector<mediapipe::ClassificationList>>("HANDEDNESS")];

	auto taskRunner = mediapipe::tasks::core::TaskRunner::Create(
		graph.GetConfig(),
		absl::make_unique<mediapipe::tasks::core::MediaPipeBuiltinOpResolver>());

	if (!taskRunner.ok())
	{
		outErr = std::string{taskRunner.status().message()};
		return false;
	}

	m_handLandmarker.taskRunner.taskRunner = std::shared_ptr<mediapipe::tasks::core::TaskRunner>{ std::move(taskRunner.value()) };
	return taskRunner.ok();
}
