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
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#include "mediapipe/tasks/cc/components/utils/gate.h"
#include "mediapipe/calculators/util/landmarks_smoothing_calculator.pb.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/util/resource_util_custom.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string_view>
#include <fstream>
#include <iostream>

static bool g_initialized = false;
std::shared_ptr< mpw::MotionCaptureManager> mpw::MotionCaptureManager::Create(SourceType type, const Source& source, std::string& outErr, Output enabledOutputs)
{
	if (!g_initialized)
	{
		outErr = "mpw has not bee initialized!";
		return nullptr;
	}
	auto manager = std::shared_ptr< MotionCaptureManager>{ new MotionCaptureManager{} };
	manager->m_source = source;
	manager->m_sourceType = type;
	manager->m_enabledOutputs = enabledOutputs;
	::mediapipe::api2::builder::Graph graph {};
	auto imgInput = graph[::mediapipe::api2::Input<mediapipe::Image>("IMAGE")];
	imgInput.SetName("image");

	auto trackingIdInput = graph[::mediapipe::api2::Input<std::vector<int64_t>>("TRACKING_IDS")];
	trackingIdInput.SetName("tracking_ids");

	if (!manager->CreateFaceLandmarkerTask(graph, imgInput, outErr))
	{
		outErr = "Failed to create face landmarker task: " + outErr;
		return nullptr;
	}
	if (!manager->CreatePoseLandmarkerTask(graph, imgInput, trackingIdInput, outErr))
	{
		outErr = "Failed to create pose landmarker task: " + outErr;
		return nullptr;
	}
	if (!manager->CreateHandLandmarkerTask(graph, imgInput, trackingIdInput, outErr))
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
std::shared_ptr< mpw::MotionCaptureManager> mpw::MotionCaptureManager::CreateFromImage(const std::string& source, std::string& outErr, Output enabledOutputs)
{
	return Create(SourceType::Image, source, outErr,enabledOutputs);
}
std::shared_ptr< mpw::MotionCaptureManager> mpw::MotionCaptureManager::CreateFromVideo(const std::string& source, std::string& outErr, Output enabledOutputs)
{
	return Create(SourceType::Video, source, outErr, enabledOutputs);
}
std::shared_ptr< mpw::MotionCaptureManager> mpw::MotionCaptureManager::CreateFromCamera(CameraDeviceId deviceId, std::string& outErr, Output enabledOutputs)
{
	return Create(SourceType::Camera, deviceId, outErr, enabledOutputs);
}

mpw::MotionCaptureManager::MotionCaptureManager()
	: m_source{ "" }, m_tFrameStart{}, m_enabledOutputs{Output::Default}
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
size_t mpw::MotionCaptureManager::GetFaceGeometryCount() const
{
	return m_resultData.dataSet.faceGeometries.size();
}
bool mpw::MotionCaptureManager::GetFaceGeometry(size_t index, MeshData& outMeshData) const
{
	if (index >= m_resultData.dataSet.faceGeometries.size())
		return false;
	outMeshData = m_resultData.dataSet.faceGeometries[index];
	return true;
}
const mpw::MeshData* mpw::MotionCaptureManager::GetFaceGeometry(size_t index) const
{
	if (index >= m_resultData.dataSet.faceGeometries.size())
		return nullptr;
	return &m_resultData.dataSet.faceGeometries[index];
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

	auto& faceGeometries = m_resultData.tmpDataSet.faceGeometries;
	faceGeometries.clear();

	auto& poseLandmarkLists = m_resultData.tmpDataSet.poseLandmarkLists;
	poseLandmarkLists.clear();

	auto& handLandmarkLists = m_resultData.tmpDataSet.handLandmarkLists;
	handLandmarkLists.clear();

	m_resultData.tmpDataSet.errorMessage = {};

	mediapipe::Image* image;
	size_t frameIndex = 0;
	if (!UpdateFrame(outErr, &image, frameIndex))
		return false;
	// Process blend shapes
	auto& taskRunner = get_mp_task_runner(m_taskRunner);
	auto t = std::chrono::steady_clock::now();

	auto packetImg = mediapipe::MakePacket<mediapipe::Image>(*image);
	packetImg = packetImg.At(mediapipe::Timestamp(frameIndex)); // Frame index is required for smoother landmarks

	auto packetArea = mediapipe::MakePacket<mediapipe::NormalizedRect>(MakeNormRect(0.5, 0.5, 1.0, 1.0, 0));
	packetArea = packetArea.At(mediapipe::Timestamp(frameIndex));

	std::vector<int64_t> trackingIds = { 0 }; // TODO
	auto packetTrackingIds = mediapipe::MakePacket<std::vector<int64_t>>(trackingIds);
	packetTrackingIds = packetTrackingIds.At(mediapipe::Timestamp(frameIndex));

	auto outputPackets = taskRunner.Process(
		{ {"image", packetImg},

		// Documentation states that this input is optional, but that appears to be false
		{"norm_rect",packetArea},

		// Smoothing
		{"tracking_ids",packetTrackingIds}
		});
	auto dt = std::chrono::steady_clock::now() - t;
	if (!outputPackets.ok()) {
		outErr = "Failed to process face landmarker task: " + std::string{outputPackets.status().message()};
		return false;
	}
	m_packetMap = std::move(outputPackets.value());
	auto& packetBlendshapes = m_packetMap["blendshapes"];
	if (!packetBlendshapes.IsEmpty() && IsOutputEnabled(Output::BlendShapeCoefficients)) {
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

	auto& packetFaceGeometry = m_packetMap["face_geometry"];
	if (!packetFaceGeometry.IsEmpty() && IsOutputEnabled(Output::FaceGeometry)) {
		auto& packetFaceGeometryLists = packetFaceGeometry.Get<std::vector<mediapipe::tasks::vision::face_geometry::proto::FaceGeometry>>();

		faceGeometries.resize(packetFaceGeometryLists.size());
		for (auto i = decltype(packetFaceGeometryLists.size()){0u}; i < packetFaceGeometryLists.size(); ++i) {
			auto& packetFaceGeometryList = packetFaceGeometryLists[i];
			auto& mesh = packetFaceGeometryList.mesh();
			auto& geometry = faceGeometries[i];

			auto numIndices = mesh.index_buffer_size();
			auto numVertValues = mesh.vertex_buffer_size();
			auto numVerts = numVertValues / 5;
			geometry.indices.resize(numIndices);
			geometry.vertices.resize(numVerts);
			for (auto j = decltype(numIndices){0u}; j < numIndices; ++j) {
				geometry.indices[j] = mesh.index_buffer(j);
			}
			for (auto j = decltype(numVerts){0u}; j < numVerts; ++j) {
				geometry.vertices[j] = {
					mesh.vertex_buffer(j *5),
					mesh.vertex_buffer(j * 5 +1),
					mesh.vertex_buffer(j * 5 +2)
				};
			}
		}
	}

	auto& packetPose = m_packetMap["filtered_pose_world_landmarks"];
	if (!packetPose.IsEmpty() && IsOutputEnabled(Output::PoseWorldLandmarks)) {
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
	if (!packetHand.IsEmpty() && IsOutputEnabled(Output::HandWorldLandmarks)) {
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
	if (!packetHandedness.IsEmpty() && IsOutputEnabled(Output::HandWorldLandmarks)) {
		// m_resultData.handednessList = &packetHandedness.Get<std::vector<mediapipe::ClassificationList>>(); // TODO
	}
	return true;
}

void mpw::MotionCaptureManager::Stop()
{
	if (!m_running)
		return;
	m_running = false;
	m_taskCondition.notify_all();
	m_frameCompleteCondition.notify_all();

	if (m_mainThread.joinable())
		m_mainThread.join();
	cv::destroyWindow("MediaPipe");
}

bool mpw::MotionCaptureManager::Start(std::string& outErr)
{
	cv::namedWindow("MediaPipe");
	return StartNextFrame(outErr);
}

bool mpw::MotionCaptureManager::IsOutputEnabled(Output output) const
{
	return (static_cast<uint32_t>(m_enabledOutputs) & static_cast<uint32_t>(output)) != 0;
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

bool mpw::MotionCaptureManager::UpdateFrame(std::string& outErr, mediapipe::Image** outImg, size_t& outFrameIndex)
{
	outFrameIndex = 0;
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
		// Loop
		cap.set(cv::CAP_PROP_POS_FRAMES, 0);
		return false;
	}
	auto msTime = cap.get(cv::CAP_PROP_POS_MSEC);

	//cv::rotate(frameBgr, frameBgr, cv::ROTATE_90_COUNTERCLOCKWISE);

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

	auto t = msTime *1000.f;
	outFrameIndex = static_cast<size_t>(t);

	auto& img = *streamInputData.currentFrameImage;
	img = { input_frame_for_input };
	*outImg = &img;
	auto origWidth = 1226.f;// 367.f;
	auto origHeight = 690.f;// 653.f;
	auto aspectRatio = origHeight /origWidth;
	cv::resize(frameBgr, frameBgr, cv::Size(600, 600 * aspectRatio), cv::INTER_AREA);
	cv::imshow("MediaPipe", frameBgr);
	cv::waitKey(1);
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
	streamInputData->startTime = std::chrono::steady_clock::now();
	auto& cap = *streamInputData->capture;

	// cap.set(cv::CAP_PROP_ORIENTATION_AUTO, true);
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
template <class TPayloadImg>
bool mpw::MotionCaptureManager::CreateFaceLandmarkerTask(::mediapipe::api2::builder::Graph& graph, TPayloadImg& imgInput, std::string& outErr) {
	auto& faceLandmarker = graph.AddNode("mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph");

	auto* options = &faceLandmarker.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>();
	options->mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", g_dataPath, m_faceLandmarkerModel));
	options->mutable_face_detector_graph_options()->set_num_faces(1);
	options->mutable_base_options()->set_use_stream_mode(true);

	imgInput >>
		faceLandmarker.In("IMAGE");

	if (IsOutputEnabled(Output::BlendShapeCoefficients)) {
		faceLandmarker.Out("BLENDSHAPES").SetName("blendshapes") >>
			graph[::mediapipe::api2::Output< std::vector<mediapipe::ClassificationList>>("BLENDSHAPES")];
	}
	if (IsOutputEnabled(Output::FaceGeometry)) {
		faceLandmarker.Out("FACE_GEOMETRY").SetName("face_geometry") >>
			graph[::mediapipe::api2::Output< std::vector<mediapipe::tasks::vision::face_geometry::proto::FaceGeometry>>("FACE_GEOMETRY")];
	}
	return true;
}
template <class TPayloadImg, class TPayloadTrackingIds>
bool mpw::MotionCaptureManager::CreatePoseLandmarkerTask(::mediapipe::api2::builder::Graph& graph, TPayloadImg& imgInput, TPayloadTrackingIds& trackingIdsInput, std::string& outErr) {
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

	if (IsOutputEnabled(Output::PoseWorldLandmarks)) {
		auto outWorldLandmarks = poseLandmarker.Out("WORLD_LANDMARKS").SetName("pose_world_landmarks") >>
			graph[::mediapipe::api2::Output< std::vector<mediapipe::LandmarkList>>("POSE_WORLD_LANDMARKS")];
		CreateSmoothFilter(graph, outWorldLandmarks, trackingIdsInput, "filtered_pose_world_landmarks", "FILTERED_LANDMARKS");
	}


	return true;
}
template <class TPayloadImg, class TPayloadTrackingIds>
bool mpw::MotionCaptureManager::CreateHandLandmarkerTask(::mediapipe::api2::builder::Graph& graph, TPayloadImg& imgInput, TPayloadTrackingIds& trackingIdsInput, std::string& outErr) {
	auto& handLandmarker = graph.AddNode(
		"mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph");

	auto* options = &handLandmarker.GetOptions<mediapipe::tasks::vision::hand_landmarker::proto::HandLandmarkerGraphOptions>();
	options->mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", g_dataPath, m_handLandmarkerModel));
	options->mutable_hand_detector_graph_options()->set_num_hands(2);
	options->mutable_base_options()->set_use_stream_mode(true);

	imgInput >>
		handLandmarker.In("IMAGE");
	if (IsOutputEnabled(Output::HandWorldLandmarks)) {
		auto outWorldLandmarks = handLandmarker.Out("WORLD_LANDMARKS").SetName("hand_world_landmarks") >>
			graph[::mediapipe::api2::Output<std::vector<mediapipe::LandmarkList>>("HAND_WORLD_LANDMARKS")];
		CreateSmoothFilter(graph, outWorldLandmarks, trackingIdsInput, "filtered_hand_world_landmarks", "FILTERED_HAND_LANDMARKS");

		handLandmarker.Out("HANDEDNESS").SetName("handedness") >>
			graph[::mediapipe::api2::Output< std::vector<mediapipe::ClassificationList>>("HANDEDNESS")];
	}
	return true;
}
template <class TPayloadWorldLandmarks, class TPayloadTrackingIds>
void mpw::MotionCaptureManager::CreateSmoothFilter(::mediapipe::api2::builder::Graph& graph, TPayloadWorldLandmarks& worldLandmarks, TPayloadTrackingIds& trackingIdsInput, const char* outputName, const char* graphOutputName)
{
	// Smoothing
	auto& smoothCalculator = graph.AddNode(
		"MultiWorldLandmarksSmoothingCalculator");
	auto* options = &smoothCalculator.GetOptions<mediapipe::LandmarksSmoothingCalculatorOptions>();

	// Min cutoff 0.05 results into ~0.01 alpha in landmark EMA filter when
	// landmark is static.
	options->mutable_one_euro_filter()->set_min_cutoff(0.05f);
	// Beta 80.0 in combintation with min_cutoff 0.05 results into ~0.94
	// alpha in landmark EMA filter when landmark is moving fast.
	options->mutable_one_euro_filter()->set_beta(80.0f);
	// Derivative cutoff 1.0 results into ~0.17 alpha in landmark velocity
	// EMA filter.
	options->mutable_one_euro_filter()->set_derivate_cutoff(1.0f);
	worldLandmarks >>
		smoothCalculator.In("LANDMARKS");
	trackingIdsInput >>
		smoothCalculator.In("TRACKING_IDS");
	smoothCalculator.Out("FILTERED_LANDMARKS").SetName(outputName) >>
		graph[::mediapipe::api2::Output< std::vector<mediapipe::LandmarkList>>(graphOutputName)];
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
				m_resultData.dataSet.faceGeometries = std::move(m_resultData.tmpDataSet.faceGeometries);
				m_resultData.dataSet.errorMessage = std::move(m_resultData.tmpDataSet.errorMessage);
				++m_resultData.frameIndex;
				m_resultData.resultDataMutex.unlock();
				m_frameCompleteCondition.notify_all();
			}
			if (!m_running)
				break;
			if (m_autoAdvance) {
				std::string err;
				StartNextFrame(err);
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
	Stop();
	if(m_mainThread.joinable())
		m_mainThread.join();
}

const char* mpw::get_blend_shape_name(BlendShape blendShape)
{
	switch (blendShape)
	{
	case BlendShape::Neutral:
		return "_neutral";
	case BlendShape::BrowDownLeft:
		return "browDownLeft";
	case BlendShape::BrowDownRight:
		return "browDownRight";
	case BlendShape::BrowInnerUp:
		return "browInnerUp";
	case BlendShape::BrowOuterUpLeft:
		return "browOuterUpLeft";
	case BlendShape::BrowOuterUpRight:
		return "browOuterUpRight";
	case BlendShape::CheekPuff:
		return "cheekPuff";
	case BlendShape::CheekSquintLeft:
		return "cheekSquintLeft";
	case BlendShape::CheekSquintRight:
		return "cheekSquintRight";
	case BlendShape::EyeBlinkLeft:
		return "eyeBlinkLeft";
	case BlendShape::EyeBlinkRight:
		return "eyeBlinkRight";
	case BlendShape::EyeLookDownLeft:
		return "eyeLookDownLeft";
	case BlendShape::EyeLookDownRight:
		return "eyeLookDownRight";
	case BlendShape::EyeLookInLeft:
		return "eyeLookInLeft";
	case BlendShape::EyeLookInRight:
		return "eyeLookInRight";
	case BlendShape::EyeLookOutLeft:
		return "eyeLookOutLeft";
	case BlendShape::EyeLookOutRight:
		return "eyeLookOutRight";
	case BlendShape::EyeLookUpLeft:
		return "eyeLookUpLeft";
	case BlendShape::EyeLookUpRight:
		return "eyeLookUpRight";
	case BlendShape::EyeSquintLeft:
		return "eyeSquintLeft";
	case BlendShape::EyeSquintRight:
		return "eyeSquintRight";
	case BlendShape::EyeWideLeft:
		return "eyeWideLeft";
	case BlendShape::EyeWideRight:
		return "eyeWideRight";
	case BlendShape::JawForward:
		return "jawForward";
	case BlendShape::JawLeft:
		return "jawLeft";
	case BlendShape::JawOpen:
		return "jawOpen";
	case BlendShape::JawRight:
		return "jawRight";
	case BlendShape::MouthClose:
		return "mouthClose";
	case BlendShape::MouthDimpleLeft:
		return "mouthDimpleLeft";
	case BlendShape::MouthDimpleRight:
		return "mouthDimpleRight";
	case BlendShape::MouthFrownLeft:
		return "mouthFrownLeft";
	case BlendShape::MouthFrownRight:
		return "mouthFrownRight";
	case BlendShape::MouthFunnel:
		return "mouthFunnel";
	case BlendShape::MouthLeft:
		return "mouthLeft";
	case BlendShape::MouthLowerDownLeft:
		return "mouthLowerDownLeft";
	case BlendShape::MouthLowerDownRight:
		return "mouthLowerDownRight";
	case BlendShape::MouthPressLeft:
		return "mouthPressLeft";
	case BlendShape::MouthPressRight:
		return "mouthPressRight";
	case BlendShape::MouthPucker:
		return "mouthPucker";
	case BlendShape::MouthRight:
		return "mouthRight";
	case BlendShape::MouthRollLower:
		return "mouthRollLower";
	case BlendShape::MouthRollUpper:
		return "mouthRollUpper";
	case BlendShape::MouthShrugLower:
		return "mouthShrugLower";
	case BlendShape::MouthShrugUpper:
		return "mouthShrugUpper";
	case BlendShape::MouthSmileLeft:
		return "mouthSmileLeft";
	case BlendShape::MouthSmileRight:
		return "mouthSmileRight";
	case BlendShape::MouthStretchLeft:
		return "mouthStretchLeft";
	case BlendShape::MouthStretchRight:
		return "mouthStretchRight";
	case BlendShape::MouthUpperUpLeft:
		return "mouthUpperUpLeft";
	case BlendShape::MouthUpperUpRight:
		return "mouthUpperUpRight";
	case BlendShape::NoseSneerLeft:
		return "noseSneerLeft";
	case BlendShape::NoseSneerRight:
		return "noseSneerRight";
	default:
		return "";
	}
}
std::optional<mpw::BlendShape> mpw::get_blend_shape_enum(const char* name)
{
	static const std::unordered_map<std::string, BlendShape> blendShapeMap = {
		{"_neutral", BlendShape::Neutral},
		{"browDownLeft", BlendShape::BrowDownLeft},
		{"browDownRight", BlendShape::BrowDownRight},
		{"browInnerUp", BlendShape::BrowInnerUp},
		{"browOuterUpLeft", BlendShape::BrowOuterUpLeft},
		{"browOuterUpRight", BlendShape::BrowOuterUpRight},
		{"cheekPuff", BlendShape::CheekPuff},
		{"cheekSquintLeft", BlendShape::CheekSquintLeft},
		{"cheekSquintRight", BlendShape::CheekSquintRight},
		{"eyeBlinkLeft", BlendShape::EyeBlinkLeft},
		{"eyeBlinkRight", BlendShape::EyeBlinkRight},
		{"eyeLookDownLeft", BlendShape::EyeLookDownLeft},
		{"eyeLookDownRight", BlendShape::EyeLookDownRight},
		{"eyeLookInLeft", BlendShape::EyeLookInLeft},
		{"eyeLookInRight", BlendShape::EyeLookInRight},
		{"eyeLookOutLeft", BlendShape::EyeLookOutLeft},
		{"eyeLookOutRight", BlendShape::EyeLookOutRight},
		{"eyeLookUpLeft", BlendShape::EyeLookUpLeft},
		{"eyeLookUpRight", BlendShape::EyeLookUpRight},
		{"eyeSquintLeft", BlendShape::EyeSquintLeft},
		{"eyeSquintRight", BlendShape::EyeSquintRight},
		{"eyeWideLeft", BlendShape::EyeWideLeft},
		{"eyeWideRight", BlendShape::EyeWideRight},
		{"jawForward", BlendShape::JawForward},
		{"jawLeft", BlendShape::JawLeft},
		{"jawOpen", BlendShape::JawOpen},
		{"jawRight", BlendShape::JawRight},
		{"mouthClose", BlendShape::MouthClose},
		{"mouthDimpleLeft", BlendShape::MouthDimpleLeft},
		{"mouthDimpleRight", BlendShape::MouthDimpleRight},
		{"mouthFrownLeft", BlendShape::MouthFrownLeft},
		{"mouthFrownRight", BlendShape::MouthFrownRight},
		{"mouthFunnel", BlendShape::MouthFunnel},
		{"mouthLeft", BlendShape::MouthLeft},
		{"mouthLowerDownLeft", BlendShape::MouthLowerDownLeft},
		{"mouthLowerDownRight", BlendShape::MouthLowerDownRight},
		{"mouthPressLeft", BlendShape::MouthPressLeft},
		{"mouthPressRight", BlendShape::MouthPressRight},
		{"mouthPucker", BlendShape::MouthPucker},
		{"mouthRight", BlendShape::MouthRight},
		{"mouthRollLower", BlendShape::MouthRollLower},
		{"mouthRollUpper", BlendShape::MouthRollUpper},
		{"mouthShrugLower", BlendShape::MouthShrugLower},
		{"mouthShrugUpper", BlendShape::MouthShrugUpper},
		{"mouthSmileLeft", BlendShape::MouthSmileLeft},
		{"mouthSmileRight", BlendShape::MouthSmileRight},
		{"mouthStretchLeft", BlendShape::MouthStretchLeft},
		{"mouthStretchRight", BlendShape::MouthStretchRight},
		{"mouthUpperUpLeft", BlendShape::MouthUpperUpLeft},
		{"mouthUpperUpRight", BlendShape::MouthUpperUpRight},
		{"noseSneerLeft", BlendShape::NoseSneerLeft},
		{"noseSneerRight", BlendShape::NoseSneerRight}
	};

	auto it = blendShapeMap.find(name);
	if (it != blendShapeMap.end()) {
		return it->second;
	}

	return {};
}

const char* mpw::get_pose_landmark_name(PoseLandmark poseLandmark)
{
	switch (poseLandmark)
	{
	case PoseLandmark::Nose:
		return "nose";
	case PoseLandmark::LeftEyeInner:
		return "left eye (inner)";
	case PoseLandmark::LeftEye:
		return "left eye";
	case PoseLandmark::LeftEyeOuter:
		return "left eye (outer)";
	case PoseLandmark::RightEyeInner:
		return "right eye (inner)";
	case PoseLandmark::RightEye:
		return "right eye";
	case PoseLandmark::RightEyeOuter:
		return "right eye (outer)";
	case PoseLandmark::LeftEar:
		return "left ear";
	case PoseLandmark::RightEar:
		return "right ear";
	case PoseLandmark::MouthLeft:
		return "mouth (left)";
	case PoseLandmark::MouthRight:
		return "mouth (right)";
	case PoseLandmark::LeftShoulder:
		return "left shoulder";
	case PoseLandmark::RightShoulder:
		return "right shoulder";
	case PoseLandmark::LeftElbow:
		return "left elbow";
	case PoseLandmark::RightElbow:
		return "right elbow";
	case PoseLandmark::LeftWrist:
		return "left wrist";
	case PoseLandmark::RightWrist:
		return "right wrist";
	case PoseLandmark::LeftPinky:
		return "left pinky";
	case PoseLandmark::RightPinky:
		return "right pinky";
	case PoseLandmark::LeftIndex:
		return "left index";
	case PoseLandmark::RightIndex:
		return "right index";
	case PoseLandmark::LeftThumb:
		return "left thumb";
	case PoseLandmark::RightThumb:
		return "right thumb";
	case PoseLandmark::LeftHip:
		return "left hip";
	case PoseLandmark::RightHip:
		return "right hip";
	case PoseLandmark::LeftKnee:
		return "left knee";
	case PoseLandmark::RightKnee:
		return "right knee";
	case PoseLandmark::LeftAnkle:
		return "left ankle";
	case PoseLandmark::RightAnkle:
		return "right ankle";
	case PoseLandmark::LeftHeel:
		return "left heel";
	case PoseLandmark::RightHeel:
		return "right heel";
	case PoseLandmark::LeftFootIndex:
		return "left foot index";
	case PoseLandmark::RightFootIndex:
		return "right foot index";
	default:
		return "";
	}
}
std::optional<mpw::PoseLandmark> mpw::get_pose_landmark_enum(const char* name)
{
	static const std::unordered_map<std::string, PoseLandmark> poseLandmarkMap = {
		{"nose", PoseLandmark::Nose},
		{"left eye (inner)", PoseLandmark::LeftEyeInner},
		{"left eye", PoseLandmark::LeftEye},
		{"left eye (outer)", PoseLandmark::LeftEyeOuter},
		{"right eye (inner)", PoseLandmark::RightEyeInner},
		{"right eye", PoseLandmark::RightEye},
		{"right eye (outer)", PoseLandmark::RightEyeOuter},
		{"left ear", PoseLandmark::LeftEar},
		{"right ear", PoseLandmark::RightEar},
		{"mouth (left)", PoseLandmark::MouthLeft},
		{"mouth (right)", PoseLandmark::MouthRight},
		{"left shoulder", PoseLandmark::LeftShoulder},
		{"right shoulder", PoseLandmark::RightShoulder},
		{"left elbow", PoseLandmark::LeftElbow},
		{"right elbow", PoseLandmark::RightElbow},
		{"left wrist", PoseLandmark::LeftWrist},
		{"right wrist", PoseLandmark::RightWrist},
		{"left pinky", PoseLandmark::LeftPinky},
		{"right pinky", PoseLandmark::RightPinky},
		{"left index", PoseLandmark::LeftIndex},
		{"right index", PoseLandmark::RightIndex},
		{"left thumb", PoseLandmark::LeftThumb},
		{"right thumb", PoseLandmark::RightThumb},
		{"left hip", PoseLandmark::LeftHip},
		{"right hip", PoseLandmark::RightHip},
		{"left knee", PoseLandmark::LeftKnee},
		{"right knee", PoseLandmark::RightKnee},
		{"left ankle", PoseLandmark::LeftAnkle},
		{"right ankle", PoseLandmark::RightAnkle},
		{"left heel", PoseLandmark::LeftHeel},
		{"right heel", PoseLandmark::RightHeel},
		{"left foot index", PoseLandmark::LeftFootIndex},
		{"right foot index", PoseLandmark::RightFootIndex}
	};

	auto it = poseLandmarkMap.find(name);
	if (it != poseLandmarkMap.end()) {
		return it->second;
	}

	return {};
}

const char* mpw::get_hand_landmark_name(HandLandmark handLandmark)
{
	switch (handLandmark)
	{
	case HandLandmark::Wrist:
		return "wrist";
	case HandLandmark::ThumbCMC:
		return "thumb CMC";
	case HandLandmark::ThumbMCP:
		return "thumb MCP";
	case HandLandmark::ThumbIP:
		return "thumb IP";
	case HandLandmark::ThumbTip:
		return "thumb tip";
	case HandLandmark::IndexFingerMCP:
		return "index finger MCP";
	case HandLandmark::IndexFingerPIP:
		return "index finger PIP";
	case HandLandmark::IndexFingerDIP:
		return "index finger DIP";
	case HandLandmark::IndexFingerTip:
		return "index finger tip";
	case HandLandmark::MiddleFingerMCP:
		return "middle finger MCP";
	case HandLandmark::MiddleFingerPIP:
		return "middle finger PIP";
	case HandLandmark::MiddleFingerDIP:
		return "middle finger DIP";
	case HandLandmark::MiddleFingerTip:
		return "middle finger tip";
	case HandLandmark::RingFingerMCP:
		return "ring finger MCP";
	case HandLandmark::RingFingerPIP:
		return "ring finger PIP";
	case HandLandmark::RingFingerDIP:
		return "ring finger DIP";
	case HandLandmark::RingFingerTip:
		return "ring finger tip";
	case HandLandmark::PinkyMCP:
		return "pinky MCP";
	case HandLandmark::PinkyPIP:
		return "pinky PIP";
	case HandLandmark::PinkyDIP:
		return "pinky DIP";
	case HandLandmark::PinkyTip:
		return "pinky tip";
	default:
		return "";
	}
}
std::optional<mpw::HandLandmark> mpw::get_hand_landmark_enum(const char* name)
{
	static const std::unordered_map<std::string, HandLandmark> handLandmarkMap = {
		{"wrist", HandLandmark::Wrist},
		{"thumb CMC", HandLandmark::ThumbCMC},
		{"thumb MCP", HandLandmark::ThumbMCP},
		{"thumb IP", HandLandmark::ThumbIP},
		{"thumb tip", HandLandmark::ThumbTip},
		{"index finger MCP", HandLandmark::IndexFingerMCP},
		{"index finger PIP", HandLandmark::IndexFingerPIP},
		{"index finger DIP", HandLandmark::IndexFingerDIP},
		{"index finger tip", HandLandmark::IndexFingerTip},
		{"middle finger MCP", HandLandmark::MiddleFingerMCP},
		{"middle finger PIP", HandLandmark::MiddleFingerPIP},
		{"middle finger DIP", HandLandmark::MiddleFingerDIP},
		{"middle finger tip", HandLandmark::MiddleFingerTip},
		{"ring finger MCP", HandLandmark::RingFingerMCP},
		{"ring finger PIP", HandLandmark::RingFingerPIP},
		{"ring finger DIP", HandLandmark::RingFingerDIP},
		{"ring finger tip", HandLandmark::RingFingerTip},
		{"pinky MCP", HandLandmark::PinkyMCP},
		{"pinky PIP", HandLandmark::PinkyPIP},
		{"pinky DIP", HandLandmark::PinkyDIP},
		{"pinky tip", HandLandmark::PinkyTip}
	};

	auto it = handLandmarkMap.find(name);
	if (it != handLandmarkMap.end()) {
		return it->second;
	}

	return {};
}
