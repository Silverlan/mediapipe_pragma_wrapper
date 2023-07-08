/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2023 Silverlan
 */

#ifndef __MEDIAPIPE_PRAGMA_WRAPPER_HPP__
#define __MEDIAPIPE_PRAGMA_WRAPPER_HPP__

#include <memory>
#include <string>
#include <map>
#include <vector>
#include <variant>
#include <thread>
#include <mutex>
#include <optional>
#include <atomic>
#include <array>
#include <functional>
#include <condition_variable>
#include <cinttypes>
#include <chrono>

#ifdef DLLMPW_EX
#ifdef __linux__
#define DLLMPW __attribute__((visibility("default")))
#else
#define DLLMPW __declspec(dllexport)
#endif
#else
#ifdef __linux__
#define DLLMPW
#else
#define DLLMPW __declspec(dllimport)
#endif
#endif

namespace mediapipe {
	class Image;
	class Packet;
	class ClassificationList;
	class LandmarkList;
	namespace tasks {
		namespace core {
			class TaskRunner;
		};
	};
	namespace api2 {
		namespace builder {
			class Graph;
		};
	};
};

namespace cv {
	class VideoCapture;
};

namespace mpw {
	using MpTaskRunner = mediapipe::tasks::core::TaskRunner;
	struct TaskRunner {
		std::shared_ptr<void> taskRunner;
	};
	MpTaskRunner& get_mp_task_runner(TaskRunner& taskRunner);
	using CameraDeviceId = uint32_t;

	enum class BlendShape : uint32_t {
		Neutral = 0,
		BrowDownLeft,
		BrowDownRight,
		BrowInnerUp,
		BrowOuterUpLeft,
		BrowOuterUpRight,
		CheekPuff,
		CheekSquintLeft,
		CheekSquintRight,
		EyeBlinkLeft,
		EyeBlinkRight,
		EyeLookDownLeft,
		EyeLookDownRight,
		EyeLookInLeft,
		EyeLookInRight,
		EyeLookOutLeft,
		EyeLookOutRight,
		EyeLookUpLeft,
		EyeLookUpRight,
		EyeSquintLeft,
		EyeSquintRight,
		EyeWideLeft,
		EyeWideRight,
		JawForward,
		JawLeft,
		JawOpen,
		JawRight,
		MouthClose,
		MouthDimpleLeft,
		MouthDimpleRight,
		MouthFrownLeft,
		MouthFrownRight,
		MouthFunnel,
		MouthLeft,
		MouthLowerDownLeft,
		MouthLowerDownRight,
		MouthPressLeft,
		MouthPressRight,
		MouthPucker,
		MouthRight,
		MouthRollLower,
		MouthRollUpper,
		MouthShrugLower,
		MouthShrugUpper,
		MouthSmileLeft,
		MouthSmileRight,
		MouthStretchLeft,
		MouthStretchRight,
		MouthUpperUpLeft,
		MouthUpperUpRight,
		NoseSneerLeft,
		NoseSneerRight,

		Count
	};

	enum class PoseLandmark : uint32_t {
		Nose = 0,
		LeftEyeInner,
		LeftEye,
		LeftEyeOuter,
		RightEyeInner,
		RightEye,
		RightEyeOuter,
		LeftEar,
		RightEar,
		MouthLeft,
		MouthRight,
		LeftShoulder,
		RightShoulder,
		LeftElbow,
		RightElbow,
		LeftWrist,
		RightWrist,
		LeftPinky,
		RightPinky,
		LeftIndex,
		RightIndex,
		LeftThumb,
		RightThumb,
		LeftHip,
		RightHip,
		LeftKnee,
		RightKnee,
		LeftAnkle,
		RightAnkle,
		LeftHeel,
		RightHeel,
		LeftFootIndex,
		RightFootIndex,

		Count
	};

	enum class HandLandmark : uint32_t {
		Wrist = 0,
		ThumbCMC,
		ThumbMCP,
		ThumbIP,
		ThumbTip,
		IndexFingerMCP,
		IndexFingerPIP,
		IndexFingerDIP,
		IndexFingerTip,
		MiddleFingerMCP,
		MiddleFingerPIP,
		MiddleFingerDIP,
		MiddleFingerTip,
		RingFingerMCP,
		RingFingerPIP,
		RingFingerDIP,
		RingFingerTip,
		PinkyMCP,
		PinkyPIP,
		PinkyDIP,
		PinkyTip,

		Count
	};

	DLLMPW void init(const char* rootPath, const char* dataPath);

	struct BaseInputData {};

	struct ImageInputData : public BaseInputData {
		std::shared_ptr< mediapipe::Image> image = nullptr;
	};

	struct StreamInputData : public BaseInputData {
		std::shared_ptr< cv::VideoCapture> capture = nullptr;
		std::shared_ptr< mediapipe::Image> currentFrameImage = nullptr;
	};

	class DLLMPW MotionCaptureManager {
	public:
		enum class LogSeverity : uint8_t {
			Info = 0,
			Warning,
			Error
		};

		struct LandmarkData {
			std::array<float, 3> pos;
			float presence;
			float visibility;
		};

		static std::shared_ptr< MotionCaptureManager> CreateFromImage(const std::string& source, std::string& outErr);
		static std::shared_ptr< MotionCaptureManager> CreateFromVideo(const std::string& source, std::string& outErr);
		static std::shared_ptr< MotionCaptureManager> CreateFromCamera(CameraDeviceId deviceId, std::string& outErr);
		~MotionCaptureManager();
		bool Start(std::string& outErr);

		void LockResultData();
		void UnlockResultData();

		size_t GetBlendShapeCollectionCount() const;
		bool GetBlendShapeCoefficient(size_t collectionIndex, BlendShape blendShape, float& outCoefficient) const;
		bool GetBlendShapeCoefficients(size_t collectionIndex, std::vector<float> &outCoefficients) const;
		void GetBlendShapeCoefficientLists(std::vector<std::vector<float>> &outCoefficientLists) const;

		size_t GetPoseCollectionCount() const;
		bool GetPoseWorldLandmark(size_t collectionIndex, PoseLandmark poseLandmark, LandmarkData &outLandmarkData) const;
		bool GetPoseWorldLandmarks(size_t collectionIndex, std::vector<LandmarkData>& outLandmarks) const;
		void GetPoseWorldLandmarkLists(std::vector<std::vector<LandmarkData>>& outLandmarks) const;

		size_t GetHandCollectionCount() const;
		bool GetHandWorldLandmark(size_t collectionIndex, HandLandmark handLandmark, LandmarkData& outLandmarkData) const;
		bool GetHandWorldLandmarks(size_t collectionIndex, std::vector<LandmarkData>& outLandmarks) const;
		void GetHandWorldLandmarkLists(std::vector<std::vector<LandmarkData>>& outLandmarks) const;

		std::optional<std::string> GetLastError() const;

		bool IsFrameComplete() const;
		void WaitForFrame();
	private:
		enum class SourceType : uint32_t {
			Image = 0,
			Video,
			Camera
		};
		using Source = std::variant<std::string, CameraDeviceId>;
		static std::shared_ptr< MotionCaptureManager> Create(SourceType type, const Source& source, std::string& outErr);
		MotionCaptureManager();
		bool CreateFaceLandmarkerTask(::mediapipe::api2::builder::Graph &graph, void *imgInput, std::string& outErr);
		bool CreatePoseLandmarkerTask(::mediapipe::api2::builder::Graph &graph, void* imgInput, std::string& outErr);
		bool CreateHandLandmarkerTask(::mediapipe::api2::builder::Graph &graph, void* imgInput, std::string& outErr);
		void InitializeThreads();
		bool InitializeSource(std::string& outErr);
		bool UpdateFrame(std::string &outErr, mediapipe::Image **outImg);
		bool StartNextFrame(std::string& outErr);
		bool Process(std::string& outErr);

		std::string m_faceLandmarkerModel{ "face_landmarker_v2_with_blendshapes.task" };
		std::string m_poseLandmarkerModel{ "pose_landmarker.task" };
		std::string m_handLandmarkerModel{ "hand_landmarker.task" };
		TaskRunner m_taskRunner;

		std::thread m_mainThread;
		bool m_hasTask = false;
		std::condition_variable m_taskCondition;
		std::mutex m_taskMutex;

		std::atomic<bool> m_frameComplete = false;
		std::condition_variable m_frameCompleteCondition;
		std::mutex m_frameCompleteMutex;
		std::atomic<bool> m_running = true;
		std::atomic<bool> m_autoAdvance = true;
		bool m_firstFrameStarted = false;
		std::chrono::steady_clock::time_point m_tFrameStart;

		std::map<std::string, mediapipe::Packet> m_packetMap;
		struct ResultDataSet {
			bool hasBlendShapeCoefficients = false;
			std::vector<std::vector<float>> blendShapeCoefficientLists;

			bool hasPoseLandmarks = false;
			std::vector<std::vector<LandmarkData>> poseLandmarkLists;

			bool hasHandLandmarks = false;
			std::vector<std::vector<LandmarkData>> handLandmarkLists;

			std::optional<std::string> errorMessage {};
		};
		struct {
			size_t frameIndex = 0;
			std::recursive_mutex resultDataMutex;
			ResultDataSet dataSet;
			ResultDataSet tmpDataSet;
		} m_resultData;

		std::unique_ptr< BaseInputData> m_inputData = nullptr;
		SourceType m_sourceType = SourceType::Image;
		Source m_source;
	};
};

#endif
