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
#include <cinttypes>

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
	};

	class DLLMPW MotionCaptureManager {
	public:
		static std::shared_ptr< MotionCaptureManager> CreateFromImage(const std::string& source, std::string& outErr);
		static std::shared_ptr< MotionCaptureManager> CreateFromVideo(const std::string& source, std::string& outErr);
		static std::shared_ptr< MotionCaptureManager> CreateFromCamera(CameraDeviceId deviceId, std::string& outErr);
		virtual ~MotionCaptureManager() = default;
		bool Start(std::string& outErr);
		bool ProcessNextFrame(std::string& outErr);

		size_t GetBlendShapeCollectionCount() const;
		bool GetBlendShapeCoefficient(size_t collectionIndex, BlendShape blendShape, float& outCoefficient);

		size_t GetPoseCollectionCount() const;
		bool GetPoseWorldLandmarkPosition(size_t collectionIndex, PoseLandmark poseLandmark, std::array<float, 3>& outPosition, float& outPresence, float& outVisibility);

		size_t GetHandCollectionCount() const;
		bool GetHandWorldLandmarkPosition(size_t collectionIndex, HandLandmark handLandmark, std::array<float, 3>& outPosition, float& outPresence, float& outVisibility);
	private:
		enum class SourceType : uint32_t {
			Image = 0,
			Video,
			Camera
		};
		bool ProcessImage(mediapipe::Image& image, std::string& outErr);
		using Source = std::variant<std::string, CameraDeviceId>;
		static std::shared_ptr< MotionCaptureManager> Create(SourceType type, const Source& source, std::string& outErr);
		MotionCaptureManager();
		bool CreateFaceLandmarkerTask(std::string& outErr);
		bool CreatePoseLandmarkerTask(std::string& outErr);
		bool CreateHandLandmarkerTask(std::string& outErr);

		struct Task {
			Task(const std::string& mdlName)
				: modelName{ mdlName }
			{}
			std::string modelName;
			TaskRunner taskRunner;
		};
		Task m_faceLandmarker{ "face_landmarker_v2_with_blendshapes.task" };
		Task m_poseLandmarker{ "pose_landmarker.task" };
		Task m_handLandmarker{ "hand_landmarker.task" };

		struct {
			std::map<std::string, mediapipe::Packet> packetMap;
			const std::vector<mediapipe::ClassificationList>* collections = nullptr;
		} m_blendShapeResult;

		struct {
			std::map<std::string, mediapipe::Packet> packetMap;
			const std::vector<mediapipe::LandmarkList>* landmarkLists = nullptr;
		} m_poseResult;

		struct {
			std::map<std::string, mediapipe::Packet> packetMap;
			const std::vector<mediapipe::LandmarkList>* landmarkLists = nullptr;
			const std::vector<mediapipe::ClassificationList>* handednessList = nullptr;
		} m_handResult;

		std::unique_ptr< BaseInputData> m_inputData = nullptr;
		SourceType m_sourceType = SourceType::Image;
		Source m_source;
	};
};

#endif
