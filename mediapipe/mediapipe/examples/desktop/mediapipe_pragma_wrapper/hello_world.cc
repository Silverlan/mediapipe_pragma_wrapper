// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// A simple example to print out "Hello World!" from a MediaPipe graph.

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>

namespace mediapipe {

absl::Status PrintHelloWorld() {
  // Configures a simple graph, which concatenates 2 PassThroughCalculators.
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "in"
          output_stream: "out1"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "out1"
          output_stream: "out"
        }
      )pb");

  CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  ASSIGN_OR_RETURN(OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("out"));
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  // Give 10 input packets that contains the same string "Hello World!".
  for (int i = 0; i < 10; ++i) {
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "in", MakePacket<std::string>("Hello World!").At(Timestamp(i))));
  }
  // Close the input stream "in".
  MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
  mediapipe::Packet packet;
  // Get the output packets string.
  while (poller.Next(&packet)) {
    LOG(INFO) << packet.Get<std::string>();
  }
  return graph.WaitUntilDone();
}
}  // namespace mediapipe


#pragma warning(disable:4996)
// Marked as deprecated but is still used internally by mediapipe, so we'll keep using it as well for now...
#include "mediapipe/framework/api2/builder.h"
#pragma warning(default:4996)

#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/util/resource_util_custom.h"
#include "mediapipe/tasks/cc/components/utils/gate.h"
#include "mediapipe/calculators/util/association_calculator.pb.h"

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kFaceBlendshapesModel[] = "face_blendshapes.tflite";
constexpr char kInLandmarks[] = "face_blendshapes_in_landmarks.prototxt";
constexpr char kOutBlendshapes[] = "face_blendshapes_out.prototxt";
constexpr float kSimilarityThreshold = 0.1;
constexpr std::string_view kGeneratedGraph =
"face_blendshapes_generated_graph.pbtxt";

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kLandmarksName[] = "landmarks";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kImageSizeName[] = "image_size";
constexpr char kBlendshapesTag[] = "BLENDSHAPES";
constexpr char kBlendshapesName[] = "blendshapes";

static auto create_face_runner_task(std::string model_name, bool output_blendshape, bool output_face_geometry) {
	::mediapipe::api2::builder::Graph graph;
	auto& face_blendshapes_graph = graph.AddNode(
		"mediapipe.tasks.vision.face_landmarker.FaceBlendshapesGraph");
	auto& options =
		face_blendshapes_graph.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceBlendshapesGraphOptions>();
	options.mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", kTestDataDirectory, kFaceBlendshapesModel));

	graph[::mediapipe::api2::Input<mediapipe::NormalizedLandmarkList>(kLandmarksTag)].SetName(kLandmarksName) >>
		face_blendshapes_graph.In(kLandmarksTag);
	graph[::mediapipe::api2::Input<std::pair<int, int>>(kImageSizeTag)].SetName(kImageSizeName) >>
		face_blendshapes_graph.In(kImageSizeTag);
	face_blendshapes_graph.Out(kBlendshapesTag).SetName(kBlendshapesName) >>
		graph[::mediapipe::api2::Output<mediapipe::ClassificationList>(kBlendshapesTag)];

	return mediapipe::tasks::core::TaskRunner::Create(
		graph.GetConfig(), absl::make_unique<mediapipe::tasks::core::MediaPipeBuiltinOpResolver>());
}
namespace mediapipe::tasks::ios::test::vision::utils {
	absl::Status get_proto_from_pbtxt(const std::string file_path, google::protobuf::Message& proto);
}  // namespace mediapipe::tasks::ios::test::vision::utils

namespace mediapipe::tasks::ios::test::vision::utils {

	namespace {
		using ::google::protobuf::TextFormat;
	}

	absl::Status get_proto_from_pbtxt(const std::string file_path, google::protobuf::Message& proto) {

		std::ifstream file_input_stream(file_path);
		if (!file_input_stream.is_open()) return absl::InvalidArgumentError(
			"Cannot read input file.");

		std::stringstream strings_stream;
		strings_stream << file_input_stream.rdbuf();

		return TextFormat::ParseFromString(strings_stream.str(), &proto) ? absl::OkStatus() : absl::InvalidArgumentError(
			"Cannot read a valid proto from the input file.");
	}

}  // namespace mediapipe::tasks::ios::test::vision::utils

mediapipe::NormalizedLandmarkList GetLandmarks(const std::string &rootDir,absl::string_view filename) {
	// ::file::GetTextProto appears to be from some internal Google library that wasn't made open-source, so we can't use it.
	//CHECK_OK(::file::GetTextProto(mediapipe::file::JoinPath("./", kTestDataDirectory, filename),
	//	&landmarks, ::file::Defaults()))
	
	// Fortunately someone re-created it: https://github.com/google/mediapipe/pull/4455#discussion_r1204387674
	mediapipe::NormalizedLandmarkList landmarks;
	auto res = mediapipe::tasks::ios::test::vision::utils::get_proto_from_pbtxt(mediapipe::file::JoinPath(rootDir, kTestDataDirectory, filename),landmarks);
	std::cout << "Res: " << res.ok() << std::endl;

		return landmarks;
}


static void face(const std::string& rootDir)
{
	auto in_landmarks = GetLandmarks(rootDir,kInLandmarks);
	std::pair<int, int> in_image_size = { 820, 1024 };
	auto task_runner = create_face_runner_task("", true, false);
	if (!task_runner.ok())
		return;
	auto output_packets = (*task_runner)->Process(
		{ {kLandmarksName,
		  mediapipe::MakePacket<mediapipe::NormalizedLandmarkList>(std::move(in_landmarks))},
		 {kImageSizeName,
		  mediapipe::MakePacket<std::pair<int, int>>(std::move(in_image_size))} });

	const auto& actual_blendshapes =
		(*output_packets)[kBlendshapesName].Get<mediapipe::ClassificationList>();
	std::cout << "Blendshapes:" << std::endl;
	std::cout << actual_blendshapes.Utf8DebugString() << std::endl;
}
/*
static auto create_face_landmark_runner_task(std::string model_name, bool output_blendshape, bool output_face_geometry) {
	::mediapipe::api2::builder::Graph graph;
	auto& face_blendshapes_graph = graph.AddNode(
		"mediapipe.tasks.vision.face_landmark.FaceLandmarkGraph");
	auto& options =
		face_blendshapes_graph.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceBlendshapesGraphOptions>();
	options.mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", kTestDataDirectory, kFaceBlendshapesModel));

	graph[::mediapipe::api2::Input<mediapipe::NormalizedLandmarkList>(kLandmarksTag)].SetName(kLandmarksName) >>
		face_blendshapes_graph.In(kLandmarksTag);
	graph[::mediapipe::api2::Input<std::pair<int, int>>(kImageSizeTag)].SetName(kImageSizeName) >>
		face_blendshapes_graph.In(kImageSizeTag);
	face_blendshapes_graph.Out(kBlendshapesTag).SetName(kBlendshapesName) >>
		graph[::mediapipe::api2::Output<mediapipe::ClassificationList>(kBlendshapesTag)];

	return mediapipe::tasks::core::TaskRunner::Create(
		graph.GetConfig(), absl::make_unique<mediapipe::tasks::core::MediaPipeBuiltinOpResolver>());
}
*/

////////////////////////////////////////

#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"

constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kFaceRectsTag[] = "FACE_RECTS";
constexpr char kFaceRectsNextFrameTag[] = "FACE_RECTS_NEXT_FRAME";
constexpr char kExpandedFaceRectsTag[] = "EXPANDED_FACE_RECTS";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kLoopTag[] = "LOOP";
constexpr char kPrevLoopTag[] = "PREV_LOOP";
constexpr char kMainTag[] = "MAIN";
constexpr char kIterableTag[] = "ITERABLE";
constexpr char kFaceLandmarksTag[] = "FACE_LANDMARKS";
constexpr char kFaceGeometryTag[] = "FACE_GEOMETRY";
constexpr char kEnvironmentTag[] = "ENVIRONMENT";
constexpr char kSizeTag[] = "SIZE";
constexpr char kVectorTag[] = "VECTOR";
constexpr char kItemTag[] = "ITEM";
constexpr char kNormFilteredLandmarksTag[] = "NORM_FILTERED_LANDMARKS";
constexpr char kFaceDetectorTFLiteName[] = "face_detector.tflite";
constexpr char kFaceLandmarksDetectorTFLiteName[] =
"face_landmarks_detector.tflite";
constexpr char kFaceBlendshapeTFLiteName[] = "face_blendshapes.tflite";
constexpr char kFaceGeometryPipelineMetadataName[] =
"geometry_pipeline_metadata_landmarks.binarypb";

struct FaceLandmarkerOutputs {
	mediapipe::api2::builder::Source<std::vector<mediapipe::NormalizedLandmarkList>> landmark_lists;
	mediapipe::api2::builder::Source<std::vector<mediapipe::NormalizedRect>> face_rects_next_frame;
	mediapipe::api2::builder::Source<std::vector<mediapipe::NormalizedRect>> face_rects;
	mediapipe::api2::builder::Source<std::vector<mediapipe::Detection>> detections;
	std::optional<mediapipe::api2::builder::Source<std::vector<mediapipe::ClassificationList>>> face_blendshapes;
	std::optional<mediapipe::api2::builder::Source<std::vector<mediapipe::tasks::vision::face_geometry::proto::FaceGeometry>>> face_geometry;
	mediapipe::api2::builder::Source<mediapipe::Image> image;
};

// Sets the base options in the sub tasks.
absl::Status SetSubTaskBaseOptions(const mediapipe::tasks::core::ModelAssetBundleResources& resources,
	mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions* options,
	bool is_copy) {
	auto* face_detector_graph_options =
		options->mutable_face_detector_graph_options();
	if (!face_detector_graph_options->base_options().has_model_asset()) {
		ASSIGN_OR_RETURN(const auto face_detector_file,
			resources.GetFile(kFaceDetectorTFLiteName));
		mediapipe::tasks::metadata::SetExternalFile(face_detector_file,
			face_detector_graph_options->mutable_base_options()
			->mutable_model_asset(),
			is_copy);
	}
	face_detector_graph_options->mutable_base_options()
		->mutable_acceleration()
		->CopyFrom(options->base_options().acceleration());
	face_detector_graph_options->mutable_base_options()->set_use_stream_mode(
		options->base_options().use_stream_mode());
	auto* face_landmarks_detector_graph_options =
		options->mutable_face_landmarks_detector_graph_options();
	if (!face_landmarks_detector_graph_options->base_options()
		.has_model_asset()) {
		ASSIGN_OR_RETURN(const auto face_landmarks_detector_file,
			resources.GetFile(kFaceLandmarksDetectorTFLiteName));
		mediapipe::tasks::metadata::SetExternalFile(
			face_landmarks_detector_file,
			face_landmarks_detector_graph_options->mutable_base_options()
			->mutable_model_asset(),
			is_copy);
	}
	face_landmarks_detector_graph_options->mutable_base_options()
		->mutable_acceleration()
		->CopyFrom(options->base_options().acceleration());
	face_landmarks_detector_graph_options->mutable_base_options()
		->set_use_stream_mode(options->base_options().use_stream_mode());

	absl::StatusOr<absl::string_view> face_blendshape_model =
		resources.GetFile(kFaceBlendshapeTFLiteName);
	if (face_blendshape_model.ok()) {
		mediapipe::tasks::metadata::SetExternalFile(*face_blendshape_model,
			face_landmarks_detector_graph_options
			->mutable_face_blendshapes_graph_options()
			->mutable_base_options()
			->mutable_model_asset(),
			is_copy);
		face_landmarks_detector_graph_options
			->mutable_face_blendshapes_graph_options()
			->mutable_base_options()
			->mutable_acceleration()
			->mutable_xnnpack();
		LOG(WARNING) << "Face blendshape model contains CPU only ops. Sets "
			<< "FaceBlendshapesGraph acceleration to Xnnpack.";
	}

	return absl::OkStatus();
}

#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/environment.pb.h"
#include "mediapipe/util/graph_builder_utils.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/calculators/core/clip_vector_size_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/calculators/util/collection_has_min_size_calculator.pb.h"
#if 0
namespace mediapipe {
	namespace tasks {
		namespace vision {
			namespace face_landmarker {

class FaceLandmarkerGraph : public mediapipe::tasks::core::ModelTaskGraph {
public:
	absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
		mediapipe::SubgraphContext* sc) override {
		mediapipe::api2::builder::Graph graph;
		bool output_geometry = HasOutput(sc->OriginalNode(), kFaceGeometryTag);
		if (sc->Options<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>()
			.base_options()
			.has_model_asset()) {
			ASSIGN_OR_RETURN(
				const auto * model_asset_bundle_resources,
				CreateModelAssetBundleResources<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>(sc));
			// Copies the file content instead of passing the pointer of file in
			// memory if the subgraph model resource service is not available.
			MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
				*model_asset_bundle_resources,
				sc->MutableOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>(),
				!sc->Service(::mediapipe::tasks::core::kModelResourcesCacheService)
				.IsAvailable()));
			if (output_geometry) {
				// Set the face geometry metadata file for
				// FaceGeometryFromLandmarksGraph.
				ASSIGN_OR_RETURN(auto face_geometry_pipeline_metadata_file,
					model_asset_bundle_resources->GetFile(
						kFaceGeometryPipelineMetadataName));
				mediapipe::tasks::metadata::SetExternalFile(face_geometry_pipeline_metadata_file,
					sc->MutableOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>()
					->mutable_face_geometry_graph_options()
					->mutable_geometry_pipeline_options()
					->mutable_metadata_file());
			}
		}
		std::optional<mediapipe::api2::builder::SidePacket<mediapipe::tasks::vision::face_geometry::proto::Environment>> environment;
		if (HasSideInput(sc->OriginalNode(), kEnvironmentTag)) {
			environment = std::make_optional<>(
				graph.SideIn(kEnvironmentTag).Cast<mediapipe::tasks::vision::face_geometry::proto::Environment>());
		}
		bool output_blendshapes = HasOutput(sc->OriginalNode(), kBlendshapesTag);
		if (output_blendshapes && !sc->Options<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>()
			.face_landmarks_detector_graph_options()
			.has_face_blendshapes_graph_options()) {
			return absl::InvalidArgumentError(absl::StrFormat(
				"BLENDSHAPES Tag and blendshapes model must be both set. Get "
				"BLENDSHAPES is set: %v, blendshapes "
				"model "
				"is set: %v",
				output_blendshapes,
				sc->Options<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>()
				.face_landmarks_detector_graph_options()
				.has_face_blendshapes_graph_options()));
		}
		std::optional<mediapipe::api2::builder::Source<mediapipe::NormalizedRect>> norm_rect_in;
		if (HasInput(sc->OriginalNode(), kNormRectTag)) {
			norm_rect_in = graph.In(kNormRectTag).Cast<mediapipe::NormalizedRect>();
		}
		ASSIGN_OR_RETURN(
			auto outs,
			BuildFaceLandmarkerGraph(
				*sc->MutableOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>(),
				graph[mediapipe::api2::Input<mediapipe::Image>(kImageTag)], norm_rect_in, environment,
				output_blendshapes, output_geometry, graph));
		outs.landmark_lists >>
			graph[mediapipe::api2::Output<std::vector<mediapipe::NormalizedLandmarkList>>(kNormLandmarksTag)];
		outs.face_rects_next_frame >>
			graph[mediapipe::api2::Output<std::vector<mediapipe::NormalizedRect>>(kFaceRectsNextFrameTag)];
		outs.face_rects >>
			graph[mediapipe::api2::Output<std::vector<mediapipe::NormalizedRect>>(kFaceRectsTag)];
		outs.detections >> graph[mediapipe::api2::Output<std::vector<mediapipe::Detection>>(kDetectionsTag)];
		outs.image >> graph[mediapipe::api2::Output<mediapipe::Image>(kImageTag)];
		if (outs.face_blendshapes) {
			*outs.face_blendshapes >>
				graph[mediapipe::api2::Output<std::vector<mediapipe::ClassificationList>>(kBlendshapesTag)];
		}
		if (outs.face_geometry) {
			*outs.face_geometry >>
				graph[mediapipe::api2::Output<std::vector<mediapipe::tasks::vision::face_geometry::proto::FaceGeometry>>(kFaceGeometryTag)];
		}

		// TODO remove when support is fixed.
		// As mediapipe GraphBuilder currently doesn't support configuring
		// InputStreamInfo, modifying the CalculatorGraphConfig proto directly.
		mediapipe::CalculatorGraphConfig config = graph.GetConfig();
		for (int i = 0; i < config.node_size(); ++i) {
			if (config.node(i).calculator() == "PreviousLoopbackCalculator") {
				auto* info = config.mutable_node(i)->add_input_stream_info();
				info->set_tag_index(kLoopTag);
				info->set_back_edge(true);
				break;
			}
		}
		return config;
	}

private:
	// Adds a mediapipe face landmarker graph into the provided builder::Graph
	// instance.
	//
	// tasks_options: the mediapipe tasks module FaceLandmarkerGraphOptions.
	// image_in: (mediapipe::Image) stream to run face landmark detection on.
	// graph: the mediapipe graph instance to be updated.
	absl::StatusOr<FaceLandmarkerOutputs> BuildFaceLandmarkerGraph(
		mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions& tasks_options, mediapipe::api2::builder::Source<mediapipe::Image> image_in,
		std::optional<mediapipe::api2::builder::Source<mediapipe::NormalizedRect>> norm_rect_in,
		std::optional<mediapipe::api2::builder::SidePacket<mediapipe::tasks::vision::face_geometry::proto::Environment>> environment,
		bool output_blendshapes, bool output_geometry, mediapipe::api2::builder::Graph& graph) {
		const int max_num_faces =
			tasks_options.face_detector_graph_options().num_faces();

		auto& face_detector =
			graph.AddNode("mediapipe.tasks.vision.face_detector.FaceDetectorGraph");
		face_detector.GetOptions<mediapipe::tasks::vision::face_detector::proto::FaceDetectorGraphOptions>().Swap(
			tasks_options.mutable_face_detector_graph_options());
		const auto& face_detector_options =
			face_detector.GetOptions<mediapipe::tasks::vision::face_detector::proto::FaceDetectorGraphOptions>();
		auto& clip_face_rects =
			graph.AddNode("ClipNormalizedRectVectorSizeCalculator");
		clip_face_rects.GetOptions<mediapipe::ClipVectorSizeCalculatorOptions>()
			.set_max_vec_size(max_num_faces);
		auto clipped_face_rects = clip_face_rects.Out("");

		auto& face_landmarks_detector_graph = graph.AddNode(
			"mediapipe.tasks.vision.face_landmarker."
			"MultiFaceLandmarksDetectorGraph");
		face_landmarks_detector_graph
			.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarksDetectorGraphOptions>()
			.Swap(tasks_options.mutable_face_landmarks_detector_graph_options());
		image_in >> face_landmarks_detector_graph.In(kImageTag);
		clipped_face_rects >> face_landmarks_detector_graph.In(kNormRectTag);

		mediapipe::api2::builder::Source<std::vector<mediapipe::NormalizedLandmarkList>> face_landmarks =
			face_landmarks_detector_graph.Out(kNormLandmarksTag)
			.Cast<std::vector<mediapipe::NormalizedLandmarkList>>();
		auto face_rects_for_next_frame =
			face_landmarks_detector_graph.Out(kFaceRectsNextFrameTag)
			.Cast<std::vector<mediapipe::NormalizedRect>>();

		auto& image_properties = graph.AddNode("ImagePropertiesCalculator");
		image_in >> image_properties.In(kImageTag);
		auto image_size = image_properties.Out(kSizeTag);

		// Apply smoothing filter only on the single face landmarks, because
		// landmarks smoothing calculator doesn't support multiple landmarks yet.
		if (face_detector_options.num_faces() == 1) {
			face_landmarks_detector_graph
				.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarksDetectorGraphOptions>()
				.set_smooth_landmarks(tasks_options.base_options().use_stream_mode());
		}
		else if (face_detector_options.num_faces() > 1 &&
			face_landmarks_detector_graph
			.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarksDetectorGraphOptions>()
			.smooth_landmarks()) {
			return absl::InvalidArgumentError(
				"Currently face landmarks smoothing only support a single face.");
		}

		if (tasks_options.base_options().use_stream_mode()) {
			auto& previous_loopback = graph.AddNode("PreviousLoopbackCalculator");
			image_in >> previous_loopback.In(kMainTag);
			auto prev_face_rects_from_landmarks =
				previous_loopback[mediapipe::api2::Output<std::vector<mediapipe::NormalizedRect>>(kPrevLoopTag)];

			auto& min_size_node =
				graph.AddNode("NormalizedRectVectorHasMinSizeCalculator");
			prev_face_rects_from_landmarks >> min_size_node.In(kIterableTag);
			min_size_node.GetOptions<mediapipe::CollectionHasMinSizeCalculatorOptions>()
				.set_min_size(max_num_faces);
			auto has_enough_faces = min_size_node.Out("").Cast<bool>();

			// While in stream mode, skip face detector graph when we successfully
			// track the faces from the last frame.
			auto image_for_face_detector =
				mediapipe::tasks::components::utils::DisallowIf(image_in, has_enough_faces, graph);
			image_for_face_detector >> face_detector.In(kImageTag);
			std::optional<mediapipe::api2::builder::Source<mediapipe::NormalizedRect>> norm_rect_in_for_face_detector;
			if (norm_rect_in) {
				norm_rect_in_for_face_detector =
					mediapipe::tasks::components::utils::DisallowIf(norm_rect_in.value(), has_enough_faces, graph);
			}
			if (norm_rect_in_for_face_detector) {
				*norm_rect_in_for_face_detector >> face_detector.In("NORM_RECT");
			}
			auto expanded_face_rects_from_face_detector =
				face_detector.Out(kExpandedFaceRectsTag);
			auto& face_association = graph.AddNode("AssociationNormRectCalculator");
			face_association.GetOptions<mediapipe::AssociationCalculatorOptions>()
				.set_min_similarity_threshold(
					tasks_options.min_tracking_confidence());
			prev_face_rects_from_landmarks >>
				face_association[mediapipe::api2::Input<std::vector<mediapipe::NormalizedRect>>::Multiple("")][0];
			expanded_face_rects_from_face_detector >>
				face_association[mediapipe::api2::Input<std::vector<mediapipe::NormalizedRect>>::Multiple("")][1];
			auto face_rects = face_association.Out("");
			face_rects >> clip_face_rects.In("");
			// Back edge.
			face_rects_for_next_frame >> previous_loopback.In(kLoopTag);
		}
		else {
			// While not in stream mode, the input images are not guaranteed to be
			// in series, and we don't want to enable the tracking and rect
			// associations between input images. Always use the face detector
			// graph.
			image_in >> face_detector.In(kImageTag);
			if (norm_rect_in) {
				*norm_rect_in >> face_detector.In(kNormRectTag);
			}
			auto face_rects = face_detector.Out(kExpandedFaceRectsTag);
			face_rects >> clip_face_rects.In("");
		}

		// Optional blendshape output.
		std::optional<mediapipe::api2::builder::Source<std::vector<mediapipe::ClassificationList>>> blendshapes;
		if (output_blendshapes) {
			blendshapes = std::make_optional<>(
				face_landmarks_detector_graph.Out(kBlendshapesTag)
				.Cast<std::vector<mediapipe::ClassificationList>>());
		}

		// Optional face geometry output.
		std::optional<mediapipe::api2::builder::Source<std::vector<mediapipe::tasks::vision::face_geometry::proto::FaceGeometry>>> face_geometry;
		if (output_geometry) {
			auto& face_geometry_from_landmarks = graph.AddNode(
				"mediapipe.tasks.vision.face_geometry."
				"FaceGeometryFromLandmarksGraph");
			face_geometry_from_landmarks
				.GetOptions<mediapipe::tasks::vision::face_geometry::proto::FaceGeometryGraphOptions>()
				.Swap(tasks_options.mutable_face_geometry_graph_options());
			if (environment.has_value()) {
				*environment >> face_geometry_from_landmarks.SideIn(kEnvironmentTag);
			}
			face_landmarks >> face_geometry_from_landmarks.In(kFaceLandmarksTag);
			image_size >> face_geometry_from_landmarks.In(kImageSizeTag);
			face_geometry = face_geometry_from_landmarks.Out(kFaceGeometryTag)
				.Cast<std::vector<mediapipe::tasks::vision::face_geometry::proto::FaceGeometry>>();
		}

		// TODO: Replace PassThroughCalculator with a calculator that
		// converts the pixel data to be stored on the target storage (CPU vs
		// GPU).
		auto& pass_through = graph.AddNode("PassThroughCalculator");
		image_in >> pass_through.In("");

		return { {
				/* landmark_lists= */ face_landmarks,
				/* face_rects_next_frame= */
				face_rects_for_next_frame,
				/* face_rects= */
				face_detector.Out(kFaceRectsTag).Cast<std::vector<mediapipe::NormalizedRect>>(),
				/* face_detections */
				face_detector.Out(kDetectionsTag).Cast<std::vector<mediapipe::Detection>>(),
				/* face_blendshapes= */ blendshapes,
				/* face_geometry= */ face_geometry,
				/* image= */
				pass_through[mediapipe::api2::Output<mediapipe::Image>("")],
			} };
	}
};

REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerGraph);
}  // namespace face_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
#endif
constexpr char kImageName[] = "image";
constexpr char kNormRectName[] = "norm_rect";
constexpr char kNormLandmarksName[] = "norm_landmarks";
constexpr char kFaceGeometryName[] = "face_geometry";
constexpr char kFaceLandmarkerModelBundleName[] = "face_landmarker_v2.task";
constexpr char kFaceLandmarkerWithBlendshapesModelBundleName[] =
"face_landmarker_v2_with_blendshapes.task";
static absl::lts_20230125::StatusOr<std::unique_ptr<mediapipe::tasks::core::TaskRunner,std::default_delete<mediapipe::tasks::core::TaskRunner>>> create_face_landmarker_runner_task(std::string model_name, bool output_blendshape, bool output_face_geometry) {
	//::mediapipe::api2::builder::Graph graph;
	//auto& face_blendshapes_graph = graph.AddNode(
	//	"mediapipe.tasks.vision.face_landmarker.FaceBlendshapesGraph");
	/*auto& options =
		face_blendshapes_graph.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceBlendshapesGraphOptions>();
	options.mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", kTestDataDirectory, kFaceBlendshapesModel));

	graph[::mediapipe::api2::Input<mediapipe::NormalizedLandmarkList>(kLandmarksTag)].SetName(kLandmarksName) >>
		face_blendshapes_graph.In(kLandmarksTag);
	graph[::mediapipe::api2::Input<std::pair<int, int>>(kImageSizeTag)].SetName(kImageSizeName) >>
		face_blendshapes_graph.In(kImageSizeTag);
	face_blendshapes_graph.Out(kBlendshapesTag).SetName(kBlendshapesName) >>
		graph[::mediapipe::api2::Output<mediapipe::ClassificationList>(kBlendshapesTag)];
		*/

	::mediapipe::api2::builder::Graph graph;

	auto& face_landmarker = graph.AddNode(
		"mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph");

	// TODO: This overwrites the global kTestDataDirectory. For some reason with this path "mediapipe/" is added as prefix already.
	std::string kTestDataDirectory = "/tasks/testdata/vision/";

	auto* options = &face_landmarker.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>();
	options->mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", kTestDataDirectory, model_name));
	options->mutable_face_detector_graph_options()->set_num_faces(1);
	options->mutable_base_options()->set_use_stream_mode(true);

	//options->mutable_face_landmarks_detector_graph_options()->mutable_face_blendshapes_graph_options()->mutable_model_asset()->set_file_name(
	//	mediapipe::file::JoinPath("./", kTestDataDirectory, "face_blendshapes.tflite"));

	graph[::mediapipe::api2::Input<mediapipe::Image>(kImageTag)].SetName(kImageName) >>
		face_landmarker.In(kImageTag);
	//graph[::mediapipe::api2::Input<mediapipe::NormalizedRect>(kNormRectTag)].SetName(kNormRectName) >>
	//	face_landmarker.In(kNormRectTag);

	face_landmarker.Out(kBlendshapesTag).SetName(kBlendshapesName) >>
		graph[::mediapipe::api2::Output<mediapipe::ClassificationList>(kBlendshapesTag)];
	/*face_landmarker.Out(kNormLandmarksTag).SetName(kNormLandmarksName) >>
		graph[::mediapipe::api2::Output<std::vector<mediapipe::NormalizedLandmarkList>>(kNormLandmarksTag)];
	if (output_blendshape) {
		face_landmarker.Out(kBlendshapesTag).SetName(kBlendshapesName) >>
			graph[::mediapipe::api2::Output<std::vector<mediapipe::ClassificationList>>(kBlendshapesTag)];
	}
	if (output_face_geometry) {
		face_landmarker.Out(kFaceGeometryTag).SetName(kFaceGeometryName) >>
			graph[::mediapipe::api2::Output<std::vector<mediapipe::tasks::vision::face_geometry::proto::FaceGeometry>>(kFaceGeometryTag)];
	}*/

#if 0
	bool output_blendshapes = HasOutput(sc->OriginalNode(), kBlendshapesTag);
	if (output_blendshapes && !sc->Options<FaceLandmarkerGraphOptions>()
		.face_landmarks_detector_graph_options()
		.has_face_blendshapes_graph_options()) {
		return absl::InvalidArgumentError(absl::StrFormat(
			"BLENDSHAPES Tag and blendshapes model must be both set. Get "
			"BLENDSHAPES is set: %v, blendshapes "
			"model "
			"is set: %v",
			output_blendshapes,
			sc->Options<FaceLandmarkerGraphOptions>()
			.face_landmarks_detector_graph_options()
			.has_face_blendshapes_graph_options()));
	}
#endif

	return mediapipe::tasks::core::TaskRunner::Create(
		graph.GetConfig(),
		absl::make_unique<mediapipe::tasks::core::MediaPipeBuiltinOpResolver>());

	/*::mediapipe::api2::builder::Graph graph;
	auto& face_blendshapes_graph = graph.AddNode(
		"mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph");
	auto& options =
		face_blendshapes_graph.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceBlendshapesGraphOptions>();
	options.mutable_base_options()->mutable_model_asset()->set_file_name(
		mediapipe::file::JoinPath("./", kTestDataDirectory, "face_landmarker.task"));

	graph[::mediapipe::api2::Input<mediapipe::Image>(kImageTag)].SetName("image") >>
		face_blendshapes_graph.In(kImageTag);

	face_blendshapes_graph.Out(kNormLandmarksTag).SetName(kLandmarksName) >>
		graph[::mediapipe::api2::Output<mediapipe::NormalizedLandmarkList>(kNormLandmarksTag)];

	auto taskRunner = mediapipe::tasks::core::TaskRunner::Create(
		graph.GetConfig(), absl::make_unique<mediapipe::tasks::core::MediaPipeBuiltinOpResolver>());
	std::cout << "OK: " << taskRunner.ok() << std::endl;
	std::cout << "---" << std::endl;
	return taskRunner;*/
	
	
	/*::mediapipe::api2::builder::Graph graph;
	std::cout << "a" << std::endl;
	auto& face_landmarker = graph.AddNode(
		"mediapipe.tasks.vision.face_landmarker."
		"FaceLandmarkerGraph");
	//auto& options =
	//	face_blendshapes_graph.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>();
	//options.mutable_base_options()->mutable_model_asset()->set_file_name(
	//	mediapipe::file::JoinPath("./", kTestDataDirectory, kFaceBlendshapesModel));

	graph[::mediapipe::api2::Input<mediapipe::Image>(kImageTag)].SetName("image") >>
		face_landmarker.In(kImageTag);

	face_landmarker.Out(kNormLandmarksTag).SetName(kLandmarksName) >>
		graph[::mediapipe::api2::Output<mediapipe::NormalizedLandmarkList>(kNormLandmarksTag)];
	return mediapipe::tasks::core::TaskRunner::Create(
		graph.GetConfig(), absl::make_unique<mediapipe::tasks::core::MediaPipeBuiltinOpResolver>());
		*/
	
	//::mediapipe::api2::builder::Graph graph;
	//auto& face_blendshapes_graph = graph.AddNode< FaceLandmarkerGraph>();
	/*auto& options =
		face_blendshapes_graph.GetOptions<mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions>();
	//options.mutable_base_options()->mutable_model_asset()->set_file_name(
	//	mediapipe::file::JoinPath("./", kTestDataDirectory, kFaceBlendshapesModel));

	graph[::mediapipe::api2::Input<mediapipe::Image>(kImageTag)].SetName("image") >>
		face_blendshapes_graph.In(kImageTag);

	face_blendshapes_graph.Out(kNormLandmarksTag).SetName(kLandmarksName) >>
		graph[::mediapipe::api2::Output<mediapipe::NormalizedLandmarkList>(kNormLandmarksTag)];*/

	//return mediapipe::tasks::core::TaskRunner::Create(
	//	graph.GetConfig(), absl::make_unique<mediapipe::tasks::core::MediaPipeBuiltinOpResolver>());

	/*mediapipe::SubgraphContext subgraphContext{};
	FaceLandmarkerGraph graph{  };
	auto config = graph.GetConfig(&subgraphContext);
	if (!config.ok()) {
		absl::lts_20230125::StatusOr<std::unique_ptr<mediapipe::tasks::core::TaskRunner, std::default_delete<mediapipe::tasks::core::TaskRunner>>> status {};
		return status;
	}
	return mediapipe::tasks::core::TaskRunner::Create(
		*config, absl::make_unique<mediapipe::tasks::core::MediaPipeBuiltinOpResolver>());*/
}

#include <opencv2/opencv.hpp>
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
constexpr char kPortraitImageName[] = "portrait.jpg";

// Helper function to construct NormalizeRect proto.
mediapipe::NormalizedRect MakeNormRect(float x_center, float y_center, float width,
	float height, float rotation) {
	mediapipe::NormalizedRect face_rect;
	face_rect.set_x_center(x_center);
	face_rect.set_y_center(y_center);
	face_rect.set_width(width);
	face_rect.set_height(height);
	face_rect.set_rotation(rotation);
	return face_rect;
}

#include <sstream>
static void face_landmarker()
{
	auto task_runner = create_face_landmarker_runner_task(kFaceLandmarkerWithBlendshapesModelBundleName, true, true);
	std::cout << "ok: " << task_runner.ok() << std::endl;
	if (!task_runner.ok())
		return;

	{
		auto image = mediapipe::tasks::vision::DecodeImageFromFile(mediapipe::file::JoinPath("E:/projects/test_mediapipe_wrapper/x64/Debug/",
			"portrait.jpg"));
		std::cout << "Image ok: " << image.ok() << std::endl;
		auto output_packets = (*task_runner)->Process(
			{ {kImageName, mediapipe::MakePacket<mediapipe::Image>(std::move(*image))}
			});
		std::cout << "output_packets: " << output_packets.ok() << std::endl;

		const auto& actual_blendshapes =
			(*output_packets)[kBlendshapesName].Get<std::vector<mediapipe::ClassificationList>>();
		std::cout << "ClassificationList (" << actual_blendshapes.size() << ")" << std::endl;
		//struct Classification {
		//	uint32_t index;
		//	float score;
		//	std::string label;
		//};

		std::stringstream ss;
		ss << "local blendShapes = {\n";
		for (auto& bs : actual_blendshapes[0].classification()) {
			std::cout << bs.Utf8DebugString() << std::endl;
			ss << "\t{\n";
			ss << "\t\tscore = " << bs.score() << ",\n";
			ss << "\t\tindex = " << bs.index() << ",\n";
			ss << "\t\tlabel = \"" << bs.label() << "\"\n";
			ss << "\t},\n";
		}
		ss << "}\n";
		std::cout << ss.str() << std::endl;

	}



	cv::VideoCapture cap;
	//(deviceId);
	cap.open("E:/projects/test_mediapipe_wrapper/x64/Debug/portrait.png"); // portrait.jpg
	
	if (!cap.isOpened()) {
		std::cerr << "Could not open device #0. Is a camera/webcam attached?" << std::endl;
		return;
	}
	cv::Mat frame_bgr;
	uint32_t m_frame_timestamp = 0;
	if (cap.read(frame_bgr)) {
		// Convert frame from BGR to RGB
		cv::Mat frame_rgb;
		cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);

		auto image_format = mediapipe::ImageFormat::SRGB;
		auto data = frame_rgb.data;
		auto width = frame_rgb.cols;
		auto height = frame_rgb.rows;

		if (data == nullptr) {
			//LOG(INFO) << __FUNCTION__ << " input data is nullptr!";
			return;
		}
		if (!mediapipe::ImageFormat::Format_IsValid(image_format)) {
			//LOG(INFO) << __FUNCTION__ << " input image format (" << image_format << ") is invalid!";
			return;
		}

		auto input_frame_for_input = std::make_shared<mediapipe::ImageFrame>();
		auto mp_image_format = static_cast<mediapipe::ImageFormat::Format>(image_format);
		input_frame_for_input->CopyPixelData(mp_image_format, width, height, data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

		mediapipe::Image *img = new mediapipe::Image { input_frame_for_input };
		auto packet = mediapipe::Adopt(img);
		auto output_packets = (*task_runner)->Process(
			{ {"image",
			  packet}
			});
		if (output_packets.ok())
			std::cout << "ok" << std::endl;
		std::cout << "Msg: " << output_packets.status().message() << std::endl;
		++m_frame_timestamp;

		const auto& actual_blendshapes =
			(*output_packets)[kBlendshapesName].Get<std::vector<mediapipe::ClassificationList>>();
		std::cout << "ClassificationList ("<< actual_blendshapes.size()<<")"<< std::endl;
		//struct Classification {
		//	uint32_t index;
		//	float score;
		//	std::string label;
		//};

		std::stringstream ss;
		ss << "local blendShapes = {\n";
		for (auto& bs : actual_blendshapes[0].classification()) {
			std::cout << bs.Utf8DebugString() << std::endl;
			ss << "\t{\n";
			ss << "\t\tscore = " << bs.score() << ",\n";
			ss << "\t\tindex = " << bs.index() << ",\n";
			ss << "\t\tlabel = \"" << bs.label() << "\"\n";
			ss << "\t},\n";
		}
		ss << "}\n";
		std::cout << ss.str() << std::endl;
	}

}

int main(int argc, char** argv) {
	/*std::string rootPath = "E:/projects/test_mediapipe_py/deps/mediapipe/bazel-bin/mediapipe/examples/desktop/holistic_tracking/";
	mediapipe::SetCustomGlobalResourceProvider([rootPath](const std::string& path, std::string* output, bool read_as_binary) {
		auto newPath = rootPath + path;
		std::cout << "LOAD FILE: " << newPath << std::endl;
		return mediapipe::file::GetContents(newPath, output, read_as_binary);
		});
	//LOAD FILE: E:/projects/test_mediapipe_py/deps/mediapipe/bazel-bin/mediapipe/examples/desktop/holistic_tracking/./mediapipe/tasks/testdata/vision/face_blendshapes.tflite
	//auto task = create_face_runner_task("", true, false);
	//face();
	*/
	face_landmarker();
	std::cout << "Done" << std::endl;
	//std::cout << "AAAAA" << task.ok() << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds{12});
	//google::InitGoogleLogging(argv[0]);
	//CHECK(mediapipe::PrintHelloWorld().ok());
	return 0;
}

#include "absl/flags/flag.h"

extern absl::Flag<std::string> FLAGS_resource_root_dir;

extern "C" {
	__declspec(dllexport)
		void test_exp(const char *rootPath) {
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
			std::cout << "LOAD FILE: " << newPath << std::endl;
			return mediapipe::file::GetContents(newPath, output, read_as_binary);
		});*/

		face_landmarker();
		//face(srootPath);
	}
}
