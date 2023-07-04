licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_shared_library(
    name = "mediapipe_pragma_wrapper",
    user_link_flags = [
        "-Wl",
    ],
    deps = [
        "mediapipe_pragma_wrapper_static",
    ],
)

cc_library(
    name = "mediapipe_pragma_wrapper_static",
    srcs = ["hello_world.cc","face_landmarker_graph.cc"],
    data = [
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/holistic_landmark:hand_recrop.tflite",
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/holistic_tracking:holistic_tracking_cpu_graph_deps",
		"//mediapipe/framework/api2:builder",
		"//mediapipe/tasks/cc/core:external_file_handler",
		"//mediapipe/tasks/metadata:metadata_schema_cc",
		"//mediapipe/tasks/cc/core:base_task_api",
		"//mediapipe/tasks/cc/core:base_options",
		"//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_graph",
		"//mediapipe/tasks/cc/vision/face_geometry:face_geometry_from_landmarks_graph",
    ],
)
