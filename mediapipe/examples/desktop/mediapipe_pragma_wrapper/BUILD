load("//mediapipe/examples/desktop/mediapipe_pragma_wrapper:windows_dll_library.bzl", "windows_dll_library")

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "mediapipe_pragma_wrapper",
    srcs = ["hello_world.cc","face_landmarker_graph.cc"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_models"
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
		"//mediapipe/tasks/cc/vision/utils:image_utils",
    ],
    linkshared = 1,
    copts = ["/DCOMPILING_DLL"],
)
