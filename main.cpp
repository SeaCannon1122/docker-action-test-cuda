#include "yolos/core/version.hpp"
#include <atomic>
#include <csignal>
#include <unistd.h>
#include <librealsense2/rs.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "yolos/tasks/detection.hpp"


std::atomic<bool> keep_running{true};

void handle_signal(int signum) {
  keep_running.store(false);
}

int main(int argc, char** argv)
{
    printf("test");

    if (argc != 8)
    {
        printf("usage: ./camera_app <ip> <port> <width> <height> <fps> <engine_file> <labels_file>\n");
        return -1;
    }

    std::string ip = std::string(argv[1]);
    int port = atoi(argv[2]);
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);
    int fps = atoi(argv[5]);
    std::string engine = std::string(argv[6]);
    std::string labels = std::string(argv[7]);


  gst_init(NULL, NULL);

  yolos::det::YOLODetector hazmat_detector(engine, labels);

  std::string gst_pipeline_desc = 
    "appsrc name=appsrc format=time "
    "caps=video/x-raw,format=BGR,width=" + std::to_string(width) + 
    ",height=" + std::to_string(height) + ",framerate=" + std::to_string(fps) + "/1 "
    "! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency key-int-max=" + std::to_string(fps*2) + " "
    "! rtph264pay config-interval=1 ! udpsink host=" + ip + " port=" + std::to_string(port) + " sync=false";

  GstElement* gst_pipeline;
  GstElement* gst_appsrc;  

  GError *error = nullptr;
  gst_pipeline = gst_parse_launch(gst_pipeline_desc.c_str(), &error);

  if (!gst_pipeline || error) {
    printf("Failed to create Gstreamer pipeline");
    return -1;
  }

  gst_appsrc = gst_bin_get_by_name(GST_BIN(gst_pipeline), "appsrc");
  gst_element_set_state(gst_pipeline, GST_STATE_PLAYING);

  rs2::config rs_cfg; rs2::pipeline rs_pipeline;

  rs_cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
  rs_pipeline.start(rs_cfg);
  printf("Streaming camera on %s:%d", ip.c_str(), port);

  signal(SIGTERM, handle_signal);
  signal(SIGINT, handle_signal);

  while (keep_running.load())
  {
    rs2::frameset frames = rs_pipeline.wait_for_frames();
    rs2::video_frame color_frame = frames.get_color_frame();
    cv::Mat image(height, width, CV_8UC3, (void*)color_frame.get_data());

    std::vector<yolos::det::Detection> hazmat_results, paintroller_results;

    hazmat_results = hazmat_detector.detect(image);

    GstBuffer *buffer = gst_buffer_new_allocate(nullptr, 3 * width * height, nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, image.data, 3 * width * height);

    cv::Mat mapped_image(height, width, CV_8UC3, (void*)map.data);
    hazmat_detector.drawDetections(mapped_image, hazmat_results);

    gst_buffer_unmap(buffer, &map);
    gst_app_src_push_buffer(GST_APP_SRC(gst_appsrc), buffer);
  }
 
  rs_pipeline.stop();
  printf("Closed camera");

  gst_element_set_state(gst_pipeline, GST_STATE_NULL);
  gst_object_unref(gst_pipeline);

}


  