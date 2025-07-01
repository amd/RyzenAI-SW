/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 */

#include "common-sample-utils.h"

#ifdef _WIN32
#include <Mferror.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfreadwrite.h>

#pragma comment(lib, "Mfplat.lib")
#pragma comment(lib, "Mf.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#endif

using amd::cvml::sample::utils::CamRes;

namespace amd {
namespace cvml {
namespace sample {
namespace utils {

#ifdef _WIN32
/**
 * camera supported media type
 */
struct MediaTypeInfo {
  GUID type;  /// Image type
  UINT32 width;
  UINT32 height;  /// Resolution
  UINT32 fps;     /// Frame rate
};

/**
 * Helper function to enumerate camera supported image type and resolution.
 *
 * @param camera_index: selected camera index
 * @return Enumeration of image type and resolution
 */
std::vector<MediaTypeInfo> EnumerateCameraImageTypes(int camera_index) {
  std::vector<MediaTypeInfo> formats;

  // Initialize Media Foundation
  HRESULT hr = MFStartup(MF_VERSION);
  if (FAILED(hr)) {
    std::cout << "Failed to initialize Media Foundation" << std::endl;
    return formats;
  }

  // Enumerate video capture devices
  IMFAttributes* pAttributes = nullptr;
  IMFActivate** ppDevices = nullptr;
  UINT32 devicecount = 0;

  hr = MFCreateAttributes(&pAttributes, 1);
  if (FAILED(hr) || pAttributes == nullptr) {
    std::cerr << "Failed to create source resolver" << std::endl;
    MFShutdown();
    return formats;
  }

  hr = pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
  if (FAILED(hr)) {
    std::cerr << "Failed to set device capture attribute" << std::endl;
    pAttributes->Release();
    MFShutdown();
    return formats;
  }

  hr = MFEnumDeviceSources(pAttributes, &ppDevices, &devicecount);
  pAttributes->Release();
  if (FAILED(hr) || ppDevices == nullptr || devicecount == 0 ||
      camera_index >= static_cast<int>(devicecount)) {
    std::cerr << "No valid video capture devices found" << std::endl;
    if (ppDevices) {
      for (UINT32 i = 0; i < devicecount; i++) {
        ppDevices[i]->Release();
      }
      CoTaskMemFree(ppDevices);
    }
    MFShutdown();
    return formats;
  }

  // Activate the selected device
  IMFMediaSource* pMediaSource = nullptr;
  hr = ppDevices[camera_index]->ActivateObject(IID_PPV_ARGS(&pMediaSource));
  for (UINT32 i = 0; i < devicecount; i++) {
    ppDevices[i]->Release();
  }
  CoTaskMemFree(ppDevices);
  if (FAILED(hr) || pMediaSource == nullptr) {
    std::cerr << "Failed to activate media source" << std::endl;
    MFShutdown();
    return formats;
  }

  IMFSourceReader* pSourceReader = nullptr;
  hr = MFCreateSourceReaderFromMediaSource(pMediaSource, nullptr, &pSourceReader);
  pMediaSource->Release();
  if (FAILED(hr) || pSourceReader == nullptr) {
    std::cerr << "Failed to create source reader" << std::endl;
    MFShutdown();
    return formats;
  }

  // Enumerate available formats
  DWORD dwStreamIndex = 0, mediaTypeIndex = 0;
  while (true) {
    IMFMediaType* pType = nullptr;
    hr = pSourceReader->GetNativeMediaType(dwStreamIndex, mediaTypeIndex, &pType);
    if (hr == MF_E_NO_MORE_TYPES) {
      mediaTypeIndex = 0;
      dwStreamIndex++;
      hr = pSourceReader->GetNativeMediaType(dwStreamIndex, mediaTypeIndex, &pType);
      if (hr == MF_E_INVALIDREQUEST || hr == MF_E_NO_MORE_TYPES) break;
    }
    if (FAILED(hr)) break;
    GUID subtype;
    hr = pType->GetGUID(MF_MT_SUBTYPE, &subtype);
    if (SUCCEEDED(hr)) {
      // Get the resolution
      UINT32 width = 0, height = 0;
      hr = MFGetAttributeSize(pType, MF_MT_FRAME_SIZE, &width, &height);
      if (SUCCEEDED(hr)) {
        // Get the frame rate
        UINT32 numerator = 0, denominator = 0;
        hr = MFGetAttributeRatio(pType, MF_MT_FRAME_RATE, &numerator, &denominator);
        if (SUCCEEDED(hr) && denominator != 0) {
          UINT32 fps = numerator / denominator;
          if (fps <= 30) {
            formats.push_back({subtype, width, height, fps});
          }
        }
      }
    }
    pType->Release();
    mediaTypeIndex++;
  }

  pSourceReader->Release();
  MFShutdown();

  return formats;
}
#endif

bool SetupCamera(int camera_index, const std::vector<CamRes>& res_list, cv::VideoCapture* camera) {
  // list certain API preferences before CAP_ANY to try them first
  // regardless of opencv's ordering
  static const int camera_api_preference[] = {
#ifdef _WIN32
      cv::CAP_DSHOW, cv::CAP_MSMF,
#else
      cv::CAP_V4L2,
#endif
      cv::CAP_ANY};

  if (camera == nullptr) {
    return false;
  }

#ifdef _WIN32
  std::vector<MediaTypeInfo> camera_format = EnumerateCameraImageTypes(camera_index);
#endif

  for (auto api : camera_api_preference) {
    try {
      camera->open(camera_index, api);
      if (camera->isOpened()) {
        break;
      }
    } catch (std::exception& e) {
      std::cout << "SetupCamera exception(" << api << "): " << e.what() << std::endl;
    }
  }

  if (camera->isOpened() != true) {
    std::cout << "Failed to open camera device with id:" << camera_index << std::endl;
    return false;
  }

  bool result = false;
#ifdef _WIN32
  int selected_index = -1;
  UINT32 highest_fps = 0;
  GUID selected_type = GUID_NULL;
#endif

  for (auto res : res_list) {
#ifdef _WIN32
    // Nested loop to compare camera_format resolution and match exactly, or as closely as possible
    for (size_t i = camera_format.size() - 1; i > 0; i--) {
      // Check for resolution match and either highest fps or MJPG format
      if (camera_format[i].width == res.width && camera_format[i].height == res.height &&
          camera_format[i].fps <= 30 &&
          (camera_format[i].fps > highest_fps ||
           (camera_format[i].type == MFVideoFormat_MJPG && selected_type != MFVideoFormat_MJPG))) {
        selected_index = static_cast<int>(i);
        highest_fps = camera_format[i].fps;
        selected_type = camera_format[i].type;
      }
    }
    if (selected_index >= 0) {
      break;
    }
  }
  if (selected_index >= 0) {
    camera->set(cv::CAP_PROP_FRAME_WIDTH, camera_format[selected_index].width);
    camera->set(cv::CAP_PROP_FRAME_HEIGHT, camera_format[selected_index].height);
    camera->set(cv::CAP_PROP_FPS, camera_format[selected_index].fps);
    if (selected_type == MFVideoFormat_MJPG) {  // select MJPG if it was available from previous for
                                                // loop, else use default in OpenCV camera API
      camera->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
      std::cout << "Selecting MJPG format." << std::endl;
    }
    auto w = camera->get(cv::CAP_PROP_FRAME_WIDTH);
    auto h = camera->get(cv::CAP_PROP_FRAME_HEIGHT);
    auto fps = camera->get(cv::CAP_PROP_FPS);
    if (w != camera_format[selected_index].width || h != camera_format[selected_index].height) {
      std::cout << "Camera doesn't support " << camera_format[selected_index].width << "x"
                << camera_format[selected_index].height << std::endl;
    } else {
      std::cout << "Camera enabled at " << w << "x" << h << "@" << fps << std::endl;
      result = true;
    }
#else
    camera->set(cv::CAP_PROP_FRAME_WIDTH, res.width);
    camera->set(cv::CAP_PROP_FRAME_HEIGHT, res.height);
    camera->set(cv::CAP_PROP_FPS, 30);
    auto w = camera->get(cv::CAP_PROP_FRAME_WIDTH);
    auto h = camera->get(cv::CAP_PROP_FRAME_HEIGHT);
    auto fps = camera->get(cv::CAP_PROP_FPS);
    if (w != res.width || h != res.height) {
      std::cout << "Camera doesn't support " << res.width << "x" << res.height << std::endl;
    } else {
      std::cout << "Camera enabled at " << w << "x" << h << "@" << fps << std::endl;
      result = true;
    }
#endif
  }

  if (!result) {
    std::cout << "No supported resolution for camera." << std::endl;
    camera->release();
  }
  std::cout << "Selected " << camera->get(cv::CAP_PROP_FRAME_WIDTH) << "x"
            << camera->get(cv::CAP_PROP_FRAME_HEIGHT) << "@" << camera->get(cv::CAP_PROP_FPS)
            << std::endl;
  return result;
}

}  // namespace utils
}  // namespace sample
}  // namespace cvml
}  // namespace amd
