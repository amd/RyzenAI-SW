#pragma once
#ifndef MAIZE_UTIL_H
#define MAIZE_UTIL_H

#ifndef _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#endif

#include <filesystem>
#ifdef _WIN32
 #include <direct.h>
 #define getcwd _getcwd
#else
#include <unistd.h>
#endif
//#include <experimental/filesystem>

#include <sys/stat.h>

inline std::string path_separator() {
#ifdef _WIN32
  return "\\";
#else
  return "/";
#endif
}

inline std::string get_xclbin_string(std::string xclbin_whole_string) {
  std::string xclbin_fnm;
  char search_str1 = '\\';
  char search_str2 = '/';

  for (int i = xclbin_whole_string.length() - 1; i >= 0; --i) {
    if (xclbin_whole_string[i] == search_str1 || xclbin_whole_string[i] == search_str2) {
      return xclbin_fnm;
    }
    xclbin_fnm = xclbin_whole_string[i] + xclbin_fnm;
  }

  return xclbin_whole_string;
}

inline std::string get_dir(std::string xclbin_whole_string) {
  std::string dir_name;
  char search_str1 = '\\';
  char search_str2 = '/';
  auto pos1 = xclbin_whole_string.find_last_of(search_str1);
  auto pos2 = xclbin_whole_string.find_last_of(search_str2);

  if (pos1 != std::string::npos && pos2 != std::string::npos) {
    if (pos1 > pos2) {
      dir_name = xclbin_whole_string.substr(0, pos1);
      return dir_name;
    } else {
      dir_name = xclbin_whole_string.substr(0, pos2);
      return dir_name;
    }
  } else if (pos2 != std::string::npos) {
    dir_name = xclbin_whole_string.substr(0, pos2);
    return dir_name;
  } else if (pos1 != std::string::npos) {
    dir_name = xclbin_whole_string.substr(0, pos1);
    return dir_name;
  }
  return "";
}

inline std::string get_env_var_dir() {
  // Supported options for env var XLNX_VART_FIRMWARE: Directory path or empty
  std::string env_var = "XLNX_VART_FIRMWARE";
  std::string default_val = "";
#ifdef _WIN32
  char* value = nullptr;
  size_t size = 0;
  errno_t err = _dupenv_s(&value, &size, env_var.c_str());
  std::string result = (!err && (value != nullptr)) ? std::string{value} : default_val;
  free(value);
#else
  const char* value = std::getenv(env_var.c_str());
  std::string result = (value != nullptr) ? std::string(value) : default_val;
#endif
  if (result != "") {
    if(result.find(".xclbin") != std::string::npos){
      std::string dir_name = get_dir(result);
      return dir_name + path_separator();
    }
    else
      return result + path_separator();
  }
  else
    return result;
}

inline bool isFileDirExists(std::string dirFilePath) { 
    const char* path_c = dirFilePath.c_str();

    //const char* folder;
    // folder = "C:\\Users\\SaMaN\\Desktop\\Ppln";
    //folder = "/tmp";
    //struct stat sb;

   // if (stat(path_c, &sb) == 0 && S_ISDIR(sb.st_mode)) {
     // printf("YES\n");
    //} else {
     // printf("NO\n");
    //}


   // bool file_exists(char* filename) {
      struct stat buffer;
      return (stat(path_c, &buffer) == 0);
    //}

}

inline std::string getXclbinJsonPath(std::string extractFNameJsonXclbin, std::string extractXclbin) {
  std::string whole_xclbin_path;
  std::string dir_xclbin;
  std::string xclbin_fnm = get_xclbin_string(extractXclbin);
  std::string env_var_directory = get_env_var_dir();

  char cwd[4096];//4096 being PATH_MAX


  if (!env_var_directory.empty()) {
    whole_xclbin_path = env_var_directory + xclbin_fnm;
    dir_xclbin = env_var_directory;
  }

  //find a replacement filesystem 
  // - to check if given dir or file pathe exists
  // - get current directory path
  // - 

  


  if (!isFileDirExists(whole_xclbin_path)) {
    if (isFileDirExists(extractXclbin)) {
      whole_xclbin_path = extractXclbin;
      dir_xclbin = get_dir(extractXclbin);
    } else {
      if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::string cur_dir(cwd);
        whole_xclbin_path = cur_dir + path_separator() + xclbin_fnm;
      }
      // std::filesystem::path current_dir = std::filesystem::current_path();
      // whole_xclbin_path = current_dir.string() + path_separator() + xclbin_fnm;

      if (!isFileDirExists(whole_xclbin_path)) {
        whole_xclbin_path = "C:\\Windows\\System32\\AMD" + path_separator() + xclbin_fnm;
        dir_xclbin = "C:\\Windows\\System32\\AMD" + path_separator();
      } else {
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
          std::string cur_dir(cwd);
          dir_xclbin = cur_dir + path_separator();
        }
        // dir_xclbin = current_dir.string() + path_separator();
      }
    }
  }



  /* if (!std::filesystem::exists(whole_xclbin_path)) {
    if (std::filesystem::exists(extractXclbin)) {
      whole_xclbin_path = extractXclbin;
      dir_xclbin = get_dir(extractXclbin);
    } else {
      
      if (getcwd(cwd, sizeof(cwd))!=NULL) {
        std::string cur_dir(cwd);
        whole_xclbin_path = cur_dir + path_separator() + xclbin_fnm;
      }
        //std::filesystem::path current_dir = std::filesystem::current_path();
      //whole_xclbin_path = current_dir.string() + path_separator() + xclbin_fnm;

      if (!std::filesystem::exists(whole_xclbin_path)) {
        whole_xclbin_path = "C:\\Windows\\System32\\AMD" + path_separator() + xclbin_fnm;
        dir_xclbin = "C:\\Windows\\System32\\AMD" + path_separator();
      } else {
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
          std::string cur_dir(cwd);
            dir_xclbin = cur_dir + path_separator();
        }
        //dir_xclbin = current_dir.string() + path_separator();
      }
    }
  }*/



  if (!isFileDirExists(whole_xclbin_path)) {
    throw std::runtime_error("Provided xclbin file does not exist");
  }

  // if (!std::filesystem::exists(whole_xclbin_path)) {
    //throw std::runtime_error("Provided xclbin file does not exist");
  //}

  if (extractFNameJsonXclbin.find(".xclbin") != std::string::npos) {
    return whole_xclbin_path;
  }

  std::string whole_json_path = "";
  if (extractFNameJsonXclbin.find(".json") != std::string::npos) {
    std::string jsonName = get_xclbin_string(extractFNameJsonXclbin);
    whole_json_path = dir_xclbin + path_separator() + jsonName;
    if (!isFileDirExists(whole_json_path)) {
     // if (!std::filesystem::exists(whole_json_path)) {
      throw std::runtime_error(
          jsonName+" provided json file does not exist. xclbin and json files should be in same directory "+dir_xclbin);
    }
    return whole_json_path;
  }

  return extractFNameJsonXclbin;
}
#endif
