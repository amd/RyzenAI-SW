#include <fstream>

#include "mobile_net_v2/model.hpp"
#include "processing/executor.hpp"
#include "processing/pipeline.hpp"
#include "resnet50/model.hpp"
#include "retinaface/model.hpp"
#include "segmentation/model.hpp"
#include "util/fs.hpp"
#include "yolovx/model.hpp"

std::string read_all(const std::string& file_path) {
  std::ifstream f{file_path};
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void start(const Config& config) {
  SessionManager::get_instance().set_singleton(true);
  std::vector<std::shared_ptr<AsyncTask>> tasks;
  auto gui_task = std::make_shared<GuiTask>();
  int split_channel_matrix_size_square{0};
  {
    CONFIG_GET(config, Config, gui_config, "screen");
    CONFIG_GET(gui_config, int, split_channel_matrix_size,
               "split_channel_matrix_size")
    split_channel_matrix_size_square =
        split_channel_matrix_size * split_channel_matrix_size;
    gui_task->init(gui_config);
  }
  CHECK(split_channel_matrix_size_square >= 1)
  auto x = config["pipelines"];
  CONFIG_GET_ARRAY(config, Config, pipeline_configs, "pipelines");
  for (auto& pipeline_config_pair : pipeline_configs.items()) {
    auto piepeline_config = pipeline_config_pair.value();
    auto pipeline_tasks = create_model_pipeline(piepeline_config, gui_task);
    {
      CONFIG_GET(piepeline_config, Config, sort_config, "sort")
      CONFIG_GET(sort_config, int, channel_matrix_id, "channel_matrix_id");
      CHECK(channel_matrix_id >= 0)
      CHECK(channel_matrix_id < split_channel_matrix_size_square)
    }
    tasks.insert(tasks.end(), pipeline_tasks.begin(), pipeline_tasks.end());
  }
  tasks.push_back(std::dynamic_pointer_cast<AsyncTask>(gui_task));
  ThreadExcutor tjread_executor{};
  PRINT("Running ... \nClose window to stop ")
  tjread_executor.run(tasks);
  tjread_executor.wait();
}

int main(int argc, char* argv[]) {
  try {
    GLOBAL_APP_NAME = "ipu_modelx4_demo";
    if (argc <= 1) {
#include "help_info.inl"
      return 0;
    }
    std::string config_path = argv[1];
    CHECK_WITH_INFO(is_file(config_path), config_path)
    CHECK_WITH_INFO(check_extension(config_path, ".json"), config_path)
    auto config_content = read_all(config_path);
    auto config = Config::parse(config_content);
    CONFIG_GET_ARRAY(config, Config, pipelines_config, "pipelines")
    CHECK_WITH_INFO(!pipelines_config.empty(), "pipeline config size zero!!!")
    start(config);
  } catch (MyExceptoin& e) {
    std::cout << "DemoExcption: " << e.what() << "\n";
  } catch (std::exception& e) {
    std::cout << "Excption: " << e.what() << "\n";
  }
  return 0;
}