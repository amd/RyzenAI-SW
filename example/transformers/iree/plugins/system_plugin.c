// The system plugin for IREE with matmul_f32() kernels.
// This is customized from the example at
// https://github.com/openxla/iree/blob/main/samples/custom_dispatch/cpu/plugin/system_plugin.c
// for matmul_f32_impl(), which is defined in matmul.cpp.
// This module is built into system_plugin.dll.

#include <inttypes.h>
#include <stdio.h>

#include "executable_plugin.h"
#include "matmul.h"

typedef struct {
  iree_hal_executable_plugin_allocator_t host_allocator;
  FILE *file;
} system_plugin_t;

static int matmul_f32_workgroup(void *params_ptr, void *context,
                                void *reserved) {
  system_plugin_t *plugin = (system_plugin_t *)context;
  typedef struct {
    const float *restrict binding0;
    size_t binding0_offset;
    const float *restrict binding1;
    size_t binding1_offset;
    float *restrict binding2;
    size_t binding2_offset;
    size_t d0;
    size_t d1;
    size_t d2;
    const uint64_t *restrict processor_data;
    uint32_t processor_id;
  } params_t;
  const params_t *params = (const params_t *)params_ptr;

#ifdef IREE_PLUGINS_SYSTEM_PLUIN_C_LOG1
  fprintf(plugin->file, "processor_id=%u\n", params->processor_id);
  if (params->processor_data) {
    fprintf(plugin->file, "processor_data[0]=%" PRIX64 "\n",
            params->processor_data[0]);
  }

  for (size_t i = 0; i < params->d1; ++i) {
    params->binding2[params->binding2_offset + i] =
        params->binding0[params->binding0_offset + i] *
        params->binding1[params->binding2_offset + i];
    fprintf(plugin->file, "mul[%zu](%g * %g = %g)\n", i,
            params->binding0[params->binding0_offset + i],
            params->binding1[params->binding1_offset + i],
            params->binding2[params->binding2_offset + i]);
  }

  printf("size_0: %zu; size_1: %zu; size_2: %zu\n", params->d0, params->d1,
         params->d2);

  printf("Input A (LHS): \n");
  printf("Size: %zu x %zu ", params->d0, params->d1);
  for (size_t i = 0; i < params->d0; ++i) {
    printf("\n");
    printf("a[%zu]=", i);
    for (size_t j = 0; j < params->d1; ++j) {
      fprintf(plugin->file, "%g\t",
              params->binding0[params->binding0_offset + (i * params->d1) + j]);
    }
  }
  printf("\n \n");
  printf("Input B (RHS): \n");
  printf("Size: %zu x %zu ", params->d1, params->d2);
  for (size_t i = 0; i < params->d1; ++i) {
    printf("\n");
    printf("b[%zu]=", i);
    for (size_t j = 0; j < params->d2; ++j) {
      fprintf(plugin->file, "%g\t",
              params->binding1[params->binding1_offset + (i * params->d2) + j]);
    }
  }
  printf("\n \n");
#endif

  matmul_f32_impl(&params->binding0[params->binding0_offset],
                  &params->binding1[params->binding1_offset],
                  &params->binding2[params->binding2_offset], params->d0,
                  params->d1, params->d2);

#ifdef IREE_PLUGINS_SYSTEM_PLUIN_C_LOG2
  printf("Output size: %zu x %zu", params->d0, params->d2);
  for (size_t i = 0; i < params->d0; ++i) {
    printf("\n");
    printf("out[%zu]=", i);
    for (size_t j = 0; j < params->d2; ++j) {
      fprintf(plugin->file, "%g\t",
              params->binding2[params->binding2_offset + (i * params->d2) + j]);
    }
  }
  printf("\n \n");
#endif
  return 0;
}

// Called once for each plugin load and paired with a future call to unload.
// Even in standalone mode we could allocate using environment->host_allocator,
// set an out_self pointer, and parse parameters but here in system mode we can
// do whatever we want.
//
// If any state is required it should be allocated and stored in |out_self|.
// This self value will be passed to all future calls related to the particular
// instance. Note that there may be multiple instances of a plugin in any
// particular process and this must be thread-safe.
static iree_hal_executable_plugin_status_t system_plugin_load(
    const iree_hal_executable_plugin_environment_v0_t *environment,
    size_t param_count, const iree_hal_executable_plugin_string_pair_t *params,
    void **out_self) {
  // Allocate the plugin state.
  system_plugin_t *plugin = NULL;
  iree_hal_executable_plugin_status_t status =
      iree_hal_executable_plugin_allocator_malloc(
          environment->host_allocator, sizeof(*plugin), (void **)&plugin);
  if (status)
    return status;
  plugin->host_allocator = environment->host_allocator;

  // "Open standard out" simulating us doing some syscalls or other expensive
  // stateful/side-effecting things.
  plugin->file = stdout;

  // Pass back the plugin instance that'll be passed to resolve.
  *out_self = plugin;
  return iree_hal_executable_plugin_ok_status();
}

// Called to free any plugin state allocated in load.
static void system_plugin_unload(void *self) {
  system_plugin_t *plugin = (system_plugin_t *)self;
  iree_hal_executable_plugin_allocator_t host_allocator =
      plugin->host_allocator;

  // "Close standard out" simulating us doing some syscalls and other expensive
  // stateful/side-effecting things.
  fflush(plugin->file);
  plugin->file = NULL;

  // Free the plugin state using the same allocator it came from.
  iree_hal_executable_plugin_allocator_free(host_allocator, plugin);
}

// Called to resolve one or more imports by symbol name.
// See the plugin API header for more information. Note that some of the
// functions may already be resolved and some may be optional.
static iree_hal_executable_plugin_status_t system_plugin_resolve(
    void *self, const iree_hal_executable_plugin_resolve_params_v0_t *params,
    iree_hal_executable_plugin_resolution_t *out_resolution) {
  system_plugin_t *plugin = (system_plugin_t *)self;
  *out_resolution = 0;
  bool any_required_not_found = false;
  for (size_t i = 0; i < params->count; ++i) {
    if (params->out_fn_ptrs[i])
      continue;
    const char *symbol_name = params->symbol_names[i];
    bool is_optional =
        iree_hal_executable_plugin_import_is_optional(symbol_name);
    if (is_optional)
      ++symbol_name;
    if (iree_hal_executable_plugin_strcmp(symbol_name, "accel_matmul_f32") ==
        0) {
      params->out_fn_ptrs[i] = matmul_f32_workgroup;
      params->out_fn_contexts[i] = plugin; // passing plugin to each import call
    } else {
      if (is_optional) {
        *out_resolution |=
            IREE_HAL_EXECUTABLE_PLUGIN_RESOLUTION_MISSING_OPTIONAL;
      } else {
        any_required_not_found = true;
      }
    }
  }
  return any_required_not_found
             ? iree_hal_executable_plugin_status_from_code(
                   IREE_HAL_EXECUTABLE_PLUGIN_STATUS_NOT_FOUND)
             : iree_hal_executable_plugin_ok_status();
}

// Exported on the shared library and used by the runtime to query the plugin
// interface. When statically linking the plugin this is just a function that
// can be called and can have any name to allow for multiple plugins. When
// dynamically linking the exported symbol must be exactly this with no C++
// name mangling.
IREE_HAL_EXECUTABLE_PLUGIN_EXPORT const iree_hal_executable_plugin_header_t **
iree_hal_executable_plugin_query(
    iree_hal_executable_plugin_version_t max_version, void *reserved) {
  static const iree_hal_executable_plugin_header_t header = {
      // Declares what library version is present: newer runtimes may support
      // loading older plugins but newer plugins cannot load on older runtimes.
      .version = IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST,
      // Name and description are used for tracing/logging/diagnostics.
      .name = "sample_system",
      .description = "matmul system plugin sample (system_plugin.c)",
      .features = 0,
      // Let the runtime know what sanitizer this plugin was compiled with.
      .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND,
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header = &header,
      .load = system_plugin_load,
      .unload = system_plugin_unload,
      .resolve = system_plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t **)&plugin
             : NULL;
}
