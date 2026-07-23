# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

def _get_ep_paths() -> dict[str, str]:
    from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
        InitializeOptions,
        initialize
    )
    import winui3.microsoft.windows.ai.machinelearning as winml
    eps = {}
    with initialize(options = InitializeOptions.ON_NO_MATCH_SHOW_UI):
        pass
        catalog = winml.ExecutionProviderCatalog.get_default()
        providers = catalog.find_all_providers()
        for provider in providers:
            # print('provider name: ', provider.name, 'ready state: ', provider.ready_state)
            if provider.name != 'VitisAIExecutionProvider':
                continue
            if provider.ready_state == winml.ExecutionProviderReadyState.READY:
                eps[provider.name] = provider.library_path
                continue
            elif provider.ready_state == winml.ExecutionProviderReadyState.NOT_PRESENT:
                # print(f"Downloading and ensuring EP {provider.name} is ready. This may take a while...")
                result = provider.ensure_ready_async().get()
                if result.status != winml.ExecutionProviderReadyResultState.SUCCESS:
                    print(f"Failed to ensure EP {provider.name} is ready. Status: {result.status}")
            elif provider.ready_state == winml.ExecutionProviderReadyState.NOT_READY:
                # print(f"Ensuring EP {provider.name} is ready.")
                result = provider.ensure_ready_async().get()
                if result.status != winml.ExecutionProviderReadyResultState.SUCCESS:
                    print(f"Failed to ensure EP {provider.name} is ready. Status: {result.status}")
            else:
                print(f"EP {provider.name} is in unexpected state {provider.ready_state}")
            eps[provider.name] = provider.library_path
            # DO NOT call provider.try_register in python. That will register to the native env.
    return eps

if __name__ == "__main__":
    eps = _get_ep_paths()
    print(json.dumps(eps))