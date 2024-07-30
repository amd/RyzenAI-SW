#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

function Add-PathToVar {
    # Add a path-like string to an environment variable if it's not already
    # present in the variable. By default, values are added to the front.
    # Use the optional flag -Append to add to the end.
    #
    # Note: PS cmdlets have strict name guidelines which is how this name is
    # defined.
    #
    # Example usage:
    #  - Add "$PWD\ops\python" to the PYTHONPATH at the end:
    #       Add-PathToVar "$PWD\ops\python" PYTHONPATH -Append
    param (
        # path to add
        [Parameter(Mandatory)]
        [string] $Path,
        # var to add to
        [Parameter(Mandatory)]
        [string] $Var,
        # by default, the new value is prepended to the front. Pass -Append
        # to add it to the end
        [Parameter(Mandatory=$false)]
        [Switch] $Append
    )

    # get the current value of the variable, if it exists
    if (Test-Path -Path Env:${Var}) {
        $envPaths = (Get-Content -Path Env:${Var}) -split [IO.Path]::PathSeparator
    } else {
        $envPaths = @()
    }

    if ($envPaths -notcontains $Path) {
        if ($Append) {
            $envPaths = $envPaths + $Path | Where-Object { $_ }
        } else {
            $envPaths = ,$Path + $envPaths | Where-Object { $_ }
        }
        Set-Content -Path Env:${Var} -Value ($envPaths -join [IO.Path]::PathSeparator)
    }
}
